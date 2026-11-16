using System.Buffers;
using System.Diagnostics;
using System.Numerics.Tensors;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Engine;

/// <summary>Sampling for <see cref="CloverBlockDiffusion"/> (HF <c>sampling_clover_moe.predict_with_confidence</c>).</summary>
/// <param name="Temperature">0 or less = greedy (argmax).</param>
/// <param name="TopK">0 = off.</param>
/// <param name="TopP">1 = off.</param>
/// <param name="Seed">Seed of the sampling RNG; null = nondeterministic.</param>
public readonly record struct CloverBlockSampling(float Temperature = 0f, int TopK = 0, float TopP = 1f, int? Seed = null)
{
    /// <summary>Whether the prediction is the argmax.</summary>
    public bool IsGreedy => !(Temperature > 0f);
}

/// <summary>One denoising step of a block: the positions filled this step (ascending) and their tokens.</summary>
/// <param name="Filled">Block positions committed this step, ascending.</param>
/// <param name="Tokens">The token placed at each of <paramref name="Filled"/>.</param>
/// <param name="Fallback">True when no pending position passed the threshold and the single most confident one was filled.</param>
public readonly record struct CloverDenoiseStep(int[] Filled, int[] Tokens, bool Fallback);

/// <summary>A resolved block and how it was resolved.</summary>
/// <param name="Tokens">The <c>BlockSize</c> resolved tokens (no mask left).</param>
/// <param name="Steps">The denoising steps, in order (one preview forward each).</param>
/// <param name="PreviewTicks">Stopwatch ticks spent in the preview forwards.</param>
/// <param name="SelectTicks">Stopwatch ticks spent in the host-side prediction / confidence / transfer.</param>
public sealed record CloverResolvedBlock(int[] Tokens, IReadOnlyList<CloverDenoiseStep> Steps, long PreviewTicks, long SelectTicks);

/// <summary>Result of <see cref="CloverBlockDiffusion.Generate"/>.</summary>
/// <param name="Response">The committed continuation (EOS included when one stopped it).</param>
/// <param name="Blocks">Every resolved block, in order.</param>
/// <param name="StoppedOnEos">Whether an EOS id stopped the generation.</param>
public sealed record CloverGenerationResult(int[] Response, IReadOnlyList<CloverResolvedBlock> Blocks, bool StoppedOnEos);

/// <summary>
/// Clover's LowConfidence block-diffusion decoding (#1828, stage 4 of #1794): DiffuSpace <c>CloverMoeModelLM.generate</c>, on top of the
/// bidirectional block preview (<see cref="ICloverBlockPreview"/>).
/// </summary>
/// <remarks>
/// <para>
/// The prompt is committed causally. Each block of <see cref="BlockSize"/> positions starts all <c>&lt;|mask|&gt;</c>; every step runs one
/// preview forward on the unchanged committed state, predicts every position (argmax, or the HF sampler) with its confidence = the
/// full-vocabulary <c>T = 1</c> softmax probability of the prediction, and fills every still-masked position whose confidence is
/// <b>strictly</b> above <see cref="Threshold"/> (HF <c>confidence &gt; threshold</c>); when none is, the single most confident pending
/// position (the first on ties). Filled positions are frozen. The resolved block is then committed with an ordinary causal forward, unless
/// it ended the generation (an EOS, or the token budget). The first generated token comes from preview row 0, not from the prefill.
/// </para>
/// <para>
/// This is not <see cref="DiffusionTextGenerator"/>: that one is DiffusionGemma's cacheless canvas with a hybrid mask, re-masking and a
/// step budget, recomputing the prompt each step. Clover keeps the prefix in its KV cache and recurrent state, never re-masks a filled
/// position, and its preview is a separate model entry point.
/// </para>
/// <para>
/// Host scratch: one vocabulary-sized float row per worker for the softmax denominators (the logits themselves are the preview's
/// <c>[T, vocab]</c> F32 result, 30.3 MiB at T = 32 on the real model). No device storage. The response penalties and length / think-end
/// controls of the official generation config (stage 6) are not applied.
/// </para>
/// </remarks>
public sealed class CloverBlockDiffusion
{
    /// <summary>HF <c>denoising_config.threshold</c> of the official config.</summary>
    public const float DefaultThreshold = 0.95f;

    private readonly IModel _model;
    private readonly ICloverBlockPreview _preview;
    private readonly int _maskTokenId;
    private readonly CloverBlockSampling _sampling;
    private readonly Random? _rng;

    /// <summary>Creates a decoder over a Clover model whose weights carry the preview tensors.</summary>
    /// <param name="model">The Clover model (a host implementing <see cref="ICloverBlockPreview"/>).</param>
    /// <param name="threshold">Fill threshold in [0, 1] (HF default 0.95).</param>
    /// <param name="blockSize">Block length; 0 = the model's <c>block_size</c>. Must divide it (HF rule).</param>
    /// <param name="sampling">Prediction rule; default greedy.</param>
    public CloverBlockDiffusion(IModel model, float threshold = DefaultThreshold, int blockSize = 0, CloverBlockSampling sampling = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        var clover = model.Config.Clover ?? throw new ArgumentException("Block-diffusion decoding is a Clover (clover_moe) mode.", nameof(model));
        if (model is not ICloverBlockPreview { SupportsBlockPreview: true } preview)
            throw new NotSupportedException("This model has no Clover block preview (an official DiffuSpace Clover GGUF on the CPU or Vulkan host is required).");
        if (!(threshold >= 0f && threshold <= 1f)) throw new ArgumentOutOfRangeException(nameof(threshold), "threshold must be in [0, 1].");
        if (blockSize == 0) blockSize = clover.BlockSize;
        if (blockSize <= 0 || clover.BlockSize % blockSize != 0)
            throw new ArgumentOutOfRangeException(nameof(blockSize), $"block size must be positive and divide the model's block_size ({clover.BlockSize}).");
        if ((uint)clover.MaskTokenId >= (uint)model.Config.VocabSize)
            throw new ArgumentException("The model declares no <|mask|> token.", nameof(model));
        if (!sampling.IsGreedy && (sampling.TopP <= 0f || sampling.TopP > 1f))
            throw new ArgumentOutOfRangeException(nameof(sampling), "top_p must be in (0, 1].");
        _model = model;
        _preview = preview;
        _maskTokenId = clover.MaskTokenId;
        Threshold = threshold;
        BlockSize = blockSize;
        _sampling = sampling;
        _rng = sampling.IsGreedy ? null : sampling.Seed is { } seed ? new Random(seed) : new Random();
    }

    /// <summary>Fill threshold (strict).</summary>
    public float Threshold { get; }

    /// <summary>Block length.</summary>
    public int BlockSize { get; }

    /// <summary>KV positions a generation of <paramref name="maxNewTokens"/> after <paramref name="promptLength"/> needs: every block is
    /// previewed at full length, also the last one (HF).</summary>
    public int RequiredKvLength(int promptLength, int maxNewTokens)
        => checked(promptLength + (maxNewTokens + BlockSize - 1) / BlockSize * BlockSize);

    /// <summary>Commits tokens causally at <paramref name="startPosition"/> (the prompt, or a resolved block).</summary>
    public void Commit(ReadOnlySpan<int> tokens, int startPosition, IKvCache kvCache)
    {
        if (tokens.IsEmpty) return;
        int[] positions = ArrayPool<int>.Shared.Rent(tokens.Length);
        try
        {
            for (int i = 0; i < tokens.Length; i++) positions[i] = startPosition + i;
            using var logits = _model.Forward(tokens, positions.AsSpan(0, tokens.Length), deviceId: -1, kvCache);
        }
        finally { ArrayPool<int>.Shared.Return(positions); }
    }

    /// <summary>Resolves one all-mask block after the committed prefix of length <paramref name="committedLength"/>.</summary>
    /// <param name="kvCache">The committed KV cache (its length must equal <paramref name="committedLength"/>).</param>
    /// <param name="committedLength">Committed prefix length = the block's first position.</param>
    /// <param name="onStep">Optional callback after each step with the live block (mask id at unresolved positions).</param>
    public CloverResolvedBlock ResolveBlock(IKvCache kvCache, int committedLength, Action<ReadOnlySpan<int>, CloverDenoiseStep>? onStep = null)
    {
        int B = BlockSize, vocab = _model.Config.VocabSize;
        var block = new int[B];
        Array.Fill(block, _maskTokenId);
        var pending = new bool[B];
        Array.Fill(pending, true);
        int pendingCount = B;
        var prediction = new int[B];
        var confidence = new float[B];
        var steps = new List<CloverDenoiseStep>();
        long previewTicks = 0, selectTicks = 0;

        for (int step = 0; step < B && pendingCount > 0; step++)
        {
            long t0 = Stopwatch.GetTimestamp();
            using ITensor logits = _preview.ForwardBlockPreview(block, committedLength, kvCache, gdnState: null, deviceId: -1);
            long t1 = Stopwatch.GetTimestamp();
            Predict(logits, vocab, pending, prediction, confidence);

            int filledCount = 0;
            for (int i = 0; i < B; i++)
                if (pending[i] && confidence[i] > Threshold) filledCount++;
            bool fallback = filledCount == 0;
            int[] filled;
            if (fallback)
            {
                int best = -1;
                float bestConf = float.NegativeInfinity;
                for (int i = 0; i < B; i++)
                    if (pending[i] && (best < 0 || confidence[i] > bestConf)) { best = i; bestConf = confidence[i]; }
                filled = [best];
            }
            else
            {
                filled = new int[filledCount];
                for (int i = 0, n = 0; i < B; i++)
                    if (pending[i] && confidence[i] > Threshold) filled[n++] = i;
            }
            var tokens = new int[filled.Length];
            for (int n = 0; n < filled.Length; n++)
            {
                int pos = filled[n];
                tokens[n] = prediction[pos];
                block[pos] = prediction[pos];
                pending[pos] = false;
            }
            pendingCount -= filled.Length;
            var record = new CloverDenoiseStep(filled, tokens, fallback);
            steps.Add(record);
            previewTicks += t1 - t0;
            selectTicks += Stopwatch.GetTimestamp() - t1;
            onStep?.Invoke(block, record);
        }
        if (pendingCount > 0) throw new InvalidOperationException("Denoising step limit reached with unresolved masks.");
        return new CloverResolvedBlock(block, steps, previewTicks, selectTicks);
    }

    /// <summary>
    /// HF <c>generate</c> end to end: prefill, blocks, EOS stop after full block resolution, causal commit between blocks. The KV cache
    /// must be empty and hold <see cref="RequiredKvLength"/> positions; the model-owned recurrent state is reset first.
    /// </summary>
    public CloverGenerationResult Generate(ReadOnlySpan<int> prompt, int maxNewTokens, IKvCache kvCache, IReadOnlyCollection<int> eosTokenIds)
    {
        if (prompt.IsEmpty) throw new ArgumentException("The prompt must have at least one token.", nameof(prompt));
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        if (kvCache.CurrentLength != 0) throw new ArgumentException("The KV cache must be empty.", nameof(kvCache));
        _model.ResetSequenceState();
        var response = new List<int>(maxNewTokens);
        var blocks = new List<CloverResolvedBlock>();
        if (maxNewTokens == 0) return new CloverGenerationResult([], blocks, false);

        Commit(prompt, 0, kvCache);
        int committed = prompt.Length;
        bool stopped = false;
        while (response.Count < maxNewTokens && !stopped)
        {
            int remaining = Math.Min(BlockSize, maxNewTokens - response.Count);
            var resolved = ResolveBlock(kvCache, committed);
            blocks.Add(resolved);
            for (int i = 0; i < remaining; i++)
            {
                int token = resolved.Tokens[i];
                response.Add(token);
                if (eosTokenIds.Contains(token)) { stopped = true; break; }
            }
            if (response.Count < maxNewTokens && !stopped)
            {
                Commit(resolved.Tokens.AsSpan(0, remaining), committed, kvCache);
                committed += remaining;
            }
        }
        return new CloverGenerationResult(response.ToArray(), blocks, stopped);
    }

    /// <summary>Per pending row: the prediction and its confidence (T = 1 softmax probability); non-pending rows get -inf.</summary>
    private unsafe void Predict(ITensor logits, int vocab, bool[] pending, int[] prediction, float[] confidence)
    {
        int rows = pending.Length;
        if (logits.Shape[0] < rows || logits.Shape[logits.Shape.Rank - 1] != vocab)
            throw new InvalidOperationException($"Preview returned [{logits.Shape[0]}, {logits.Shape[logits.Shape.Rank - 1]}] logits; expected [{rows}, {vocab}].");
        float* basePtr = (float*)logits.DataPointer;
        if (_rng is null)
        {
            nint p = (nint)basePtr;
            Parallel.For(0, rows, () => ArrayPool<float>.Shared.Rent(vocab), (r, _, scratch) =>
            {
                if (!pending[r]) { confidence[r] = float.NegativeInfinity; prediction[r] = -1; return scratch; }
                var row = new ReadOnlySpan<float>((float*)p + (long)r * vocab, vocab);
                int arg = TensorPrimitives.IndexOfMax(row);   // the first maximal index, as torch.argmax
                prediction[r] = arg;
                confidence[r] = ProbabilityOf(row, arg, scratch.AsSpan(0, vocab));
                return scratch;
            }, scratch => ArrayPool<float>.Shared.Return(scratch));
            return;
        }

        // Sampled prediction (HF predict_with_confidence): temperature, top-k, top-p, multinomial; the confidence is still the T = 1
        // probability of the sampled token. Serial so the RNG stream is deterministic for a seed.
        float[] buf = ArrayPool<float>.Shared.Rent(vocab);
        try
        {
            for (int r = 0; r < rows; r++)
            {
                if (!pending[r]) { confidence[r] = float.NegativeInfinity; prediction[r] = -1; continue; }
                var row = new ReadOnlySpan<float>(basePtr + (long)r * vocab, vocab);
                int tok = Sample(row, buf.AsSpan(0, vocab));
                prediction[r] = tok;
                confidence[r] = ProbabilityOf(row, tok, buf.AsSpan(0, vocab));
            }
        }
        finally { ArrayPool<float>.Shared.Return(buf); }
    }

    private static float ProbabilityOf(ReadOnlySpan<float> row, int token, Span<float> scratch)
    {
        float max = TensorPrimitives.Max(row);
        TensorPrimitives.Add(row, -max, scratch);
        TensorPrimitives.Exp(scratch, scratch);
        float sum = TensorPrimitives.Sum(scratch);
        return MathF.Exp(row[token] - max) / sum;
    }

    private int Sample(ReadOnlySpan<float> row, Span<float> scratch)
    {
        int vocab = row.Length;
        float invT = 1f / _sampling.Temperature;
        TensorPrimitives.Multiply(row, invT, scratch);
        if (_sampling.TopK > 0 && _sampling.TopK < vocab)
        {
            // cutoff = the k-th largest scaled logit; everything strictly below it is removed (HF masked_fill(scaled < cutoff)).
            float[] copy = ArrayPool<float>.Shared.Rent(vocab);
            try
            {
                scratch.CopyTo(copy);
                Array.Sort(copy, 0, vocab);
                float cutoff = copy[vocab - _sampling.TopK];
                for (int i = 0; i < vocab; i++) if (scratch[i] < cutoff) scratch[i] = float.NegativeInfinity;
            }
            finally { ArrayPool<float>.Shared.Return(copy); }
        }
        float max = TensorPrimitives.Max(scratch);
        TensorPrimitives.Add(scratch, -max, scratch);
        TensorPrimitives.Exp(scratch, scratch);
        float sum = TensorPrimitives.Sum(scratch);
        if (_sampling.TopP < 1f)
        {
            // Keep the smallest prefix of the descending-probability order whose mass before each kept token is <= top_p.
            int[] order = ArrayPool<int>.Shared.Rent(vocab);
            float[] keys = ArrayPool<float>.Shared.Rent(vocab);
            try
            {
                for (int i = 0; i < vocab; i++) { order[i] = i; keys[i] = -scratch[i]; }
                Array.Sort(keys, order, 0, vocab);
                float cum = 0f, kept = 0f;
                for (int n = 0; n < vocab; n++)
                {
                    int i = order[n];
                    float p = scratch[i] / sum;
                    if (cum > _sampling.TopP) scratch[i] = 0f; else kept += scratch[i];
                    cum += p;
                }
                sum = kept;
            }
            finally
            {
                ArrayPool<int>.Shared.Return(order);
                ArrayPool<float>.Shared.Return(keys);
            }
        }
        double u = _rng!.NextDouble() * sum, acc = 0;
        int last = 0;
        for (int i = 0; i < vocab; i++)
        {
            if (scratch[i] <= 0f) continue;
            last = i;
            acc += scratch[i];
            if (acc >= u) return i;
        }
        return last;
    }
}
