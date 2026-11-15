using System.Runtime.CompilerServices;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cpu.Threading;

namespace DotLLM.Models.Architectures;

/// <summary>
/// The Gated-DeltaNet token mixer of Clover's bidirectional block preview (#1815): DiffuSpace <c>modeling_clover_moe.py</c>
/// <c>bd_gdn_conv1d</c> + <c>bd_gdn_local_memory</c>, run over one block after a committed causal prefix. Reads the prefix's conv and
/// recurrent state and never writes either.
/// </summary>
/// <remarks>
/// <para>Per block of <c>T</c> tokens, against the causal mixer (<see cref="GdnTokenMixer"/>):</para>
/// <list type="number">
///   <item>Conv: the same causal 4-tap conv over <c>[clean conv state | block]</c>, PLUS a right conv
///         <c>out[t] += x[t+d] * w_right[c, d-1]</c> for <c>d = 1..min(dConv, T)-1</c> and <c>t+d &lt; T</c> (raw pre-conv projections,
///         in-block only), then SiLU. The conv state is not advanced.</item>
///   <item>Mixer: the delta-rule recurrence is replaced by a per-value-head softmax over the block's <c>T</c> keys plus one history bucket.
///         Block logit <c>logit_scale[h] * &lt;l2n(q_i), l2n(k_j)&gt;</c> (no 1/sqrt(d), no beta, every <c>j</c>, before and after
///         <c>i</c>); history logit <c>history_bias[h] + history_c[h] * ln(T / 32)</c> (32 is HF's fixed
///         <c>gdn_reference_block_size</c>, not the configured block size); history value <c>exp(g_i) * (S_clean^T l2n(q_i) / sqrt(dS))</c>,
///         a read-only readout of the clean recurrent state decayed once by the token's own gate.
///         <c>out_i = w_hist * history_i + sum_j w_ij * v_j</c>.</item>
///   <item>The per-head gated RMSNorm and <c>ssm_out</c> are the causal mixer's (Clover gates with SiLU).</item>
/// </list>
/// <para>V-head order is the file's tiled one, so value head <c>vh</c> reads key head <c>vh % NKHead</c> exactly as the causal scan does;
/// HF's <c>repeat_interleave</c> (grouped) order is undone by the converter for the four preview tensors too.</para>
/// <para><b>Precision / storage.</b> Every buffer is the causal mixer's existing scratch; the softmax row (<c>T + 1 &lt;= 33</c> floats) lives
/// on the stack and accumulates in F32. No new stored tensor.</para>
/// </remarks>
internal static unsafe class CloverPreviewGdnMixer
{
    /// <summary>HF <c>gdn_reference_block_size</c>: the history logit's <c>ln(validBlockLen / 32)</c> denominator (fixed by the architecture).</summary>
    public const float ReferenceBlockSize = 32f;

    /// <summary>Largest block the preview accepts (the stack softmax row is <c>T + 1</c> floats).</summary>
    public const int MaxBlock = 64;

    /// <summary>Runs one GDN layer's preview over <paramref name="seqLen"/> block tokens. <paramref name="cache"/> is only read.</summary>
    [SkipLocalsInit]
    public static void Forward(GdnTokenMixingWeights w, in GatedDeltaNetConfig gdn, int ordinal, int seqLen, float eps,
                               ReadOnlySpan<float> x, Span<float> y, GdnGemm gemm, in GdnMixerScratch s, GdnStateCache cache,
                               ComputeThreadPool? pool)
    {
        var pv = w.Preview ?? throw new InvalidOperationException("This GDN layer carries no Clover bidirectional-preview tensors.");
        int nV = gdn.NVHead, nK = gdn.NKHead, dS = gdn.DState, dC = gdn.DConv;
        int convDim = (2 * nK + nV) * dS, vDim = nV * dS, kDim = nK * dS, T = seqLen;
        if (T <= 0 || T > MaxBlock) throw new ArgumentOutOfRangeException(nameof(seqLen), $"Preview block length must be 1..{MaxBlock}.");
        var qkv = s.Qkv; var z = s.Z; var alpha = s.Alpha; var convIn = s.ConvIn;
        var q = s.Q; var k = s.K; var v = s.V; var core = s.Core;

        // ── 1. projections (beta is not used by the preview) ──
        gemm(GdnProjection.Qkv, w.QkvWeight, w.QkvQuantType, x, qkv, w.QkvOutputDim, w.QkvInputDim, T);
        gemm(GdnProjection.Gate, w.GateWeight, w.GateQuantType, x, z, w.GateOutputDim, w.GateInputDim, T);
        gemm(GdnProjection.Alpha, w.AlphaWeight, w.AlphaQuantType, x, alpha, w.AlphaOutputDim, w.AlphaInputDim, T);

        // ── 2. per-token decay exp(g) = exp(softplus(alpha + dt_bias) * A) (the causal mixer's arithmetic) ──
        for (int t = 0; t < T; t++)
            for (int vh = 0; vh < nV; vh++)
            {
                float a = alpha[t * nV + vh] + w.DtBias[vh];
                float sp = MathF.Log(1f + MathF.Exp(a));
                alpha[t * nV + vh] = MathF.Exp(sp * w.A[vh]);
            }

        // ── 3. causal conv over [clean conv state | block] + in-block right conv, SiLU; the conv state is NOT advanced ──
        cache.GetStateForRead(ordinal, out var convState, out var clean);
        convState.Slice(0, (dC - 1) * convDim).CopyTo(convIn.Slice(0, (dC - 1) * convDim));
        qkv.Slice(0, T * convDim).CopyTo(convIn.Slice((dC - 1) * convDim));
        Conv1dCausal.Execute(convIn.Slice(0, (dC - 1 + T) * convDim), w.Conv1dWeight, w.Conv1dBias, qkv.Slice(0, T * convDim), dC, convDim, T);
        AddRightConv(convIn.Slice((dC - 1) * convDim, T * convDim), pv.ConvRightWeight, qkv.Slice(0, T * convDim), dC, convDim, T);
        SiLu.Execute(qkv.Slice(0, T * convDim), qkv.Slice(0, T * convDim));

        // ── 4. de-interleave [Q | K | V], L2-normalise Q and K ──
        for (int t = 0; t < T; t++)
        {
            qkv.Slice(t * convDim, kDim).CopyTo(q.Slice(t * kDim, kDim));
            qkv.Slice(t * convDim + kDim, kDim).CopyTo(k.Slice(t * kDim, kDim));
            qkv.Slice(t * convDim + 2 * kDim, vDim).CopyTo(v.Slice(t * vDim, vDim));
        }
        GatedDeltaNetScan.L2NormalizeHeads(q.Slice(0, T * kDim), dS);
        GatedDeltaNetScan.L2NormalizeHeads(k.Slice(0, T * kDim), dS);

        // ── 5. bidirectional block softmax + read-only history bucket ──
        BlockSoftmax(clean, q, k, v, alpha, pv.LogitScale, pv.HistoryBias, pv.HistoryC, core, nV, nK, dS, T, pool);

        // ── 6. per-head RMSNorm(core, ssm_norm) * silu(z) (the causal mixer's SiLU arm, verbatim) ──
        for (int t = 0; t < T; t++)
        {
            int tBase = t * vDim;
            for (int vh = 0; vh < nV; vh++)
            {
                int headOff = tBase + vh * dS;
                RmsNorm.Execute(core.Slice(headOff, dS), w.SsmNormWeight, eps, core.Slice(headOff, dS));
                var zHead = z.Slice(headOff, dS);
                var outHead = core.Slice(headOff, dS);
                for (int i = 0; i < dS; i++)
                {
                    float zi = zHead[i];
                    outHead[i] *= zi * (1f / (1f + MathF.Exp(-zi)));
                }
            }
        }

        // ── 7. ssm_out ──
        gemm(GdnProjection.Out, w.OutWeight, w.OutQuantType, core.Slice(0, T * vDim), y, w.OutOutputDim, w.OutInputDim, T);
    }

    /// <summary>
    /// <c>conv[t, c] += raw[t+d, c] * right[c*(dConv-1) + d-1]</c> for <c>d = 1..min(dConv, T)-1</c>, <c>t + d &lt; T</c>
    /// (HF <c>bd_gdn_conv1d</c>'s loop; applied to the pre-SiLU conv output).
    /// </summary>
    /// <param name="raw">The block's raw (pre-conv) projections <c>[T, convDim]</c>.</param>
    /// <param name="right">Right-conv taps <c>[convDim, dConv-1]</c> channel-major.</param>
    /// <param name="conv">The causal conv output <c>[T, convDim]</c>, updated in place.</param>
    /// <param name="dConv">Conv width.</param>
    /// <param name="convDim">Channels.</param>
    /// <param name="seqLen">Block length.</param>
    public static void AddRightConv(ReadOnlySpan<float> raw, ReadOnlySpan<float> right, Span<float> conv, int dConv, int convDim, int seqLen)
    {
        int taps = dConv - 1;
        int maxD = Math.Min(dConv, seqLen) - 1;
        for (int d = 1; d <= maxD; d++)
        {
            for (int t = 0; t + d < seqLen; t++)
            {
                var src = raw.Slice((t + d) * convDim, convDim);
                var dst = conv.Slice(t * convDim, convDim);
                for (int c = 0; c < convDim; c++)
                    dst[c] += src[c] * right[c * taps + d - 1];
            }
        }
    }

    private struct SoftmaxCtx
    {
        public float* Clean, Q, K, V, Decay, Scale, Bias, C, Out;
        public int NV, NK, DS, T;
    }

    /// <summary>
    /// The per-head bidirectional softmax with the history bucket (HF <c>bd_gdn_local_memory</c>, all block positions valid).
    /// </summary>
    /// <param name="clean">Clean recurrent state <c>[NVHead, DState(key), DState(value)]</c> (only read).</param>
    /// <param name="q">L2-normalised queries <c>[T, NKHead, DState]</c>.</param>
    /// <param name="k">L2-normalised keys <c>[T, NKHead, DState]</c>.</param>
    /// <param name="v">Values <c>[T, NVHead, DState]</c>.</param>
    /// <param name="decay">Per-token decay <c>exp(g)</c> <c>[T, NVHead]</c>.</param>
    /// <param name="logitScale">Per-head block-logit scale.</param>
    /// <param name="historyBias">Per-head history-logit bias.</param>
    /// <param name="historyC">Per-head history-logit <c>ln(T/32)</c> coefficient.</param>
    /// <param name="output">Output <c>[T, NVHead, DState]</c>.</param>
    /// <param name="nVHead">Value heads.</param>
    /// <param name="nKHead">Key heads (value head <c>vh</c> reads key head <c>vh % nKHead</c>).</param>
    /// <param name="dState">Head dimension.</param>
    /// <param name="seqLen">Block length.</param>
    /// <param name="pool">Optional pool: value heads are split across threads (each head's arithmetic is unchanged).</param>
    public static void BlockSoftmax(ReadOnlySpan<float> clean, ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                    ReadOnlySpan<float> decay, ReadOnlySpan<float> logitScale, ReadOnlySpan<float> historyBias,
                                    ReadOnlySpan<float> historyC, Span<float> output, int nVHead, int nKHead, int dState, int seqLen,
                                    ComputeThreadPool? pool = null)
    {
        if (seqLen <= 0 || seqLen > MaxBlock) throw new ArgumentOutOfRangeException(nameof(seqLen));
        if (nVHead % nKHead != 0) throw new ArgumentException("nVHead must be a multiple of nKHead.");
        fixed (float* pc = clean, pq = q, pk = k, pvv = v, pd = decay, ps = logitScale, pb = historyBias, pcc = historyC, po = output)
        {
            var ctx = new SoftmaxCtx
            {
                Clean = pc, Q = pq, K = pk, V = pvv, Decay = pd, Scale = ps, Bias = pb, C = pcc, Out = po,
                NV = nVHead, NK = nKHead, DS = dState, T = seqLen,
            };
            if (pool is null || nVHead < 2)
                SoftmaxWorker((nint)(&ctx), 0, 1);
            else
                pool.Dispatch((nint)(&ctx), &SoftmaxWorker);
        }
    }

    [SkipLocalsInit]
    private static void SoftmaxWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref readonly var c = ref *(SoftmaxCtx*)ctxPtr;
        int nV = c.NV, nK = c.NK, dS = c.DS, T = c.T;
        int per = (nV + threadCount - 1) / threadCount;
        int h0 = threadIdx * per, h1 = Math.Min(nV, h0 + per);
        float invSqrtD = 1f / MathF.Sqrt(dS);
        float logRatio = MathF.Log(T / ReferenceBlockSize);
        float* w = stackalloc float[MaxBlock + 1];
        for (int vh = h0; vh < h1; vh++)
        {
            int kh = vh % nK;
            float scale = c.Scale[vh];
            float* S = c.Clean + (long)vh * dS * dS;
            for (int i = 0; i < T; i++)
            {
                float* qi = c.Q + ((long)i * nK + kh) * dS;
                // Block logits over every key of the block (bidirectional), then the history bucket last.
                float max = float.NegativeInfinity;
                for (int j = 0; j < T; j++)
                {
                    float* kj = c.K + ((long)j * nK + kh) * dS;
                    float dot = 0f;
                    for (int e = 0; e < dS; e++) dot += qi[e] * kj[e];
                    float l = scale * dot;
                    w[j] = l;
                    if (l > max) max = l;
                }
                float hist = c.Bias[vh] + c.C[vh] * logRatio;
                w[T] = hist;
                if (hist > max) max = hist;
                float sum = 0f;
                for (int j = 0; j <= T; j++) { float e = MathF.Exp(w[j] - max); w[j] = e; sum += e; }
                float inv = 1f / sum;

                // out = w_hist * exp(g_i) * (S^T q_i / sqrt(dS)) + sum_j w_j v_j
                float* o = c.Out + ((long)i * nV + vh) * dS;
                float histW = w[T] * inv * c.Decay[i * nV + vh] * invSqrtD;
                for (int col = 0; col < dS; col++) o[col] = 0f;
                for (int row = 0; row < dS; row++)
                {
                    float qr = qi[row];
                    float* sRow = S + (long)row * dS;
                    for (int col = 0; col < dS; col++) o[col] += sRow[col] * qr;
                }
                for (int col = 0; col < dS; col++) o[col] *= histW;
                for (int j = 0; j < T; j++)
                {
                    float wj = w[j] * inv;
                    float* vj = c.V + ((long)j * nV + vh) * dS;
                    for (int col = 0; col < dS; col++) o[col] += wj * vj[col];
                }
            }
        }
    }
}
