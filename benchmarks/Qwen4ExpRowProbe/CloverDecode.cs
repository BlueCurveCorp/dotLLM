using System.Diagnostics;
using System.Runtime.InteropServices;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;

/// <summary>
/// #1814 decode-gap probe for Qwen3MoeHybrid files (Clover-35B-A3B, Qwen3.6-35B-A3B). Loads once, then:
/// <c>cloverdec &lt;gguf&gt; &lt;ctx&gt; &lt;steps&gt; &lt;reps&gt; &lt;arms&gt;</c>: per rep, every arm (interleaved, ABAB order) prefills
/// <c>ctx</c> tokens and decodes <c>steps</c> single tokens with a KV cache; tok/s per arm (min/median/max).
/// Arms (comma list, '+' combines): base, unfused (FuseDecodeEnabled off), mixed (FuseMixedMoeDecode), head
/// (LM-head dispatch skipped: perturbation, logits stale).
/// <c>cloverdec &lt;gguf&gt; bf16bench</c>: isolated LM-head-shaped BF16 GEMV, the shipped kernel's GB/s and error vs a CPU reference (a 16-byte-load NR=1/2/4/8 variant measured 200/191/139/111 GB/s against its 196: refuted, not kept).
/// <c>cloverdec &lt;gguf&gt; dump &lt;arm&gt; &lt;out.f32&gt; [maxTokens]</c>: teacher-forced single-token decode logits of <see cref="DecodeKl.Texts"/>.
/// </summary>
internal static class CloverDecode
{
    public static void Run(string[] a)
    {
        using var file = GgufFile.Open(a[1]);
        var cfg = GgufModelConfigExtractor.Extract(file.Metadata);
        using var device = VulkanDevice.Create();
        string spv = Path.Combine(AppContext.BaseDirectory, "spv");
        if (a[2] == "bf16bench") { Bf16Bench(device, spv); return; }
        var sw0 = Stopwatch.StartNew();
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, file, cfg, spv);
        Console.WriteLine($"loaded in {sw0.Elapsed.TotalSeconds:F0} s; LM head {model.Weights.OutputDeviceQuantType} [{model.Weights.OutputOutputDim} x {model.Weights.OutputInputDim}]");
        int V = cfg.VocabSize;
        if (a[2] == "dump") { Dump(model, file, a[3], a[4], a.Length > 5 ? int.Parse(a[5]) : int.MaxValue, V); return; }

        int ctx = int.Parse(a[2]), steps = int.Parse(a[3]), reps = int.Parse(a[4]);
        string[] arms = a[5].Split(',');
        int[] Ids(int n, int seed) { var r = new Random(seed); return Enumerable.Range(0, n).Select(_ => r.Next(1000, Math.Min(V, 100000))).ToArray(); }
        var ctxIds = Ids(ctx, 11);

        double Run1(string arm, int seed, bool stage = false)
        {
            Apply(arm);
            using var kv = model.CreateKvCache(ctx + steps + 8);
            model.ResetSequenceState();
            for (int p = 0; p < ctx; p += 512)
            {
                int m = Math.Min(512, ctx - p);
                using var w = model.Forward(ctxIds.AsSpan(p, m), Enumerable.Range(p, m).ToArray(), 0, kv);
            }
            var dec = Ids(steps, seed);
            long f0 = model.FusedDecodeForwards, g0 = model.FusedDecodeGeneralLayers;
            var t0 = Stopwatch.GetTimestamp();
            for (int i = 0; i < steps; i++)
            {
                using var lg = model.Forward(dec.AsSpan(i, 1), new[] { ctx + i }, 0, kv);
            }
            double sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            if (seed == 1000) Console.WriteLine($"  [{arm}] fused forwards {model.FusedDecodeForwards - f0}/{steps}, general layers {model.FusedDecodeGeneralLayers - g0}");
            return steps / sec;
        }

        foreach (var arm in arms.Distinct()) Run1(arm, 1000);   // warm + path proof per arm
        var res = arms.Select((x, i) => (key: $"{x}#{i}", list: new List<double>())).ToList();
        for (int r = 0; r < reps; r++)
        {
            // ABAB then BABA: both orders inside every rep pair.
            var order = Enumerable.Range(0, arms.Length);
            if ((r & 1) == 1) order = order.Reverse();
            foreach (int i in order) res[i].list.Add(Run1(arms[i], 2000 + r));
        }
        Apply("base");
        Console.WriteLine($"ctx={ctx} steps={steps} reps={reps}");
        foreach (var (key, list) in res)
        {
            var s = list.OrderBy(x => x).ToList();
            Console.WriteLine($"  {key,-18} tok/s min {s[0]:F2} med {s[s.Count / 2]:F2} max {s[^1]:F2}  ({string.Join(" ", list.Select(x => x.ToString("F1")))})");
        }
    }

    private static void Apply(string arm)
    {
        var parts = arm.Split('+');
        VulkanQwen3MoeHybridTransformerModel.FuseDecodeEnabled = !parts.Contains("unfused");
        VulkanQwen3MoeHybridTransformerModel.FuseMixedMoeDecode = parts.Contains("mixed");
        VulkanQwen3MoeHybridTransformerModel.PerturbSkip = parts.Contains("head") ? "head" : null;
    }

    private static unsafe void Dump(VulkanQwen3MoeHybridTransformerModel model, GgufFile gguf, string arm, string outPath, int maxTokens, int vocab)
    {
        Apply(arm);
        var tk = GgufTokenizerFactory.Load(gguf.Metadata);
        using var fs = new FileStream(outPath, FileMode.Create, FileAccess.Write, FileShare.Read, 1 << 20);
        long f0 = model.FusedDecodeForwards, g0 = model.FusedDecodeGeneralLayers;
        int steps = 0;
        foreach (var text in DecodeKl.Texts)
        {
            var ids = tk.Encode(text).Take(maxTokens).ToArray();
            using var kv = model.CreateKvCache(ids.Length + 8);
            model.ResetSequenceState();
            for (int i = 0; i < ids.Length; i++)
            {
                using var t = model.Forward(ids.AsSpan(i, 1), new[] { i }, 0, kv);
                fs.Write(new ReadOnlySpan<byte>((void*)t.DataPointer, vocab * 4));
                steps++;
            }
        }
        Console.WriteLine($"dump [{arm}] {steps} steps -> {outPath}; fused {model.FusedDecodeForwards - f0}, general layers {model.FusedDecodeGeneralLayers - g0}");
        Apply("base");
    }

    private static void Bf16Bench(VulkanDevice device, string spv)
    {
        const int M = 248320, K = 2048;
        var rng = new Random(5);
        var w = new ushort[(long)M * K];
        for (long i = 0; i < w.LongLength; i++) w[i] = (ushort)(BitConverter.SingleToInt32Bits((float)(rng.NextDouble() - 0.5) * 0.05f) >> 16);
        var x = new float[K]; for (int i = 0; i < K; i++) x[i] = (float)(rng.NextDouble() - 0.5);
        using var wb = device.AllocateDeviceLocal((long)M * K * 2);
        using var xb = device.AllocateDeviceLocal(K * 4);
        using var yb = device.AllocateDeviceLocal(M * 4);
        device.Upload((ReadOnlySpan<byte>)MemoryMarshal.AsBytes(w.AsSpan()), wb);
        device.Upload((ReadOnlySpan<float>)x, xb);
        // CPU reference on a sample of rows.
        double[] Ref(int[] rows) => rows.Select(m => { double s = 0; for (int k = 0; k < K; k++) s += BitConverter.Int32BitsToSingle(w[(long)m * K + k] << 16) * (double)x[k]; return s; }).ToArray();
        int[] sample = Enumerable.Range(0, 64).Select(i => i * (M / 64) + i).Append(M - 1).ToArray();
        var refv = Ref(sample);
        using var old = MatMulBf16GemvF32Kernel.Create(device, spv);
        var kernels = new List<(string, Action)> { ("old", () => old.Launch(wb, xb, yb, M, K)) };
        foreach (var (name, run) in kernels)
        {
            for (int i = 0; i < 5; i++) run();
            var y = new float[M]; device.Download(yb, y.AsSpan());
            double maxErr = sample.Select((m, i) => Math.Abs(y[m] - refv[i])).Max();
            var t = new List<double>();
            for (int i = 0; i < 20; i++) { var t0 = Stopwatch.GetTimestamp(); run(); t.Add(Stopwatch.GetElapsedTime(t0).TotalMilliseconds); }
            t.Sort();
            double gbs = (double)M * K * 2 / (t[t.Count / 2] * 1e6);
            Console.WriteLine($"  {name,-12} min {t[0]:F2} med {t[t.Count / 2]:F2} ms (incl. submit) -> {gbs:F0} GB/s; max |err| vs CPU {maxErr:E2}");
        }
    }
}
