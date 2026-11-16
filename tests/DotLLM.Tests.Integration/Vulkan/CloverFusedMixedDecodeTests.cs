using System.Security.Cryptography;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// Issue #1814: Clover's layer 0 has Q8_0 routed banks, which <c>RecordMoeDecodeFast</c> does not take, so single-token decode fell off the
/// one-command-buffer forward onto the per-layer path (two fence round trips per layer). <see cref="VulkanQwen3MoeHybridTransformerModel.FuseMixedMoeDecode"/>
/// records the same general MoE layer inside the fused buffer instead. Same kernels, same order, only the submission structure differs, so the
/// teacher-forced decode logits must be bit-identical; the counters prove each pass ran the path it claims. Take the GPU and memory locks first.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class CloverFusedMixedDecodeTests
{
    [SkippableFact]
    public unsafe void MixedFusedDecode_IsBitIdenticalToPerLayerPath()
    {
        Skip.If(Environment.GetEnvironmentVariable("DOTLLM_SKIP_VULKAN") == "1", "DOTLLM_SKIP_VULKAN=1");
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string? path = CloverRealGguf.Resolve();
        Skip.If(path is null, "Clover GGUF not found: set DOTLLM_CLOVER_GGUF or hf-download Diffuspace/Clover-35B-A3B-Preview-GGUF.");

        using var gguf = GgufFile.Open(path!);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        int[] ids = tokenizer.Encode(
            "The history of the Roman Empire spans many centuries, during which the city grew from a small settlement into the capital of a vast realm.")
            .Take(32).ToArray();

        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromGguf(device, gguf, config, Path.Combine(AppContext.BaseDirectory, "spv"));
        int vocab = config.VocabSize;
        bool saved = VulkanQwen3MoeHybridTransformerModel.FuseMixedMoeDecode;
        try
        {
            var hashes = new string[3];
            var fused = new long[3];
            var general = new long[3];
            for (int pass = 0; pass < 3; pass++)   // pass 0: discarded warm-up
            {
                VulkanQwen3MoeHybridTransformerModel.FuseMixedMoeDecode = pass == 2;
                long f0 = model.FusedDecodeForwards, g0 = model.FusedDecodeGeneralLayers;
                model.ResetSequenceState();
                using var cache = model.CreateKvCache(ids.Length + 8);
                using var h = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                for (int i = 0; i < ids.Length; i++)
                {
                    using var t = model.Forward(ids.AsSpan(i, 1), [i], 0, cache);
                    h.AppendData(new ReadOnlySpan<byte>((void*)t.DataPointer, vocab * sizeof(float)));
                }
                hashes[pass] = Convert.ToHexString(h.GetHashAndReset());
                fused[pass] = model.FusedDecodeForwards - f0;
                general[pass] = model.FusedDecodeGeneralLayers - g0;
            }
            Assert.Equal(0, fused[1]);                 // per-layer path: Clover is not all-fast-path, so nothing fuses without the switch
            Assert.Equal(ids.Length, fused[2]);        // every decode step ran as one command buffer
            Assert.True(general[2] >= ids.Length, "no general MoE layer was recorded inside the fused buffer");
            Assert.Equal(hashes[1], hashes[2]);
        }
        finally { VulkanQwen3MoeHybridTransformerModel.FuseMixedMoeDecode = saved; }
    }
}
