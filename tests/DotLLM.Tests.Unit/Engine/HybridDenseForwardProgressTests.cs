using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using Xunit;
using Fixture = DotLLM.Tests.Unit.Vulkan.VulkanHybridLegacyQuantForwardTests.LegacyQuantFixture;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// #1822: the CPU qwen35 hybrid-dense layer loop (Bonsai-2, the dog-food model) reports through <see cref="ForwardProgress"/> too, once
/// per layer, on the forward's own thread. Uses the 2-layer synthetic hybrid (layer 0 Gated-DeltaNet, layer 1 attention); CPU only.
/// </summary>
public sealed class HybridDenseForwardProgressTests
{
    private sealed class Target(List<(int, int)> seen) : IForwardProgressTarget
    {
        public void OnLayersCompleted(int completed, int total) => seen.Add((completed, total));
    }

    [Fact]
    public void CpuHybridDenseForward_ReportsEachLayer()
    {
        using var fx = Fixture.Build(1822, QuantizationType.Q4_1);
        using var m = Qwen3HybridDenseTransformerModel.BuildFromPrebuiltWeights(
            fx.DenseConfig, fx.DenseLayers, fx.OutputNormWeight, fx.TokenEmbedPtr, QuantizationType.F32,
            fx.OutputWeightPtr, QuantizationType.Q4_1, 8, fx.DenseConfig.HiddenSize);
        m.ResetSequenceState();
        using var kv = new SimpleKvCache(m.AttentionLayerCount, 2, 64, 16);
        int[] ids = [1, 5, 2, 7, 3, 0];

        var seen = new List<(int, int)>();
        var previous = ForwardProgress.Swap(new Target(seen));
        try
        {
            using ITensor logits = m.Forward(ids, Enumerable.Range(0, ids.Length).ToArray(), -1, kv);
        }
        finally { ForwardProgress.Swap(previous); }

        Assert.Equal([(0, 2), (1, 2)], seen);
    }
}
