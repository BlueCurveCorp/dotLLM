using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1833: the CUDA paged-KV block pool was a fixed 65536 tokens. For TwIL-8B (40 layers, 8 KV heads x 128) that is
/// 10 GiB, on a 12 GB RTX 3060 already holding ~5 GB of weights. WDDM backs the excess with shared system memory, and the lazily
/// allocated FP16 dequant scratch that every prefill GEMM reads landed there too: ~32 GB over PCIe Gen3 x4 per prefill, 1 tok/s.
/// These pin the GPU-free sizing arithmetic (<see cref="CudaPagedKvCacheFactory.FitPoolTokens"/>).
/// </summary>
public sealed class CudaPagedKvPoolSizingTests
{
    private const long MiB = 1L << 20;
    private const long GiB = 1L << 30;

    // TwIL-8B / Granite-4.2 8B: 40 layers, 8 KV heads x head_dim 128.
    private static readonly KvGeometry TwilGeometry = KvGeometry.Uniform(40, 8, 128);

    // Largest projection (ffn 12800 x 4096) as FP16: CudaForwardState.PendingLazyScratchBytes for TwIL.
    private const long TwilDequantScratch = 12800L * 4096 * 2;

    // CudaForwardState.ActivationBytesPerToken for TwIL with the FP16 forward (hidden 4096, q 4096, kv 1024, ffn 12800).
    private const long TwilActivationsPerToken = (3L * 4096 + 2 * 4096 + 2 * 1024 + 3L * 12800 + 12800 + 256) * 2 + 8;

    private static long TwilPerToken =>
        CudaPagedKvCacheFactory.PoolBytesPerToken(TwilGeometry)
        + CudaPagedKvCacheFactory.GatherScratchBytesPerToken(TwilGeometry) * CudaPagedKvCacheFactory.LiveCachesCharged
        + TwilActivationsPerToken;

    [Fact]
    public void PoolBytesPerToken_Twil_Is160KiB_SoTheOldFixedPoolWas10GiB()
    {
        Assert.Equal(160 * 1024, CudaPagedKvCacheFactory.PoolBytesPerToken(TwilGeometry));
        Assert.Equal(4 * 1024, CudaPagedKvCacheFactory.GatherScratchBytesPerToken(TwilGeometry));
        Assert.Equal(10 * GiB,
            CudaPagedKvCacheFactory.PoolBytesPerToken(TwilGeometry) * CudaPagedKvCacheFactory.DefaultMaxTotalTokens);
    }

    [Fact]
    public void PoolBytesPerToken_SumsPerLayerStrides()
    {
        // Gemma-4-style per-layer widths: a uniform-stride shortcut (layers x first stride) would give 2*2*(3*256) = 3072.
        var geom = KvGeometry.PerLayer([256, 512, 1024]);
        Assert.Equal(2L * 2 * (256 + 512 + 1024), CudaPagedKvCacheFactory.PoolBytesPerToken(geom));
        Assert.Equal(2L * 2 * 1024, CudaPagedKvCacheFactory.GatherScratchBytesPerToken(geom));
    }

    [Fact]
    public void FitPoolTokens_Twil_On12GBCard_FitsUnderFreeVram_AndIsMaximal()
    {
        // ~12 GiB card, ~5 GiB weights + context: ~6.5 GiB free when the pool is planned.
        long free = 6656 * MiB;
        long reserve = DeviceMemoryBudget.DefaultReserveBytes;

        int tokens = CudaPagedKvCacheFactory.FitPoolTokens(free, reserve, TwilDequantScratch, TwilPerToken,
            CudaPagedKvCacheFactory.DefaultMaxTotalTokens);

        Assert.True(tokens < CudaPagedKvCacheFactory.DefaultMaxTotalTokens, $"pool not reduced: {tokens}");
        Assert.True(tokens >= 16 * 1024, $"pool needlessly small: {tokens}");
        Assert.Equal(0, tokens % DeviceMemoryBudget.ContextGranularity);
        // Everything the pool and its riders take, plus the reserve, stays inside free VRAM ...
        Assert.True(tokens * TwilPerToken + TwilDequantScratch + reserve <= free);
        // ... and one more granule would not.
        Assert.True((tokens + DeviceMemoryBudget.ContextGranularity) * TwilPerToken + TwilDequantScratch + reserve > free);
    }

    [Fact]
    public void FitPoolTokens_ChargesPendingScratch()
    {
        // Free VRAM holds exactly 8 granules of pool plus the reserve, but not the dequant scratch on top.
        long reserve = 256 * MiB;
        int granules = 8;
        long free = granules * DeviceMemoryBudget.ContextGranularity * TwilPerToken + reserve;

        Assert.Equal(granules * DeviceMemoryBudget.ContextGranularity,
            CudaPagedKvCacheFactory.FitPoolTokens(free, reserve, 0, TwilPerToken, 65536));
        Assert.True(CudaPagedKvCacheFactory.FitPoolTokens(free, reserve, TwilDequantScratch, TwilPerToken, 65536)
                    < granules * DeviceMemoryBudget.ContextGranularity);
    }

    [Fact]
    public void FitPoolTokens_ChargesReserve()
    {
        long free = 4 * DeviceMemoryBudget.ContextGranularity * TwilPerToken;
        Assert.Equal(4 * DeviceMemoryBudget.ContextGranularity,
            CudaPagedKvCacheFactory.FitPoolTokens(free, 0, 0, TwilPerToken, 65536));
        Assert.Equal(3 * DeviceMemoryBudget.ContextGranularity,
            CudaPagedKvCacheFactory.FitPoolTokens(free, 256 * MiB, 0, TwilPerToken, 65536));
    }

    [Fact]
    public void FitPoolTokens_CapsAtRequested_OnABigCard()
    {
        Assert.Equal(32768, CudaPagedKvCacheFactory.FitPoolTokens(80 * GiB, 256 * MiB, TwilDequantScratch, TwilPerToken, 32768));
        Assert.Equal(CudaPagedKvCacheFactory.DefaultMaxTotalTokens,
            CudaPagedKvCacheFactory.FitPoolTokens(80 * GiB, 256 * MiB, TwilDequantScratch, TwilPerToken,
                CudaPagedKvCacheFactory.DefaultMaxTotalTokens));
    }

    [Fact]
    public void FitPoolTokens_SubGranuleRequest_GrantedWhenItFits_ZeroWhenNot()
    {
        Assert.Equal(512, CudaPagedKvCacheFactory.FitPoolTokens(1 * GiB, 0, 0, TwilPerToken, 512));
        // Less than one 1024-token granule is free, but the small request fits: granted, not rounded down to 0.
        Assert.Equal(512, CudaPagedKvCacheFactory.FitPoolTokens(600 * TwilPerToken, 0, 0, TwilPerToken, 512));
        Assert.Equal(0, CudaPagedKvCacheFactory.FitPoolTokens(511 * TwilPerToken, 0, 0, TwilPerToken, 512));
    }

    [Fact]
    public void ClampToPool_CapsASequenceAtThePoolCapacity()
    {
        // A 131072-token request (TwIL's window) against a 20480-token pool: the gather scratch must be sized to the pool
        // (80 MiB for TwIL), not the request (512 MiB, uncharged by the fit).
        Assert.Equal(20480, CudaPagedKvCacheFactory.ClampToPool(131072, totalBlocks: 1280, blockSize: 16));
        Assert.Equal(300, CudaPagedKvCacheFactory.ClampToPool(300, totalBlocks: 1280, blockSize: 16));
        Assert.Equal(int.MaxValue, CudaPagedKvCacheFactory.ClampToPool(int.MaxValue, totalBlocks: int.MaxValue, blockSize: 16));
    }

    [Fact]
    public void FitPoolTokens_NothingFits_ReturnsZero()
        => Assert.Equal(0, CudaPagedKvCacheFactory.FitPoolTokens(200 * MiB, 256 * MiB, TwilDequantScratch, TwilPerToken, 65536));
}
