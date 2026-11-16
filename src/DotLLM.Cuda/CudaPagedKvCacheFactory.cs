using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;

namespace DotLLM.Cuda;

/// <summary>
/// Factory for creating <see cref="CudaPagedKvCache"/> instances backed by a shared
/// <see cref="CudaKvBlockPool"/> (issue #252). Mirrors
/// <see cref="DotLLM.Engine.KvCache.PagedKvCacheFactory"/> — the pool is shared across all
/// sequences on this GPU, enabling efficient VRAM utilization for batch serving.
/// </summary>
public sealed class CudaPagedKvCacheFactory : IDisposable
{
    private readonly CudaKvBlockPool _pool;
    private readonly int _numLayers;

    /// <summary>The underlying block pool shared by all caches created from this factory.</summary>
    public CudaKvBlockPool Pool => _pool;

    /// <summary>Number of transformer layers this factory's pool was sized for.</summary>
    public int NumLayers => _numLayers;

    /// <summary>Total unmanaged device bytes reserved by the shared block pool.</summary>
    public long AllocatedBytes => _pool.AllocatedBytes;

    /// <summary>
    /// Pool ceiling used when the caller asks for no particular size: the pool's historical fixed size. A planner caps it at
    /// the served context and at what fits in free VRAM (<see cref="FitPoolTokens"/>).
    /// </summary>
    public const int DefaultMaxTotalTokens = 65536;

    /// <summary>
    /// Device bytes one token of pool capacity costs: K and V rows (FP16) across every KV slot of <paramref name="geometry"/>.
    /// TwIL-8B (40 layers, 8 KV heads x 128): 160 KiB, so the fixed 65536-token pool was 10 GiB (#1833).
    /// </summary>
    /// <param name="geometry">Per-layer KV row geometry.</param>
    /// <returns>Bytes per token.</returns>
    public static long PoolBytesPerToken(KvGeometry geometry)
    {
        long bytes = 0;
        for (int i = 0; i < geometry.LayerCount; i++)
            bytes += 2L * geometry.KvStrideOf(i) * sizeof(ushort);
        return bytes;
    }

    /// <summary>
    /// Device bytes one token of a sequence's <see cref="CudaPagedKvCache"/> gather scratch costs (contiguous K and V at the
    /// widest stride, allocated per request on top of the pool).
    /// </summary>
    /// <param name="geometry">Per-layer KV row geometry.</param>
    /// <returns>Bytes per token.</returns>
    public static long GatherScratchBytesPerToken(KvGeometry geometry)
    {
        int maxStride = 0;
        for (int i = 0; i < geometry.LayerCount; i++)
            maxStride = Math.Max(maxStride, geometry.KvStrideOf(i));
        return 2L * maxStride * sizeof(ushort);
    }

    /// <summary>
    /// Issue #1833: the pool size, in tokens, that leaves <paramref name="reserveBytes"/> free. The pool used to be a fixed
    /// 65536 tokens whatever the card: 10 GiB for TwIL-8B on a 12 GB RTX 3060 already holding ~5 GB of weights. Under WDDM the
    /// excess does not fail; it is backed by shared system memory, as is everything allocated after it, including the lazily
    /// allocated FP16 dequant scratch every prefill GEMM reads. Prefill then streamed ~32 GB across a PCIe Gen3 x4 link per
    /// forward (1 tok/s) while decode, which reads the quantized weights in place, stayed fast.
    /// </summary>
    /// <param name="freeBytes">Free device memory with the model loaded.</param>
    /// <param name="reserveBytes">Bytes that must stay free (<see cref="DeviceMemoryBudget.ResolveReserveBytes"/>).</param>
    /// <param name="fixedBytes">Context-independent buffers not allocated yet (lazy forward scratch).</param>
    /// <param name="bytesPerToken">Device bytes per token of capacity: pool + gather scratch + prompt activations.</param>
    /// <param name="requestedTokens">Upper bound: the served context, capped at <see cref="DefaultMaxTotalTokens"/>.</param>
    /// <returns>Tokens, a multiple of <see cref="DeviceMemoryBudget.ContextGranularity"/> unless the request is smaller;
    /// 0 when not even one granule fits.</returns>
    public static int FitPoolTokens(long freeBytes, long reserveBytes, long fixedBytes, long bytesPerToken, int requestedTokens)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(requestedTokens);
        return DeviceMemoryBudget.MaxFittingContext(freeBytes, reserveBytes, fixedBytes, bytesPerToken, requestedTokens) switch
        {
            0 when DeviceMemoryBudget.Fits(freeBytes, fixedBytes + bytesPerToken * requestedTokens, reserveBytes) => requestedTokens,
            var fit => fit,
        };
    }

    /// <summary>
    /// Creates a new factory with a shared GPU-resident block pool.
    /// </summary>
    /// <param name="geometry">Per-layer KV row geometry (see <see cref="KvGeometry"/>).</param>
    /// <param name="blockSize">Number of tokens per block (default: 16, matches the CPU pool).</param>
    /// <param name="maxTotalTokens">Maximum total tokens across all sequences
    /// (determines pool size; default: 65536).</param>
    /// <param name="context">Optional CUDA context for explicit multi-device placement; null uses the current context.</param>
    public CudaPagedKvCacheFactory(KvGeometry geometry, int blockSize = 16, int maxTotalTokens = 65536,
                                    CudaContext? context = null)
    {
        if (maxTotalTokens <= 0) throw new ArgumentOutOfRangeException(nameof(maxTotalTokens));

        _numLayers = geometry.LayerCount;
        int totalBlocks = (maxTotalTokens + blockSize - 1) / blockSize;
        _pool = new CudaKvBlockPool(geometry, blockSize, totalBlocks, context);
    }

    /// <summary>
    /// Creates a new factory with a shared GPU-resident block pool, using a single uniform
    /// <c>numKvHeads * headDim</c> stride. Byte-identical to the <see cref="KvGeometry"/>
    /// constructor with <see cref="KvGeometry.Uniform"/>.
    /// </summary>
    public CudaPagedKvCacheFactory(int numLayers, int numKvHeads, int headDim,
                                    int blockSize = 16, int maxTotalTokens = 65536,
                                    CudaContext? context = null)
        : this(KvGeometry.Uniform(numLayers, numKvHeads, headDim), blockSize, maxTotalTokens, context)
    {
    }

    /// <summary>
    /// Creates a new GPU-resident paged KV-cache for a single sequence.
    /// </summary>
    /// <param name="maxSeqLen">Maximum sequence length for this cache.</param>
    /// <returns>A new <see cref="CudaPagedKvCache"/> backed by the shared pool.</returns>
    public CudaPagedKvCache Create(int maxSeqLen) => new(_pool, maxSeqLen);

    /// <inheritdoc/>
    public void Dispose() => _pool.Dispose();
}
