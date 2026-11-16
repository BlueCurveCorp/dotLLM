using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness coverage for issue #200's direct block-table-read decode attention kernel
/// (<c>attention_f16_paged</c> / <see cref="CudaKernels.LaunchAttentionPaged"/>), the opt-in
/// (<c>DOTLLM_ATTN_PAGED_NATIVE=1</c>) alternative to <see cref="CudaPagedKvCache"/>'s default
/// gather-into-scratch dispatch (<see cref="CudaPagedKvCache.PrepareAttentionScratch"/> +
/// <see cref="CudaKernels.LaunchAttention"/>).
/// </summary>
/// <remarks>
/// <b>Why these assert bit-exact equality, not a tolerance</b>: <c>attention_f16_paged</c> is a
/// deliberate full copy of <c>attention_f16_body</c> (see <c>attention.cu</c>'s header comment)
/// with ONLY the K/V row address resolution changed — every tile's traversal order, every
/// accumulation, every reduction is byte-identical to the flat-buffer kernel. Given the SAME
/// underlying KV content (written into the pool, then either gathered into a contiguous scratch
/// buffer or read directly through the block-pointer array), both code paths must compute the
/// exact same FP16 output — there is no reassociation here (unlike the split-KV kernels' partial
/// combine), so a tolerance-based comparison would mask a real indexing bug instead of catching
/// it. This mirrors the issue's own risk framing: "the math is unchanged... this is purely a
/// memory-access-pattern change."
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaAttentionF16PagedTests : IDisposable
{
    private const int NumLayers = 1;
    private const int BlockSize = 4;
    private const int TotalBlocks = 80;
    private const int MaxSeqLen = 320;

    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;

    public CudaAttentionF16PagedTests()
    {
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
    }

    public void Dispose()
    {
        _stream?.Dispose();
        _ctx?.Dispose();
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    private CudaKvBlockPool CreatePool(int numKvHeads, int headDim) =>
        new(NumLayers, numKvHeads, headDim, BlockSize, TotalBlocks, _ctx);

    /// <summary>
    /// Writes a deterministic, exactly-FP16-representable KV history (small powers-of-two
    /// fractions, same trick <c>CudaPagedKvCacheTests</c> uses) into <paramref name="cache"/> one
    /// token at a time (matching real incremental decode), spanning block boundaries as dictated
    /// by <paramref name="seqLen"/>.
    /// </summary>
    private unsafe void FillCacheIncrementally(CudaPagedKvCache cache, int seqLen, int kvStride)
    {
        for (int pos = 0; pos < seqLen; pos++)
        {
            nint kRow = AllocAndFillDeviceFp16(kvStride, e => RowValue(pos, e, keyOffset: 0));
            nint vRow = AllocAndFillDeviceFp16(kvStride, e => RowValue(pos, e, keyOffset: 1));
            try
            {
                cache.UpdateDevice(kRow, vRow, [pos], 1, layerIndex: 0, _stream!.Handle);
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(kRow);
                CudaDriverApi.cuMemFree_v2(vRow);
            }
        }
        _stream!.Synchronize();
    }

    private static float RowValue(int pos, int elemIndex, int keyOffset) =>
        keyOffset + pos * 0.25f + elemIndex * 0.0625f;

    private static unsafe nint AllocAndFillDeviceFp16(int count, Func<int, float> pattern)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)(count * sizeof(ushort))).ThrowOnError();
        var host = new ushort[count];
        for (int i = 0; i < count; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)pattern(i));
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)p, (nuint)(count * sizeof(ushort))).ThrowOnError();
        return devPtr;
    }

    private static unsafe nint AllocAndFillDeviceFp16Random(Random rng, int count)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint devPtr, (nuint)(count * sizeof(ushort))).ThrowOnError();
        var host = new ushort[count];
        for (int i = 0; i < count; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2.0 - 1.0));
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(devPtr, (nint)p, (nuint)(count * sizeof(ushort))).ThrowOnError();
        return devPtr;
    }

    private static unsafe ushort[] ReadDeviceFp16(nint devicePtr, int count)
    {
        var host = new ushort[count];
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(count * sizeof(ushort))).ThrowOnError();
        return host;
    }

    /// <summary>
    /// Core comparison: build a scattered-block KV history of <paramref name="seqKv"/> tokens,
    /// then run the SAME decode query through (a) the existing gather-based dispatch
    /// (<see cref="CudaPagedKvCache.PrepareAttentionScratch"/> + <see cref="CudaKernels.LaunchAttention"/>)
    /// and (b) the new direct block-table-read kernel
    /// (<see cref="CudaPagedKvCache.PrepareNativeBlockPtrs"/> + <see cref="CudaKernels.LaunchAttentionPaged"/>),
    /// asserting bit-exact FP16 output equality.
    /// </summary>
    private unsafe void AssertPagedNativeMatchesGatherPath(
        int numHeads, int numKvHeads, int headDim, int seqKv, int slidingWindow = 0)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasAttentionF16Paged, "attention_f16_paged not present in PTX (stale build)");

        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;

        using var pool = CreatePool(numKvHeads, headDim);
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);
        FillCacheIncrementally(cache, seqKv, kvStride);
        Assert.Equal(seqKv, cache.CurrentLength);

        var rng = new Random(0xC0DE200 ^ numHeads ^ (numKvHeads << 4) ^ (headDim << 8) ^ (seqKv << 16));
        nint dQ = AllocAndFillDeviceFp16Random(rng, qStride);
        nint dOutGather = 0, dOutPaged = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dOutGather, (nuint)(qStride * sizeof(ushort))).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dOutPaged, (nuint)(qStride * sizeof(ushort))).ThrowOnError();

            nint s = _stream!.Handle;
            int positionOffset = seqKv - 1; // causal: query is the most-recently-cached position

            // (a) existing gather-based dispatch
            var (kGather, vGather) = cache.PrepareAttentionScratch(0, s);
            _stream.Synchronize();
            kernels.LaunchAttention(dQ, kGather, vGather, dOutGather,
                seqQ: 1, seqKv, numHeads, numKvHeads, headDim, positionOffset, slidingWindow, s);
            _stream.Synchronize();

            // (b) new direct block-table-read dispatch
            var (kBlockPtrs, vBlockPtrs, blockCount) = cache.PrepareNativeBlockPtrs(0, s);
            Assert.Equal((seqKv + BlockSize - 1) / BlockSize, blockCount);
            kernels.LaunchAttentionPaged(dQ, kBlockPtrs, vBlockPtrs, dOutPaged,
                seqQ: 1, seqKv, BlockSize, numHeads, numKvHeads, headDim, positionOffset, slidingWindow, s);
            _stream.Synchronize();

            var gather = ReadDeviceFp16(dOutGather, qStride);
            var paged = ReadDeviceFp16(dOutPaged, qStride);

            for (int i = 0; i < qStride; i++)
            {
                float p = (float)BitConverter.UInt16BitsToHalf(paged[i]);
                Assert.False(float.IsNaN(p) || float.IsInfinity(p), $"NaN/Inf in paged-native output at index {i}");
                Assert.Equal(gather[i], paged[i]); // bit-exact — see class remarks
            }
        }
        finally
        {
            if (dQ != 0) CudaDriverApi.cuMemFree_v2(dQ);
            if (dOutGather != 0) CudaDriverApi.cuMemFree_v2(dOutGather);
            if (dOutPaged != 0) CudaDriverApi.cuMemFree_v2(dOutPaged);
        }
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_AtExactBlockBoundary()
    {
        // seqKv is an exact multiple of BlockSize (4) -- no tail partial block.
        AssertPagedNativeMatchesGatherPath(numHeads: 4, numKvHeads: 4, headDim: 16, seqKv: BlockSize * 3);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_WithPartialTailBlock()
    {
        // seqKv is NOT a multiple of BlockSize -- exercises the tail-block offset arithmetic.
        AssertPagedNativeMatchesGatherPath(numHeads: 4, numKvHeads: 4, headDim: 16, seqKv: BlockSize * 3 + 1);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_SingleBlockSequence()
    {
        AssertPagedNativeMatchesGatherPath(numHeads: 4, numKvHeads: 4, headDim: 16, seqKv: 1);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_WithGqaGroupBroadcast()
    {
        // numHeads > numKvHeads (group_size = 4): exercises the hkv = hq / group_size broadcast
        // together with block-table indirection, not just a 1:1 head mapping.
        AssertPagedNativeMatchesGatherPath(numHeads: 8, numKvHeads: 2, headDim: 16, seqKv: BlockSize * 5 + 2);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_WithSlidingWindow()
    {
        // Sliding window smaller than seqKv, spanning a block boundary -- exercises the
        // pos_q - tkv >= sliding_window masking branch alongside block-table indirection.
        AssertPagedNativeMatchesGatherPath(numHeads: 4, numKvHeads: 4, headDim: 16, seqKv: BlockSize * 5, slidingWindow: 6);
    }

    [SkippableFact]
    public void PagedNative_MatchesGatherPath_AtRealBonsaiShape()
    {
        // Real Bonsai-27B decode shape (numHeads=24, numKvHeads=4, headDim=256), depth spanning
        // several blocks with a partial tail -- the shape this issue's motivating investigation
        // (docs/CUDA.md's Future Work entry) actually profiled.
        AssertPagedNativeMatchesGatherPath(numHeads: 24, numKvHeads: 4, headDim: 256, seqKv: 258);
    }

    [SkippableFact]
    public unsafe void PrepareNativeBlockPtrs_ResolvesSamePointersAsPoolAccessors()
    {
        // Allocator-logic check (not full attention numerics): the block-pointer array
        // PrepareNativeBlockPtrs uploads must resolve to EXACTLY the same device addresses
        // CudaKvBlockPool.GetKeyPtr/GetValuePtr would hand back directly for each logical block
        // in this sequence's table -- i.e. the array is a faithful, order-preserving copy, not
        // an off-by-one or stale snapshot.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        const int numKvHeads = 2, headDim = 16;
        int kvStride = numKvHeads * headDim;

        using var pool = CreatePool(numKvHeads, headDim);
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);

        int seqKv = BlockSize * 3 + 2; // 4 logical blocks, partial tail
        FillCacheIncrementally(cache, seqKv, kvStride);

        var (kBlockPtrsDevice, vBlockPtrsDevice, blockCount) = cache.PrepareNativeBlockPtrs(0, _stream!.Handle);
        _stream.Synchronize();
        Assert.Equal((seqKv + BlockSize - 1) / BlockSize, blockCount);

        var kHost = ReadDevicePtrArray(kBlockPtrsDevice, blockCount);
        var vHost = ReadDevicePtrArray(vBlockPtrsDevice, blockCount);

        for (int b = 0; b < blockCount; b++)
        {
            var (blockId, offset) = cache.BlockTable.Resolve(b * BlockSize);
            Assert.Equal(0, offset); // block-aligned position by construction
            Assert.Equal(pool.GetKeyPtr(blockId, 0), (nint)kHost[b]);
            Assert.Equal(pool.GetValuePtr(blockId, 0), (nint)vHost[b]);
        }
    }

    private static unsafe nint[] ReadDevicePtrArray(nint devicePtr, int count)
    {
        var host = new nint[count];
        fixed (nint* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devicePtr, (nuint)(count * sizeof(nint))).ThrowOnError();
        return host;
    }

    // ── Issue #1839: all-layer block-pointer arrays, upload-on-change, paged fused RoPE+KV write ──

    private static unsafe void UploadFp16(nint devicePtr, ushort[] host)
    {
        fixed (ushort* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(devicePtr, (nint)p, (nuint)(host.Length * sizeof(ushort))).ThrowOnError();
    }

    private static ushort[] RandomFp16(Random rng, int count)
    {
        var host = new ushort[count];
        for (int i = 0; i < count; i++)
            host[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextDouble() * 2.0 - 1.0));
        return host;
    }

    [SkippableFact]
    public void PrepareNativeBlockPtrs_MultiLayerScatteredBlocks_ResolveEachLayersPoolAndUploadOnce()
    {
        // Three layers, and a second cache interleaving its allocations so this cache's physical
        // blocks are NOT consecutive: a wrong layer offset into the combined array, or a pointer
        // derived from the logical block index instead of the physical ID, both show up here.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        const int layers = 3, numKvHeads = 2, headDim = 16;
        int kvStride = numKvHeads * headDim;

        using var pool = new CudaKvBlockPool(layers, numKvHeads, headDim, BlockSize, TotalBlocks, _ctx);
        using var cache = new CudaPagedKvCache(pool, MaxSeqLen);
        using var other = new CudaPagedKvCache(pool, MaxSeqLen);

        nint kRow = AllocAndFillDeviceFp16(kvStride, e => e * 0.0625f);
        nint vRow = AllocAndFillDeviceFp16(kvStride, e => 1 + e * 0.0625f);
        try
        {
            nint s = _stream!.Handle;
            int seqKv = BlockSize * 4 + 1;
            for (int pos = 0; pos < seqKv; pos++)
            {
                for (int layer = 0; layer < layers; layer++)
                {
                    cache.UpdateDevice(kRow, vRow, [pos], 1, layer, s);
                    other.UpdateDevice(kRow, vRow, [pos], 1, layer, s);
                }
            }
            _stream.Synchronize();
            Assert.NotEqual(cache.BlockTable.BlockIdAt(0) + 1, cache.BlockTable.BlockIdAt(1)); // scattered

            for (int layer = 0; layer < layers; layer++)
            {
                var (kDev, vDev, blockCount) = cache.PrepareNativeBlockPtrs(layer, s);
                _stream.Synchronize();
                Assert.Equal((seqKv + BlockSize - 1) / BlockSize, blockCount);
                var kHost = ReadDevicePtrArray(kDev, blockCount);
                var vHost = ReadDevicePtrArray(vDev, blockCount);
                for (int b = 0; b < blockCount; b++)
                {
                    int id = cache.BlockTable.BlockIdAt(b);
                    Assert.Equal(pool.GetKeyPtr(id, layer), kHost[b]);
                    Assert.Equal(pool.GetValuePtr(id, layer), vHost[b]);
                }
            }

            // One upload served every layer; asking again without a table change does not re-upload.
            Assert.Equal(1, cache.BlockPtrUploads);
            cache.PrepareNativeBlockPtrs(0, s);
            Assert.Equal(1, cache.BlockPtrUploads);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(kRow);
            CudaDriverApi.cuMemFree_v2(vRow);
        }
    }

    [SkippableFact]
    public void PrepareNativeBlockPtrs_ReUploadsAfterCopyOnWrite()
    {
        // Prefix sharing: the target is seeded with the source's blocks, uploads its pointers, then
        // rolls back into a shared block and writes, which copies-on-write that block. The next call
        // must point at the NEW physical block, not the shared one the source still owns.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        const int numKvHeads = 2, headDim = 16;
        int kvStride = numKvHeads * headDim;

        using var pool = CreatePool(numKvHeads, headDim);
        using var source = new CudaPagedKvCache(pool, MaxSeqLen);
        using var target = new CudaPagedKvCache(pool, MaxSeqLen);
        FillCacheIncrementally(source, BlockSize * 2, kvStride);

        var ids = new List<int>();
        source.SnapshotFullBlocks(BlockSize * 2, ids);
        foreach (int id in ids) pool.AddRef(id);
        target.SeedSharedPrefix(ids, BlockSize * 2);

        nint s = _stream!.Handle;
        var (k0, _, n0) = target.PrepareNativeBlockPtrs(0, s);
        _stream.Synchronize();
        Assert.Equal(pool.GetKeyPtr(ids[1], 0), ReadDevicePtrArray(k0, n0)[1]);
        Assert.Equal(1, target.BlockPtrUploads);

        target.Rollback(BlockSize + 1);
        nint kRow = AllocAndFillDeviceFp16(kvStride, e => 7.0f);
        nint vRow = AllocAndFillDeviceFp16(kvStride, e => 8.0f);
        try
        {
            target.UpdateDevice(kRow, vRow, [BlockSize + 1], 1, 0, s);
            int cowId = target.BlockTable.BlockIdAt(1);
            Assert.NotEqual(ids[1], cowId);

            var (k1, v1, n1) = target.PrepareNativeBlockPtrs(0, s);
            _stream.Synchronize();
            Assert.Equal(2, target.BlockPtrUploads);
            Assert.Equal(pool.GetKeyPtr(cowId, 0), ReadDevicePtrArray(k1, n1)[1]);
            Assert.Equal(pool.GetValuePtr(cowId, 0), ReadDevicePtrArray(v1, n1)[1]);
            Assert.Equal(pool.GetKeyPtr(ids[0], 0), ReadDevicePtrArray(k1, n1)[0]); // still shared
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(kRow);
            CudaDriverApi.cuMemFree_v2(vRow);
        }
    }

    [SkippableFact]
    public unsafe void PagedFusedRopeKvWrite_MatchesContiguousFusedWrite_BitExact()
    {
        // The paged decode write (FusedRopeAndUpdateDevice: block base + offset-in-block) must put
        // the same rotated K and the same V in the same logical rows as CudaKvCache's fused write,
        // and rotate Q identically. Two layers, scattered blocks (a second cache interleaves its
        // allocations), and a length past two block boundaries.
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasFusedRopeKvWriteKernel, "fused_rope_kv_write not present in PTX");

        const int layers = 2, numHeads = 8, numKvHeads = 2, headDim = 64, ropeDim = 64;
        const float theta = 10000f;
        const int ropeType = 1; // neox
        int kvStride = numKvHeads * headDim, qStride = numHeads * headDim;
        int seqKv = BlockSize * 2 + 3;

        using var pool = new CudaKvBlockPool(layers, numKvHeads, headDim, BlockSize, TotalBlocks, _ctx);
        using var paged = new CudaPagedKvCache(pool, MaxSeqLen);
        using var other = new CudaPagedKvCache(pool, MaxSeqLen);
        using var flat = new CudaKvCache(layers, numKvHeads, headDim, MaxSeqLen, _ctx);

        CudaDriverApi.cuMemAlloc_v2(out nint qA, (nuint)(qStride * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint qB, (nuint)(qStride * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint k, (nuint)(kvStride * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint v, (nuint)(kvStride * sizeof(ushort))).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint posDev, sizeof(int)).ThrowOnError();
        try
        {
            nint s = _stream!.Handle;
            var rng = new Random(1839);
            for (int pos = 0; pos < seqKv; pos++)
            {
                int p = pos;
                CudaDriverApi.cuMemcpyHtoD_v2(posDev, (nint)(&p), sizeof(int)).ThrowOnError();
                for (int layer = 0; layer < layers; layer++)
                {
                    var qHost = RandomFp16(rng, qStride);
                    UploadFp16(qA, qHost);
                    UploadFp16(qB, qHost);
                    UploadFp16(k, RandomFp16(rng, kvStride));
                    UploadFp16(v, RandomFp16(rng, kvStride));

                    flat.FusedRopeAndUpdateDevice(qA, k, v, posDev, pos, layer,
                        numHeads, numKvHeads, headDim, ropeDim, theta, ropeType, s, kernels);
                    paged.FusedRopeAndUpdateDevice(qB, k, v, posDev, pos, layer,
                        numHeads, numKvHeads, headDim, ropeDim, theta, ropeType, s, kernels);
                    other.UpdateDevice(k, v, [pos], 1, layer, s);
                    _stream.Synchronize();
                    Assert.Equal(ReadDeviceFp16(qA, qStride), ReadDeviceFp16(qB, qStride));
                }
            }
            Assert.Equal(seqKv, paged.CurrentLength);
            Assert.NotEqual(paged.BlockTable.BlockIdAt(0) + 1, paged.BlockTable.BlockIdAt(1)); // scattered

            for (int layer = 0; layer < layers; layer++)
            {
                var (kG, vG) = paged.PrepareAttentionScratch(layer, s);
                _stream.Synchronize();
                Assert.Equal(ReadDeviceFp16(flat.GetKeysPtr(layer), seqKv * kvStride), ReadDeviceFp16(kG, seqKv * kvStride));
                Assert.Equal(ReadDeviceFp16(flat.GetValuesPtr(layer), seqKv * kvStride), ReadDeviceFp16(vG, seqKv * kvStride));
            }
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(qA);
            CudaDriverApi.cuMemFree_v2(qB);
            CudaDriverApi.cuMemFree_v2(k);
            CudaDriverApi.cuMemFree_v2(v);
            CudaDriverApi.cuMemFree_v2(posDev);
        }
    }
}
