using System.Diagnostics;
using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;

namespace DotLLM.RoCm;

/// <summary>
/// Split KV-cache for hybrid CPU/GPU inference on AMD ROCm. Routes layers 0..N-1 to a GPU-resident
/// <see cref="HipKvCache"/> and layers N..L-1 to a CPU-resident <see cref="SimpleKvCache"/>.
/// </summary>
public sealed class HipHybridKvCache : IKvCache
{
    private readonly int _numGpuLayers;

    /// <summary>GPU-side KV-cache for layers 0..N-1 (FP16, device memory).</summary>
    internal HipKvCache GpuCache { get; }

    /// <summary>CPU-side KV-cache for layers N..L-1 (FP32, host memory).</summary>
    internal SimpleKvCache CpuCache { get; }

    /// <inheritdoc/>
    public int CurrentLength
    {
        get
        {
            Debug.Assert(GpuCache.CurrentLength == CpuCache.CurrentLength,
                "GPU and CPU KV-caches must advance in lockstep.");
            return CpuCache.CurrentLength;
        }
    }

    /// <inheritdoc/>
    public int MaxLength => CpuCache.MaxLength;

    public HipHybridKvCache(HipKvCache gpuCache, SimpleKvCache cpuCache, int numGpuLayers)
    {
        GpuCache = gpuCache;
        CpuCache = cpuCache;
        _numGpuLayers = numGpuLayers;
    }

    /// <inheritdoc/>
    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
    {
        if (layerIndex < _numGpuLayers)
            throw new InvalidOperationException("Layer is a GPU layer — should use device-side update.");
        CpuCache.Update(keys, values, positions, layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
    {
        if (layerIndex < _numGpuLayers)
            throw new InvalidOperationException("Layer is a GPU layer — should use device-side update.");
        CpuCache.Update(keys, values, positions, layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public ITensor GetKeys(int layerIndex)
    {
        if (layerIndex < _numGpuLayers)
            throw new InvalidOperationException("Layer is a GPU layer — use GpuCache directly.");
        return CpuCache.GetKeys(layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public ITensor GetValues(int layerIndex)
    {
        if (layerIndex < _numGpuLayers)
            throw new InvalidOperationException("Layer is a GPU layer — use GpuCache directly.");
        return CpuCache.GetValues(layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public TensorRef GetKeysRef(int layerIndex)
    {
        if (layerIndex < _numGpuLayers)
            return GpuCache.GetKeysRef(layerIndex);
        return CpuCache.GetKeysRef(layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public TensorRef GetValuesRef(int layerIndex)
    {
        if (layerIndex < _numGpuLayers)
            return GpuCache.GetValuesRef(layerIndex);
        return CpuCache.GetValuesRef(layerIndex - _numGpuLayers);
    }

    /// <inheritdoc/>
    public void Rollback(int length)
    {
        GpuCache.Rollback(length);
        CpuCache.Rollback(length);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        GpuCache.Dispose();
        CpuCache.Dispose();
    }
}
