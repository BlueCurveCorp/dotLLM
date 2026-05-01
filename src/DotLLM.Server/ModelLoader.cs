using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Threading;
using DotLLM.Models.Gguf;
using DotLLM.Models.Architectures;

namespace DotLLM.Server;

/// <summary>
/// Centralized utility for loading DotLLM models onto CPU, CUDA, or ROCm backends.
/// Handles device auto-detection and hybrid offloading logic.
/// </summary>
public static class ModelLoader
{
    public static IModel Load(
        GgufFile gguf,
        ModelConfig config,
        string device,
        int? gpuLayers,
        ThreadingConfig threading)
    {
        int totalLayers = config.NumLayers;
        int resolvedGpuLayers = ResolveGpuLayers(device, gpuLayers, totalLayers);

        if (resolvedGpuLayers <= 0)
        {
            return TransformerModel.LoadFromGguf(gguf, config, threading);
        }

        int gpuId = ParseGpuId(device);
        string backend = ResolveBackend(device);

        if (backend == "cuda")
        {
            return LoadCuda(gguf, config, resolvedGpuLayers, gpuId, threading);
        }
        else if (backend == "rocm")
        {
            return LoadRoCm(gguf, config, resolvedGpuLayers, gpuId, threading);
        }
        else
        {
            throw new NotSupportedException($"Unsupported GPU backend: {backend}");
        }
    }

    public static int ResolveGpuLayers(string device, int? gpuLayers, int totalLayers)
    {
        if (gpuLayers.HasValue)
            return Math.Clamp(gpuLayers.Value, 0, totalLayers);

        bool isGpu = device.StartsWith("gpu", StringComparison.OrdinalIgnoreCase) ||
                     device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase) ||
                     device.StartsWith("rocm", StringComparison.OrdinalIgnoreCase);

        return isGpu ? totalLayers : 0;
    }

    private static string ResolveBackend(string device)
    {
        if (device.StartsWith("cuda", StringComparison.OrdinalIgnoreCase)) return "cuda";
        if (device.StartsWith("rocm", StringComparison.OrdinalIgnoreCase)) return "rocm";
        
        // Auto-detect or default to CUDA
        // TODO: Could use runtime detection here
        return "cuda";
    }

    public static int ParseGpuId(string device)
    {
        int colonIdx = device.IndexOf(':');
        if (colonIdx < 0) return 0;
        if (int.TryParse(device.AsSpan(colonIdx + 1), out int id)) return id;
        return 0;
    }

    private static IModel LoadCuda(GgufFile gguf, ModelConfig config, int gpuLayers, int gpuId, ThreadingConfig threading)
    {
        if (gpuLayers >= config.NumLayers)
        {
            return DotLLM.Cuda.CudaTransformerModel.LoadFromGguf(gguf, config, gpuId);
        }
        return DotLLM.Cuda.HybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gpuId, threading);
    }

    private static IModel LoadRoCm(GgufFile gguf, ModelConfig config, int gpuLayers, int gpuId, ThreadingConfig threading)
    {
        if (gpuLayers >= config.NumLayers)
        {
            return DotLLM.RoCm.HipTransformerModel.LoadFromGguf(gguf, config, gpuId);
        }
        return DotLLM.RoCm.HipHybridTransformerModel.LoadFromGguf(gguf, config, gpuLayers, gpuId, threading);
    }

    public static string? GetVramWarning(IModel model)
    {
        return model switch
        {
            DotLLM.Cuda.CudaTransformerModel c => c.VramWarning,
            DotLLM.Cuda.HybridTransformerModel h => h.VramWarning,
            DotLLM.RoCm.HipTransformerModel r => r.VramWarning,
            DotLLM.RoCm.HipHybridTransformerModel rh => rh.VramWarning,
            _ => null
        };
    }
}
