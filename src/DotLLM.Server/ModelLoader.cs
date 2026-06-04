using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Threading;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Models.Architectures;

namespace DotLLM.Server;

/// <summary>
/// Centralized utility for loading DotLLM models onto CPU, CUDA, or ROCm backends.
/// Handles device auto-detection and hybrid offloading logic.
/// </summary>
public static class ModelLoader
{
    /// <summary>
    /// Opens the appropriate model container based on the provided path or directory.
    /// </summary>
    public static IModelContainer OpenContainer(string path)
    {
        if (Directory.Exists(path) || path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            return SafeTensorsModelContainer.Open(path);
        }
        else
        {
            return GgufModelContainer.Open(path);
        }
    }

    public static IModel Load(
        IModelContainer container,
        string device,
        int? gpuLayers,
        ThreadingConfig threading)
    {
        var config = container.Config;
        int totalLayers = config.NumLayers;
        int resolvedGpuLayers = ResolveGpuLayers(device, gpuLayers, totalLayers);

        if (resolvedGpuLayers <= 0)
        {
            return TransformerModel.Load(container, threading);
        }

        int gpuId = ParseGpuId(device);
        string backend = ResolveBackend(device);

        if (backend == "cuda")
        {
            return LoadCuda(container, resolvedGpuLayers, gpuId, threading);
        }
        else if (backend == "rocm")
        {
            return LoadRoCm(container, resolvedGpuLayers, gpuId, threading);
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

    private static IModel LoadCuda(IModelContainer container, int gpuLayers, int gpuId, ThreadingConfig threading)
    {
        var config = container.Config;
        if (gpuLayers >= config.NumLayers)
        {
            return DotLLM.Cuda.CudaTransformerModel.Load(container, gpuId);
        }
        return DotLLM.Cuda.HybridTransformerModel.Load(container, gpuLayers, gpuId, threading);
    }

    private static IModel LoadRoCm(IModelContainer container, int gpuLayers, int gpuId, ThreadingConfig threading)
    {
        var config = container.Config;
        if (gpuLayers >= config.NumLayers)
        {
            // Note: HipTransformerModel refactoring pending, for now this will cause build error
            // which I'll fix in next step.
            return DotLLM.RoCm.HipTransformerModel.Load(container, gpuId);
        }
        return DotLLM.RoCm.HipHybridTransformerModel.Load(container, gpuLayers, gpuId, threading);
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
