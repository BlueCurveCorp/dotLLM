using DotLLM.Core.Models;
using DotLLM.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Cuda;

/// <summary>
/// Convenience helper for loading a model onto a GPU from a GGUF file.
/// </summary>
public static class CudaModelLoader
{
    /// <summary>
    /// Loads a transformer model from a model container onto the specified GPU.
    /// </summary>
    /// <param name="container">Opened model container.</param>
    /// <param name="deviceId">GPU device ordinal (0-based).</param>
    /// <param name="ptxDir">Directory containing compiled PTX files. Null for auto-detect.</param>
    /// <returns>The loaded model.</returns>
    public static CudaTransformerModel Load(
        IModelContainer container, int deviceId = 0, string? ptxDir = null)
    {
        return CudaTransformerModel.Load(container, deviceId, ptxDir);
    }
}
