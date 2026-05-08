using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Handles load-time repackaging of AWQ/GPTQ SafeTensors into GGML-compatible layouts.
/// </summary>
public static class SafeTensorsRepackager
{
    /// <summary>
    /// Repackages quantized weight components (qweight, qzeros, scales) into standard GGML formats.
    /// </summary>
    public static unsafe void Repackage(
        IReadOnlyDictionary<string, ModelTensor> rawTensors,
        Dictionary<string, ModelTensor> outputTensors,
        List<nint> allocatedBuffers)
    {
        var processedPrefixes = new HashSet<string>(StringComparer.Ordinal);

        foreach (var kvp in rawTensors)
        {
            string name = kvp.Key;
            
            // Look for quantized weight components
            if (name.EndsWith(".qweight", StringComparison.OrdinalIgnoreCase))
            {
                string prefix = name.Substring(0, name.Length - ".qweight".Length);
                if (processedPrefixes.Add(prefix))
                {
                    if (rawTensors.TryGetValue(prefix + ".scales", out var scales) &&
                        rawTensors.TryGetValue(prefix + ".qzeros", out var qzeros))
                    {
                        // Output the mapped name
                        string mappedName = SafeTensorsNameMapper.MapToDotLlmName(prefix + ".weight");
                        
                        // For the current state of architecture, this establishes the bridge
                        // where GPTQ/AWQ components will be merged into a single Q4_K or Q4_0 memory block.
                        // We will allocate a temporary buffer but won't perform the actual complex bit-unpacking
                        // until the format parsers are complete.
                        
                        // Just as a placeholder to ensure it wires up: 
                        // nint buffer = (nint)NativeMemory.AlignedAlloc((nuint)size, 64);
                        // allocatedBuffers.Add(buffer);
                        
                        // Let's create a placeholder tensor mapping to Q4_0 for now, so the system recognizes it.
                        // In reality, this requires checking the exact quantization scheme used (from config).
                        outputTensors[mappedName] = new ModelTensor(
                            mappedName,
                            kvp.Value.Shape, // Note: shape needs to be reconstructed properly
                            QuantizationType.Q4_0,
                            kvp.Value.Pointer // Temporary point to qweight. Real impl must map to the new buffer.
                        );
                        
                        continue;
                    }
                }
            }
            
            // Skip components of already repackaged tensors
            if (name.EndsWith(".qzeros", StringComparison.OrdinalIgnoreCase) || 
                name.EndsWith(".scales", StringComparison.OrdinalIgnoreCase) || 
                name.EndsWith(".g_idx", StringComparison.OrdinalIgnoreCase))
            {
                string prefix = name.EndsWith(".g_idx", StringComparison.OrdinalIgnoreCase) 
                    ? name.Substring(0, name.Length - ".g_idx".Length)
                    : name.Substring(0, name.LastIndexOf('.'));
                    
                if (processedPrefixes.Contains(prefix))
                {
                    continue;
                }
            }

            // Normal unquantized or natively supported tensor
            if (!name.EndsWith(".qweight", StringComparison.OrdinalIgnoreCase))
            {
                string mappedName = SafeTensorsNameMapper.MapToDotLlmName(name);
                outputTensors[mappedName] = kvp.Value;
            }
        }
    }
}
