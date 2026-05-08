using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Models;

/// <summary>
/// Describes a single tensor entry in a model container: its name, shape, quantization type, and memory pointer.
/// </summary>
/// <param name="Name">Tensor name (e.g., "blk.0.attn_q.weight").</param>
/// <param name="Shape">Tensor dimensions.</param>
/// <param name="QuantizationType">Storage format / quantization scheme.</param>
/// <param name="Pointer">Direct pointer to the memory-mapped tensor data.</param>
public readonly record struct ModelTensor(
    string Name,
    TensorShape Shape,
    QuantizationType QuantizationType,
    nint Pointer);

/// <summary>
/// Represents an abstraction for model storage formats (GGUF, SafeTensors).
/// Decouples the storage format from the inference engine.
/// </summary>
public interface IModelContainer : IDisposable
{
    /// <summary>
    /// The model configuration extracted from the container's metadata or accompanying config files.
    /// </summary>
    ModelConfig Config { get; }

    /// <summary>
    /// Gets all tensors in the container.
    /// </summary>
    IEnumerable<ModelTensor> Tensors { get; }

    /// <summary>
    /// The offset in the source file where tensor data begins.
    /// </summary>
    long DataSectionOffset { get; }

    /// <summary>
    /// Attempts to retrieve a tensor by name.
    /// </summary>
    /// <param name="name">The internal name of the tensor.</param>
    /// <param name="tensor">The tensor descriptor if found.</param>
    /// <returns>True if the tensor exists in the container.</returns>
    bool TryGetTensor(string name, out ModelTensor tensor);

    /// <summary>
    /// Gets a list of all tensor names available in the container.
    /// </summary>
    IReadOnlyCollection<string> GetTensorNames();

    /// <summary>
    /// Retrieves a metadata value from the container.
    /// </summary>
    /// <typeparam name="T">Expected type of the metadata value.</typeparam>
    /// <param name="key">The metadata key.</param>
    /// <returns>The metadata value if found, otherwise default.</returns>
    T? GetMetadata<T>(string key);
}
