using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.Gguf;

/// <summary>
/// GGUF-backed implementation of <see cref="IModelContainer"/>.
/// Wraps a <see cref="GgufFile"/> and translates its metadata and tensor descriptors.
/// </summary>
public sealed class GgufModelContainer : IModelContainer
{
    private readonly GgufFile _file;
    private readonly ModelConfig _config;

    /// <summary>
    /// Initializes a new container from an opened GGUF file.
    /// </summary>
    /// <param name="file">The GGUF file. The container takes ownership of its disposal.</param>
    public GgufModelContainer(GgufFile file)
    {
        _file = file ?? throw new ArgumentNullException(nameof(file));
        _config = GgufModelConfigExtractor.Extract(_file.Metadata);
    }

    /// <summary>
    /// Opens a GGUF file and wraps it in a container.
    /// </summary>
    /// <param name="path">Path to the GGUF file.</param>
    /// <returns>A new <see cref="GgufModelContainer"/>.</returns>
    public static GgufModelContainer Open(string path) => new(GgufFile.Open(path));

    /// <inheritdoc/>
    public ModelConfig Config => _config;

    /// <summary>
    /// The underlying GGUF metadata.
    /// </summary>
    public GgufMetadata Metadata => _file.Metadata;

    /// <inheritdoc/>
    public long DataSectionOffset => _file.DataSectionOffset;

    /// <inheritdoc/>
    public IEnumerable<ModelTensor> Tensors => _file.TensorsByName.Values.Select(t => new ModelTensor(
        t.Name,
        t.Shape,
        t.QuantizationType,
        _file.DataBasePointer + (nint)t.DataOffset));

    /// <inheritdoc/>
    public bool TryGetTensor(string name, out ModelTensor tensor)
    {
        if (_file.TensorsByName.TryGetValue(name, out var desc))
        {
            tensor = new ModelTensor(
                desc.Name,
                desc.Shape,
                desc.QuantizationType,
                _file.DataBasePointer + (nint)desc.DataOffset);
            return true;
        }

        tensor = default;
        return false;
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<string> GetTensorNames() => (IReadOnlyCollection<string>)_file.TensorsByName.Keys;

    /// <inheritdoc/>
    public T? GetMetadata<T>(string key)
    {
        if (!_file.Metadata.ContainsKey(key))
            return default;

        // Optimized dispatch for common GGUF metadata types
        if (typeof(T) == typeof(string))
            return (T?)(object)_file.Metadata.GetString(key);
        if (typeof(T) == typeof(uint))
            return (T?)(object)_file.Metadata.GetUInt32(key);
        if (typeof(T) == typeof(int))
            return (T?)(object)_file.Metadata.GetInt32(key);
        if (typeof(T) == typeof(float))
            return (T?)(object)_file.Metadata.GetFloat32(key);
        if (typeof(T) == typeof(bool))
            return (T?)(object)_file.Metadata.GetBool(key);
        if (typeof(T) == typeof(string[]))
            return (T?)(object)_file.Metadata.GetStringArray(key);
        if (typeof(T) == typeof(uint[]))
            return (T?)(object)_file.Metadata.GetUInt32Array(key);

        // General fallback
        if (_file.Metadata.TryGetValue(key, out var entry) && entry.Value is T typed)
            return typed;

        return default;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _file.Dispose();
    }
}
