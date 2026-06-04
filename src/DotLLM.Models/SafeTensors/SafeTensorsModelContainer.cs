using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// SafeTensors implementation of <see cref="IModelContainer"/>.
/// </summary>
public sealed class SafeTensorsModelContainer : IModelContainer
{
    private readonly ModelConfig _config;
    private readonly IReadOnlyList<SafeTensorsShard> _shards;
    private readonly Dictionary<string, ModelTensor> _tensorsByName;
    private readonly List<nint> _repackagedBuffers = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="SafeTensorsModelContainer"/> class.
    /// </summary>
    /// <param name="config">The model configuration.</param>
    /// <param name="shards">The loaded shards.</param>
    private SafeTensorsModelContainer(ModelConfig config, IReadOnlyList<SafeTensorsShard> shards)
    {
        _config = config;
        _shards = shards;

        _tensorsByName = new Dictionary<string, ModelTensor>(StringComparer.Ordinal);
        var rawTensors = new Dictionary<string, ModelTensor>(StringComparer.Ordinal);
        
        foreach (var shard in _shards)
        {
            foreach (var kvp in shard.TensorsByName)
            {
                rawTensors[kvp.Key] = kvp.Value;
            }
        }

        // Repackage AWQ/GPTQ and map normal tensors
        SafeTensorsRepackager.Repackage(rawTensors, _tensorsByName, _repackagedBuffers);
    }

    /// <summary>
    /// Opens a SafeTensors model container from a file or directory path.
    /// </summary>
    /// <param name="path">The directory or file path.</param>
    /// <returns>An initialized container.</returns>
    public static SafeTensorsModelContainer Open(string path)
    {
        string directory = path;
        if (File.Exists(path))
        {
            directory = Path.GetDirectoryName(path) ?? string.Empty;
        }
        else if (!Directory.Exists(path))
        {
            throw new DirectoryNotFoundException($"SafeTensors directory or file not found: {path}");
        }

        string configPath = Path.Combine(directory, "config.json");
        if (!File.Exists(configPath))
            throw new FileNotFoundException($"config.json not found in {directory}");

        using var configStream = File.OpenRead(configPath);
        using var configDoc = JsonDocument.Parse(configStream);
        var config = SafeTensorsModelConfigExtractor.Extract(configDoc);

        var shards = new List<SafeTensorsShard>();
        string indexJsonPath = Path.Combine(directory, "model.safetensors.index.json");
        if (File.Exists(indexJsonPath))
        {
            // Sharded model
            using var indexStream = File.OpenRead(indexJsonPath);
            using var indexDoc = JsonDocument.Parse(indexStream);
            if (indexDoc.RootElement.TryGetProperty("weight_map", out var weightMap))
            {
                var shardFiles = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (var property in weightMap.EnumerateObject())
                {
                    shardFiles.Add(property.Value.GetString()!);
                }
                foreach (var file in shardFiles)
                {
                    shards.Add(SafeTensorsShard.Open(Path.Combine(directory, file)));
                }
            }
        }
        else
        {
            // Single file model
            string singlePath = Path.Combine(directory, "model.safetensors");
            if (!File.Exists(singlePath))
            {
                // Fallback to the provided path if it was a file
                if (File.Exists(path) && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
                    singlePath = path;
                else
                    throw new FileNotFoundException($"model.safetensors not found in {directory}");
            }
            shards.Add(SafeTensorsShard.Open(singlePath));
        }

        return new SafeTensorsModelContainer(config, shards);
    }

    /// <inheritdoc/>
    public ModelConfig Config => _config;

    /// <inheritdoc/>
    public long DataSectionOffset => 0;

    /// <inheritdoc/>
    public IEnumerable<ModelTensor> Tensors => _tensorsByName.Values;

    /// <inheritdoc/>
    public bool TryGetTensor(string name, out ModelTensor tensor)
    {
        return _tensorsByName.TryGetValue(name, out tensor);
    }

    /// <inheritdoc/>
    public IReadOnlyCollection<string> GetTensorNames() => _tensorsByName.Keys;

    /// <inheritdoc/>
    public T? GetMetadata<T>(string key)
    {
        return default; // Used by tokenizer factory, which we will abstract
    }

    /// <inheritdoc/>
    public unsafe void Dispose()
    {
        foreach (var shard in _shards)
        {
            shard.Dispose();
        }

        foreach (var ptr in _repackagedBuffers)
        {
            if (ptr != nint.Zero)
            {
                System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)ptr);
            }
        }
        _repackagedBuffers.Clear();
    }
}
