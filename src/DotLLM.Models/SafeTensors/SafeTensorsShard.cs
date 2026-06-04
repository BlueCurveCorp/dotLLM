using System.IO.MemoryMappedFiles;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Represents a single .safetensors memory-mapped file shard.
/// </summary>
public sealed unsafe class SafeTensorsShard : IDisposable
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _basePointer;
    private bool _disposed;

    /// <summary>The path to the shard file.</summary>
    public string FilePath { get; }

    /// <summary>Dictionary mapping tensor names to descriptors.</summary>
    public IReadOnlyDictionary<string, ModelTensor> TensorsByName { get; }

    private SafeTensorsShard(
        string filePath,
        IReadOnlyDictionary<string, ModelTensor> tensorsByName,
        MemoryMappedFile mmf,
        MemoryMappedViewAccessor accessor,
        byte* basePointer)
    {
        FilePath = filePath;
        TensorsByName = tensorsByName;
        _mmf = mmf;
        _accessor = accessor;
        _basePointer = basePointer;
    }

    /// <summary>
    /// Opens and memory-maps a single SafeTensors file.
    /// </summary>
    /// <param name="filePath">The path to the .safetensors file.</param>
    /// <returns>A loaded <see cref="SafeTensorsShard"/>.</returns>
    public static SafeTensorsShard Open(string filePath)
    {
        MemoryMappedFile? mmf = null;
        MemoryMappedViewAccessor? accessor = null;
        byte* basePointer = null;

        try
        {
            var fileInfo = new FileInfo(filePath);
            if (!fileInfo.Exists)
                throw new FileNotFoundException($"SafeTensors file not found: {filePath}");

            mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);

            if (fileInfo.Length < 8)
                throw new InvalidDataException("File is too small to be a SafeTensors file.");

            long headerSize = BitConverter.ToInt64(new ReadOnlySpan<byte>(basePointer, 8));
            if (headerSize <= 0 || 8 + headerSize > fileInfo.Length)
                throw new InvalidDataException("Invalid header size in SafeTensors file.");

            using var stream = new UnmanagedMemoryStream(basePointer + 8, headerSize);
            using var doc = JsonDocument.Parse(stream);

            nint dataBasePointer = (nint)(basePointer + 8 + headerSize);
            var tensors = new Dictionary<string, ModelTensor>(StringComparer.Ordinal);

            foreach (var property in doc.RootElement.EnumerateObject())
            {
                if (property.Name == "__metadata__")
                    continue;

                var tensorObj = property.Value;
                if (!tensorObj.TryGetProperty("dtype", out var dtypeElem) ||
                    !tensorObj.TryGetProperty("shape", out var shapeElem) ||
                    !tensorObj.TryGetProperty("data_offsets", out var offsetsElem))
                {
                    continue; // Skip invalid tensor entries
                }

                string dtypeStr = dtypeElem.GetString() ?? "";
                QuantizationType quantType = ParseDtype(dtypeStr);

                var dims = new List<int>();
                foreach (var dim in shapeElem.EnumerateArray())
                {
                    dims.Add(dim.GetInt32());
                }
                var shape = new TensorShape([.. dims]);

                long offsetStart = offsetsElem[0].GetInt64();
                
                tensors[property.Name] = new ModelTensor(
                    property.Name,
                    shape,
                    quantType,
                    dataBasePointer + (nint)offsetStart);
            }

            return new SafeTensorsShard(filePath, tensors, mmf, accessor, basePointer);
        }
        catch
        {
            if (basePointer != null && accessor != null)
                accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            accessor?.Dispose();
            mmf?.Dispose();
            throw;
        }
    }

    private static QuantizationType ParseDtype(string dtype)
    {
        return dtype switch
        {
            "F16" => QuantizationType.F16,
            "F32" => QuantizationType.F32,
            "BF16" => QuantizationType.BF16,
            "I8" or "I16" or "I32" or "I64" or "F64" or "BOOL" or "U8" => throw new NotSupportedException($"Dtype {dtype} is not natively mapped to a QuantizationType currently."),
            _ => throw new InvalidDataException($"Unknown SafeTensors dtype: {dtype}")
        };
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_basePointer != null && _accessor != null)
        {
            _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            _basePointer = null;
        }

        _accessor?.Dispose();
        _mmf?.Dispose();

        _accessor = null;
        _mmf = null;
    }
}
