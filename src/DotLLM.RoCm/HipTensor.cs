using DotLLM.Core.Tensors;
using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// GPU-resident tensor backed by <c>hipMalloc</c>. Owns the device memory
/// and frees it on disposal. <see cref="DataPointer"/> is an opaque device pointer.
/// </summary>
public sealed class HipTensor : ITensor
{
    private nint _ptr;

    /// <inheritdoc/>
    public TensorShape Shape { get; }

    /// <inheritdoc/>
    public DType DType { get; }

    /// <inheritdoc/>
    public int DeviceId { get; }

    /// <inheritdoc/>
    public nint DataPointer => _ptr;

    /// <inheritdoc/>
    public TensorMetadata Metadata => new(Shape, DType, DeviceId, _ptr);

    /// <inheritdoc/>
    public long ElementCount => Shape.ElementCount;

    /// <inheritdoc/>
    public long ByteCount { get; }

    private HipTensor(TensorShape shape, DType dtype, int deviceId, nint ptr, long byteCount)
    {
        Shape = shape;
        DType = dtype;
        DeviceId = deviceId;
        _ptr = ptr;
        ByteCount = byteCount;
    }

    /// <summary>
    /// Allocates a GPU tensor of the given shape and data type.
    /// </summary>
    public static HipTensor Allocate(TensorShape shape, DType dtype, int deviceId)
    {
        long bytes = dtype.ComputeByteCount(shape.ElementCount);

        if (bytes <= 0)
        {
            throw new ArgumentException($"Cannot allocate tensor with {shape.ElementCount} elements of {dtype} (computed byte count={bytes}).");
        }

        HipApi.hipMalloc(out nint ptr, (nuint)bytes).ThrowOnError();
        return new HipTensor(shape, dtype, deviceId, ptr, bytes);
    }

    /// <summary>
    /// Allocates a GPU tensor with an explicit byte count (for quantized types).
    /// </summary>
    public static HipTensor AllocateBytes(TensorShape shape, DType dtype, int deviceId, long byteCount)
    {
        if (byteCount <= 0)
        {
            throw new ArgumentException($"Byte count must be positive, got {byteCount}.");
        }

        HipApi.hipMalloc(out nint ptr, (nuint)byteCount).ThrowOnError();
        return new HipTensor(shape, dtype, deviceId, ptr, byteCount);
    }

    /// <summary>
    /// Wraps an existing device pointer as a non-owning tensor view.
    /// The caller is responsible for the lifetime of the pointer.
    /// </summary>
    internal static HipTensor WrapExisting(TensorShape shape, DType dtype, int deviceId, nint devicePtr, long byteCount) => new HipTensor(shape, dtype, deviceId, devicePtr, byteCount) { _ownsMemory = false };

    private bool _ownsMemory = true;

    /// <inheritdoc/>
    public void Dispose()
    {
        nint ptr = Interlocked.Exchange(ref _ptr, nint.Zero);

        if (ptr != nint.Zero && _ownsMemory)
        {
            HipApi.hipFree(ptr);
        }
    }
}
