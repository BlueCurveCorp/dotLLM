using DotLLM.Core.Backends;
using DotLLM.Core.Tensors;
using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// GPU backend using AMD ROCm via HIP Runtime API P/Invoke.
/// </summary>
public sealed class RoCmBackend : IBackend
{
    /// <inheritdoc/>
    public int DeviceCount { get; }
 
    public RoCmBackend()
    {
        HipLibraryResolver.Register();
        HipApi.hipInit(0).ThrowOnError();
        HipApi.hipGetDeviceCount(out int count).ThrowOnError();
        DeviceCount = count;
    }

    /// <inheritdoc/>
    public ITensor AllocateOnDevice(int deviceId, TensorShape shape, DType dtype)
    {
        if (deviceId < 0)
            throw new ArgumentException($"RoCmBackend requires deviceId >= 0, got {deviceId}.", nameof(deviceId));
        if (deviceId >= DeviceCount)
            throw new ArgumentException($"Device {deviceId} not available (have {DeviceCount} GPU(s)).", nameof(deviceId));

        return HipTensor.Allocate(shape, dtype, deviceId);
    }

    /// <inheritdoc/>
    public void CopyBetweenDevices(ITensor source, ITensor destination)
    {
        if (source.ByteCount != destination.ByteCount)
        {
            throw new ArgumentException($"Source ({source.ByteCount} bytes) and destination ({destination.ByteCount} bytes) sizes differ.");
        }

        nuint bytes = (nuint)source.ByteCount;

        if (source.DeviceId == -1 && destination.DeviceId >= 0)
        {
            HipApi.hipMemcpyHtoD(destination.DataPointer, source.DataPointer, bytes).ThrowOnError();
        }
        else if (source.DeviceId >= 0 && destination.DeviceId == -1)
        {
            HipApi.hipMemcpyDtoH(destination.DataPointer, source.DataPointer, bytes).ThrowOnError();
        }
        else if (source.DeviceId >= 0 && destination.DeviceId >= 0)
        {
            HipApi.hipMemcpyDtoD(destination.DataPointer, source.DataPointer, bytes).ThrowOnError();
        }
        else
        {
            throw new ArgumentException("CPU-to-CPU copy is not a RoCmBackend operation.");
        }
    }

    /// <inheritdoc/>
    public void AllReduce(ReadOnlySpan<ITensor> tensors) =>
        throw new NotSupportedException("RoCmBackend does not support AllReduce (requires RCCL, Step 51).");

    /// <inheritdoc/>
    public void Send(ITensor tensor, int targetDevice) =>
        throw new NotSupportedException("RoCmBackend does not support Send (requires RCCL, Step 51).");

    /// <inheritdoc/>
    public ITensor Receive(int sourceDevice, TensorShape shape, DType dtype) =>
        throw new NotSupportedException("RoCmBackend does not support Receive (requires RCCL, Step 51).");

    /// <inheritdoc/>
    public void Dispose()
    {
        // No owned resources — contexts and streams are managed by HipTransformerModel.
    }
}
