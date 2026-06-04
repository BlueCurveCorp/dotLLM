using DotLLM.Core.Tensors;
using DotLLM.RoCm;
using DotLLM.RoCm.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.RoCm;

/// <summary>
/// Tests for GPU tensor allocation and host↔device memory copies via HIP.
/// </summary>
[Trait("Category", "GPU")]
public class HipTensorTests : IDisposable
{
    private readonly HipContext? _ctx;

    public HipTensorTests()
    {
        if (HipDevice.IsAvailable())
            _ctx = HipContext.Create(0);
    }

    [SkippableFact]
    public void AllocateTensor_Succeeds()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");

        var shape = new TensorShape(16, 32);
        using var tensor = HipTensor.Allocate(shape, DType.Float16, 0);

        Assert.Equal(16 * 32, tensor.ElementCount);
        Assert.Equal(16 * 32 * 2, tensor.ByteCount); // FP16 = 2 bytes
        Assert.Equal(0, tensor.DeviceId);
        Assert.NotEqual(0, tensor.DataPointer);
    }

    [SkippableFact]
    public unsafe void RoundTrip_HostToDeviceToHost_PreservesData()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");

        // Create test data on host
        int n = 256;
        float[] hostSrc = new float[n];
        for (int i = 0; i < n; i++)
            hostSrc[i] = i * 0.1f;

        // Allocate on device
        long bytes = (long)n * sizeof(float);
        HipApi.hipMalloc(out nint devPtr, (nuint)bytes).ThrowOnError();

        try
        {
            // H2D
            fixed (float* srcPtr = hostSrc)
                HipApi.hipMemcpyHtoD(devPtr, (nint)srcPtr, (nuint)bytes).ThrowOnError();

            // D2H into a new array
            float[] hostDst = new float[n];
            fixed (float* dstPtr = hostDst)
                HipApi.hipMemcpyDtoH((nint)dstPtr, devPtr, (nuint)bytes).ThrowOnError();

            // Verify
            for (int i = 0; i < n; i++)
                Assert.Equal(hostSrc[i], hostDst[i]);
        }
        finally
        {
            HipApi.hipFree(devPtr);
        }
    }

    public void Dispose()
    {
        _ctx?.Dispose();
    }
}

