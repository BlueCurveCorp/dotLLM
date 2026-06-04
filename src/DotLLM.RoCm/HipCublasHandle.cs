using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// RAII wrapper around a rocBLAS handle. Enables Matrix Core math mode on creation.
/// </summary>
public sealed class HipCublasHandle : IDisposable
{
    private nint _handle;

    /// <summary>The native rocBLAS handle.</summary>
    public nint Handle => _handle;

    private HipCublasHandle(nint handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Creates a new rocBLAS handle with Matrix Core math mode enabled.
    /// </summary>
    public static HipCublasHandle Create()
    {
        RocBlasApi.rocblas_create_handle(out nint handle).ThrowOnRocBlasError();
        RocBlasApi.rocblas_set_math_mode(handle, RocBlasApi.ROCBLAS_XF32_XDL_MATH_OP).ThrowOnRocBlasError();
        return new HipCublasHandle(handle);
    }

    /// <summary>Binds this rocBLAS handle to the given HIP stream.</summary>
    public void SetStream(nint streamHandle)
    {
        RocBlasApi.rocblas_set_stream(_handle, streamHandle).ThrowOnRocBlasError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint handle = Interlocked.Exchange(ref _handle, nint.Zero);

        if (handle != nint.Zero)
        {
            RocBlasApi.rocblas_destroy_handle(handle);
        }   
    }
}
