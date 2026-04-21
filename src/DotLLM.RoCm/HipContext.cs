using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// RAII wrapper around a HIP context. One context per device.
/// </summary>
public sealed class HipContext : IDisposable
{
    private nint _ctx;

    /// <summary>Device ordinal this context is bound to.</summary>
    public int DeviceId { get; }

    /// <summary>The native HIP context handle.</summary>
    public nint Handle => _ctx;

    private HipContext(nint ctx, int deviceId)
    {
        _ctx = ctx;
        DeviceId = deviceId;
    }

    /// <summary>
    /// Creates a new HIP context on the specified device.
    /// </summary>
    public static HipContext Create(int deviceId)
    {
        HipLibraryResolver.Register();
        HipApi.hipInit(0).ThrowOnError();
        HipApi.hipGetDevice(out int device).ThrowOnError();
        HipApi.hipCtxCreate(out nint ctx, 0, device).ThrowOnError();
        return new HipContext(ctx, deviceId);
    }

    /// <summary>Makes this context current on the calling thread.</summary>
    public void MakeCurrent()
    {
        HipApi.hipCtxSetCurrent(_ctx).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint ctx = Interlocked.Exchange(ref _ctx, nint.Zero);

        if (ctx != nint.Zero)
        {
            HipApi.hipCtxDestroy(ctx);
        }
    }
}
