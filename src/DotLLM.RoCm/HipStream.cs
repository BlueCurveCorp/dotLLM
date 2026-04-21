using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// RAII wrapper around a HIP stream. Provides ordered execution of GPU operations.
/// </summary>
public sealed class HipStream : IDisposable
{
    private nint _stream;

    /// <summary>The native HIP stream handle.</summary>
    public nint Handle => _stream;

    private HipStream(nint stream)
    {
        _stream = stream;
    }

    /// <summary>Creates a new HIP stream.</summary>
    public static HipStream Create()
    {
        int errorCode = HipApi.hipStreamCreate(out nint stream);

        errorCode.ThrowOnError();

        return new HipStream(stream);
    }

    /// <summary>Blocks the host thread until all operations on this stream complete.</summary>
    public void Synchronize()
    {
        HipApi.hipStreamSynchronize(_stream).ThrowOnError();
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint stream = Interlocked.Exchange(ref _stream, nint.Zero);

        if (stream != nint.Zero)
        {
            HipApi.hipStreamDestroy(stream);
        }
    }
}
