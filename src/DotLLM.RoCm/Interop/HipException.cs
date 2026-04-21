namespace DotLLM.RoCm.Interop;

/// <summary>
/// Exception thrown when a HIP Runtime API or rocBLAS call fails.
/// </summary>
public sealed class HipException : Exception
{
    /// <summary>HIP error code.</summary>
    public int ErrorCode { get; }

    /// <summary>Creates a HIP exception with the given error code and message.</summary>
    public HipException(int errorCode, string message)
        : base($"HIP error {errorCode}: {message}")
    {
        ErrorCode = errorCode;
    }
}
