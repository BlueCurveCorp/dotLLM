using System.Runtime.InteropServices;

namespace DotLLM.RoCm.Interop;

/// <summary>
/// Extension methods for checking HIP and rocBLAS return codes.
/// </summary>
internal static class HipErrorHelper
{
    /// <summary>
    /// Throws <see cref="HipException"/> if <paramref name="result"/> is non-zero (hipSuccess = 0).
    /// </summary>
    internal static void ThrowOnError(this int result)
    {
        if (result == 0)
        {
            return;
        }

        string message = "Unknown HIP error";

        nint strPtr = HipApi.hipGetErrorString(result);

        if (strPtr != nint.Zero)
        {
            message = Marshal.PtrToStringAnsi(strPtr) ?? message;
        }

        throw new HipException(result, message);
    }

    /// <summary>
    /// Throws <see cref="HipException"/> for rocBLAS errors with a "rocBLAS" prefix.
    /// </summary>
    internal static void ThrowOnRocBlasError(this int result)
    {
        if (result == 0)
        {
            return;
        }

        string message = result switch
        {
            1 => "rocBLAS_STATUS_NOT_INITIALIZED",
            3 => "rocBLAS_STATUS_ALLOC_FAILED",
            7 => "rocBLAS_STATUS_INVALID_VALUE",
            8 => "rocBLAS_STATUS_ARCH_MISMATCH",
            9 => "rocBLAS_STATUS_MAPPING_ERROR",
            11 => "rocBLAS_STATUS_EXECUTION_FAILED",
            12 => "rocBLAS_STATUS_INTERNAL_ERROR",
            13 => "rocBLAS_STATUS_NOT_SUPPORTED",
            15 => "rocBLAS_STATUS_ARITHMETIC_EXCEPTION",
            _ => $"Unknown rocBLAS error"
        };

        throw new HipException(result, $"rocBLAS: {message}");
    }
}
