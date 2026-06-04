using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.RoCm.Interop;

/// <summary>
/// Resolves "amdhip64" and "rocblas" library names to platform-specific paths.
/// Linux: libamdhip64.so, librocblas.so
/// Windows: amdhip64.dll, rocblas.dll
/// </summary>
internal static class HipLibraryResolver
{
    private static int _registered;

    /// <summary>
    /// Registers the resolver. Safe to call multiple times (idempotent).
    /// </summary>
    internal static void Register()
    {
        if (Interlocked.Exchange(ref _registered, 1) != 0)
        {
            return;
        }

        NativeLibrary.SetDllImportResolver(typeof(HipLibraryResolver).Assembly, ResolveHipLibrary);
    }

    private static nint ResolveHipLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "amdhip64")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("amdhip64.dll", out nint handle))
                {
                    return handle;
                }
            }
            else
            {
                if (NativeLibrary.TryLoad("libamdhip64.so", out nint handle))
                {
                    return handle;
                }
            }
        }

        if (libraryName == "rocblas")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                if (NativeLibrary.TryLoad("rocblas.dll", out nint handle))
                {
                    return handle;
                }
            }
            else
            {
                if (NativeLibrary.TryLoad("librocblas.so", out nint handle))
                {
                    return handle;
                }
            }
        }

        return 0; // fall through to default resolution
    }
}
