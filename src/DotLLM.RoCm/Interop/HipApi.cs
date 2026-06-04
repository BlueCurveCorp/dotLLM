using System.Runtime.InteropServices;

namespace DotLLM.RoCm.Interop;

/// <summary>
/// Minimal P/Invoke declarations against AMD's HIP Runtime API (Module API subset).
/// libamdhip64.so — installed with ROCm driver.
/// All functions return hipError_t (int): 0 = hipSuccess, non-zero = error.
/// </summary>
internal static partial class HipApi
{
    private const string LibName = "amdhip64";

    // ── Initialization ──

    [LibraryImport(LibName)]
    internal static partial int hipInit(uint flags);

    // ── Device ──

    [LibraryImport(LibName)]
    internal static partial int hipGetDevice(out int device);

    [LibraryImport(LibName)]
    internal static partial int hipGetDeviceCount(out int count);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceGetName(
        [MarshalAs(UnmanagedType.LPArray)] byte[] name, int len, int device);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceTotalMem(out nuint bytes, int device);

    [LibraryImport(LibName)]
    internal static partial int hipMemGetInfo(out nuint free, out nuint total);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceGetAttribute(
        out int value, int attribute, int device);

    // ── Context ──

    [LibraryImport(LibName)]
    internal static partial int hipCtxCreate(out nint ctx, uint flags, int device);

    [LibraryImport(LibName)]
    internal static partial int hipCtxDestroy(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int hipCtxSetCurrent(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int hipCtxGetCurrent(out nint ctx);

    [LibraryImport(LibName)]
    internal static partial int hipCtxGetDevice(out int device);

    // ── Module (code object loading) ──

    [LibraryImport(LibName)]
    internal static partial int hipModuleLoadData(out nint module, nint imagePtr);

    [LibraryImport(LibName)]
    internal static partial int hipModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport(LibName)]
    internal static partial int hipModuleUnload(nint module);

    // ── Kernel launch ──

    [LibraryImport(LibName)]
    internal static partial int hipModuleLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams, nint extra);

    // ── Memory ──

    [LibraryImport(LibName)]
    internal static partial int hipMalloc(out nint devicePtr, nuint size);

    [LibraryImport(LibName)]
    [SuppressGCTransition]
    internal static partial int hipFree(nint devicePtr);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyHtoD(nint dst, nint src, nuint size);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoH(nint dst, nint src, nuint size);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoD(nint dst, nint src, nuint size);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyHtoDAsync(
        nint dst, nint src, nuint size, nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoHAsync(
        nint dst, nint src, nuint size, nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoDAsync(nint dst, nint src, nuint size, nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipMemset(nint dst, int value, nuint size);

    // ── Streams ──

    [LibraryImport(LibName)]
    internal static partial int hipStreamCreate(out nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipStreamDestroy(nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipStreamSynchronize(nint stream);

    // ── Error ──

    [LibraryImport(LibName)]
    internal static partial nint hipGetErrorName(int error);

    [LibraryImport(LibName)]
    internal static partial nint hipGetErrorString(int error);

    // ── Device attribute constants ──

    internal const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
    internal const int HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
    internal const int HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 236;
    internal const int HIP_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;
    internal const int HIP_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8;
    internal const int HIP_DEVICE_ATTRIBUTE_WARP_SIZE = 64;
}
