using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// Queries AMD GPU device properties via the HIP Runtime API.
/// </summary>
public sealed class HipDevice
{
    public int Ordinal { get; }
    public string Name { get; }
    public long TotalMemoryBytes { get; }
    public int ComputeCapabilityMajor { get; }
    public int ComputeCapabilityMinor { get; }
    public int MultiprocessorCount { get; }

    private HipDevice(int ordinal, string name, long totalMem, int ccMajor, int ccMinor, int smCount)
    {
        Ordinal = ordinal;
        Name = name;
        TotalMemoryBytes = totalMem;
        ComputeCapabilityMajor = ccMajor;
        ComputeCapabilityMinor = ccMinor;
        MultiprocessorCount = smCount;
    }

    public string TotalMemoryFormatted => $"{TotalMemoryBytes / (1024.0 * 1024 * 1024):F1} GB";
    public string ComputeCapability => $"{ComputeCapabilityMajor}.{ComputeCapabilityMinor}";

    /// <summary>
    /// Checks whether ROCm is available on this system.
    /// </summary>
    public static bool IsAvailable()
    {
        try
        {
            string hipLib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                ? "amdhip64.dll" : "libamdhip64.so";

            if (!NativeLibrary.TryLoad(hipLib, out nint handle))
            {
                return false;
            }

            NativeLibrary.Free(handle);
            return ProbeGpuCount();
        }
        catch
        {
            return false;
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool ProbeGpuCount()
    {
        HipLibraryResolver.Register();
        HipApi.hipInit(0).ThrowOnError();
        HipApi.hipGetDeviceCount(out int count).ThrowOnError();
        return count > 0;
    }

    public static int GetDeviceCount()
    {
        HipLibraryResolver.Register();
        HipApi.hipInit(0).ThrowOnError();
        HipApi.hipGetDeviceCount(out int count).ThrowOnError();
        return count;
    }

    public static HipDevice GetDevice(int ordinal)
    {
        HipLibraryResolver.Register();
        HipApi.hipInit(0).ThrowOnError();

        HipApi.hipGetDevice(out int device).ThrowOnError();

        byte[] nameBuffer = new byte[256];
        HipApi.hipDeviceGetName(nameBuffer, nameBuffer.Length, device).ThrowOnError();
        int nullIdx = Array.IndexOf(nameBuffer, (byte)0);
        string name = Encoding.ASCII.GetString(nameBuffer, 0, nullIdx >= 0 ? nullIdx : nameBuffer.Length).Trim();

        HipApi.hipDeviceTotalMem(out nuint totalMem, device).ThrowOnError();

        HipApi.hipDeviceGetAttribute(out int ccMajor, HipApi.HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device).ThrowOnError();
        HipApi.hipDeviceGetAttribute(out int ccMinor, HipApi.HIP_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device).ThrowOnError();
        HipApi.hipDeviceGetAttribute(out int smCount, HipApi.HIP_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device).ThrowOnError();

        return new HipDevice(ordinal, name, (long)totalMem, ccMajor, ccMinor, smCount);
    }

    public override string ToString() =>
        $"GPU {Ordinal}: {Name} ({TotalMemoryFormatted}, compute_{ComputeCapabilityMajor}.{ComputeCapabilityMinor}, {MultiprocessorCount} SMs)";
}
