using DotLLM.Core.Configuration;
using DotLLM.RoCm;
using DotLLM.RoCm.Interop;
using Xunit;

namespace DotLLM.Tests.Unit.RoCm;

/// <summary>
/// Tests individual HIP kernels against CPU reference implementations.
/// </summary>
[Trait("Category", "GPU")]
public class HipKernelTests : IDisposable
{
    private readonly HipContext? _ctx;
    private readonly HipStream? _stream;
    private readonly HipKernels? _kernels;

    public HipKernelTests()
    {
        if (!HipDevice.IsAvailable()) return;

        _ctx = HipContext.Create(0);
        _stream = HipStream.Create();

        // Find HSACO directory
        var hsacoDir = FindHsacoDir();
        if (hsacoDir != null)
            _kernels = new HipKernels(hsacoDir);
    }

    private static string? FindHsacoDir()
    {
        // Try relative to test assembly, then relative to repo root
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "hsaco"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "hsaco"),
        };

        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.hsaco").Length > 0)
                return full;
        }
        return null;
    }

    [SkippableFact]
    public unsafe void Add_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");
        Skip.If(_kernels == null, "HSACO files not found");

        int n = 128;
        nint s = _stream!.Handle;

        // Generate random FP16 data on host
        var rng = new Random(42);
        ushort[] aHost = new ushort[n];
        ushort[] bHost = new ushort[n];
        for (int i = 0; i < n; i++)
        {
            aHost[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextSingle() * 2 - 1));
            bHost[i] = BitConverter.HalfToUInt16Bits((Half)(rng.NextSingle() * 2 - 1));
        }

        long bytes = (long)n * sizeof(ushort);

        // Allocate device memory
        HipApi.hipMalloc(out nint devA, (nuint)bytes).ThrowOnError();
        HipApi.hipMalloc(out nint devB, (nuint)bytes).ThrowOnError();
        HipApi.hipMalloc(out nint devC, (nuint)bytes).ThrowOnError();

        try
        {
            // Upload
            fixed (ushort* pA = aHost) HipApi.hipMemcpyHtoD(devA, (nint)pA, (nuint)bytes).ThrowOnError();
            fixed (ushort* pB = bHost) HipApi.hipMemcpyHtoD(devB, (nint)pB, (nuint)bytes).ThrowOnError();

            // Launch kernel
            _kernels!.LaunchAdd(devA, devB, devC, n, s);
            _stream!.Synchronize();

            // Download result
            ushort[] cHost = new ushort[n];
            fixed (ushort* pC = cHost) HipApi.hipMemcpyDtoH((nint)pC, devC, (nuint)bytes).ThrowOnError();

            // Compare with CPU reference
            for (int i = 0; i < n; i++)
            {
                float expected = (float)BitConverter.UInt16BitsToHalf(aHost[i]) +
                                 (float)BitConverter.UInt16BitsToHalf(bHost[i]);
                float actual = (float)BitConverter.UInt16BitsToHalf(cHost[i]);
                Assert.True(MathF.Abs(expected - actual) < 0.01f,
                    $"Mismatch at {i}: expected {expected}, got {actual}");
            }
        }
        finally
        {
            HipApi.hipFree(devA);
            HipApi.hipFree(devB);
            HipApi.hipFree(devC);
        }
    }

    [SkippableFact]
    public unsafe void ConvertF32ToF16_RoundTrip()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");
        Skip.If(_kernels == null, "HSACO files not found");

        int n = 64;
        nint s = _stream!.Handle;

        // Source F32 data
        float[] srcHost = new float[n];
        for (int i = 0; i < n; i++)
            srcHost[i] = (i - 32) * 0.5f;

        long f32Bytes = (long)n * sizeof(float);
        long f16Bytes = (long)n * sizeof(ushort);

        HipApi.hipMalloc(out nint devF32, (nuint)f32Bytes).ThrowOnError();
        HipApi.hipMalloc(out nint devF16, (nuint)f16Bytes).ThrowOnError();
        HipApi.hipMalloc(out nint devF32Back, (nuint)f32Bytes).ThrowOnError();

        try
        {
            // Upload F32
            fixed (float* p = srcHost) HipApi.hipMemcpyHtoD(devF32, (nint)p, (nuint)f32Bytes).ThrowOnError();

            // F32 → F16
            _kernels!.LaunchConvertF32ToF16(devF32, devF16, n, s);

            // F16 → F32
            _kernels!.LaunchConvertF16ToF32(devF16, devF32Back, n, s);
            _stream!.Synchronize();

            // Download
            float[] dstHost = new float[n];
            fixed (float* p = dstHost) HipApi.hipMemcpyDtoH((nint)p, devF32Back, (nuint)f32Bytes).ThrowOnError();

            // Compare (FP16 round-trip loses precision)
            for (int i = 0; i < n; i++)
            {
                float expected = (float)(Half)srcHost[i]; // simulate FP16 round-trip
                Assert.True(MathF.Abs(expected - dstHost[i]) < 0.001f,
                    $"Mismatch at {i}: expected {expected}, got {dstHost[i]}");
            }
        }
        finally
        {
            HipApi.hipFree(devF32);
            HipApi.hipFree(devF16);
            HipApi.hipFree(devF32Back);
        }
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}

