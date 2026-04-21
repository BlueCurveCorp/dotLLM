using DotLLM.RoCm;
using DotLLM.RoCm.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.RoCm;

/// <summary>
/// Tests rocBLAS GEMM correctness against CPU reference for various matrix sizes.
/// Mirrors CudaGemmTest using HipGemm.LinearF16 / HipCublasHandle.
/// </summary>
[Trait("Category", "GPU")]
public class HipGemmTest : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly HipContext? _ctx;
    private readonly HipStream? _stream;
    private readonly HipCublasHandle? _rocblas;

    public HipGemmTest(ITestOutputHelper output)
    {
        _out = output;
        if (!HipDevice.IsAvailable()) return;
        _ctx = HipContext.Create(0);
        _stream = HipStream.Create();
        _rocblas = HipCublasHandle.Create();
        _rocblas.SetStream(_stream.Handle);
    }

    [SkippableFact]
    public unsafe void LinearF16_GEMV_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");
        TestLinear(m: 1, k: 64, n: 32, "GEMV m=1");
    }

    [SkippableFact]
    public unsafe void LinearF16_GEMM_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");
        TestLinear(m: 5, k: 64, n: 32, "GEMM m=5");
    }

    [SkippableFact]
    public unsafe void LinearF16_LargeGEMM_MatchesCpuReference()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");
        TestLinear(m: 5, k: 576, n: 576, "GEMM 576x576");
    }

    private unsafe void TestLinear(int m, int k, int n, string label)
    {
        var rng = new Random(42);

        // Generate random FP16 data
        Half[] xHost = new Half[m * k];
        Half[] wHost = new Half[n * k];
        for (int i = 0; i < xHost.Length; i++)
            xHost[i] = (Half)(rng.NextSingle() * 2 - 1);
        for (int i = 0; i < wHost.Length; i++)
            wHost[i] = (Half)(rng.NextSingle() * 2 - 1);

        // CPU reference: Y[m,n] = X[m,k] @ W^T (W is [n,k])
        float[] yCpu = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float sum = 0;
                for (int p = 0; p < k; p++)
                    sum += (float)xHost[i * k + p] * (float)wHost[j * k + p];
                yCpu[i * n + j] = sum;
            }

        // GPU: allocate, upload, compute
        long xBytes = (long)m * k * sizeof(ushort);
        long wBytes = (long)n * k * sizeof(ushort);
        long yBytes = (long)m * n * sizeof(ushort);

        HipApi.hipMalloc(out nint devX, (nuint)xBytes).ThrowOnError();
        HipApi.hipMalloc(out nint devW, (nuint)wBytes).ThrowOnError();
        HipApi.hipMalloc(out nint devY, (nuint)yBytes).ThrowOnError();

        try
        {
            fixed (Half* px = xHost) HipApi.hipMemcpyHtoD(devX, (nint)px, (nuint)xBytes).ThrowOnError();
            fixed (Half* pw = wHost) HipApi.hipMemcpyHtoD(devW, (nint)pw, (nuint)wBytes).ThrowOnError();

            HipGemm.LinearF16(_rocblas!.Handle, devX, devW, devY, m, k, n, _stream!.Handle);
            _stream!.Synchronize();

            // Download result
            Half[] yGpu = new Half[m * n];
            fixed (Half* py = yGpu) HipApi.hipMemcpyDtoH((nint)py, devY, (nuint)yBytes).ThrowOnError();

            // Compare
            float maxDiff = 0;
            for (int i = 0; i < m * n; i++)
            {
                float diff = MathF.Abs(yCpu[i] - (float)yGpu[i]);
                if (diff > maxDiff) maxDiff = diff;
            }

            _out.WriteLine($"{label}: max diff = {maxDiff:F4} (m={m}, k={k}, n={n})");

            // FP16 output should be close to FP32 reference
            // For k=576, max diff should be < 1.0 (FP16 accumulation noise)
            Assert.True(maxDiff < 2.0f, $"{label}: max diff {maxDiff} too large");
        }
        finally
        {
            HipApi.hipFree(devX);
            HipApi.hipFree(devW);
            HipApi.hipFree(devY);
        }
    }

    public void Dispose()
    {
        _rocblas?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}

