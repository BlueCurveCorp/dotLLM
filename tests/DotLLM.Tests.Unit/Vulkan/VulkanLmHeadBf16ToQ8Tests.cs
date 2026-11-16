using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1816: the opt-in BF16 -> Q8_0 LM-head requantization streams rows through the staging buffer in chunks. The device bytes must equal
/// the CPU <see cref="Quantize"/> Q8_0 encoding of the BF16 rows, row for row, across several staging chunks (the 4 KiB minimum staging
/// buffer holds 60 rows of 68 bytes here, so 200 rows take 4 chunks: a row-offset slip inside or across chunks changes the bytes).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanLmHeadBf16ToQ8Tests
{
    [SkippableFact]
    public unsafe void Bf16ToQ8Upload_MatchesCpuQuantizeAcrossStagingChunks()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        const int M = 200, K = 64;
        var rng = new Random(1816);
        var bf16 = new ushort[M * K];
        for (int i = 0; i < bf16.Length; i++)
            bf16[i] = (ushort)(BitConverter.SingleToInt32Bits((float)(rng.NextDouble() - 0.5) * (1 + i % 7)) >> 16);

        long rowBytes = Dequantize.RowByteSize(K, QuantizationType.Q8_0);
        var expected = new byte[rowBytes * M];
        var row = new float[K];
        for (int m = 0; m < M; m++)
        {
            for (int k = 0; k < K; k++) row[k] = BitConverter.Int32BitsToSingle(bf16[m * K + k] << 16);
            Quantize.FromFloat32(row, K, QuantizationType.Q8_0, expected.AsSpan((int)(m * rowBytes), (int)rowBytes));
        }

        using var device = VulkanDevice.Create();
        using var staging = VulkanStagingBuffer.Create(device, 4096);
        Assert.True(staging.Capacity < rowBytes * M / 2, "the fixture must span several staging chunks");
        VulkanDevice.Buffer buf;
        long bytes;
        fixed (ushort* p = bf16)
            buf = VulkanQwen3MoeHybridWeights.UploadBf16AsQ8_0(device, staging, (nint)p, M, K, out bytes);
        using (buf)
        {
            Assert.Equal(expected.LongLength, bytes);
            var got = new float[bytes / 4];
            device.Download(buf, got);
            Assert.Equal(expected, MemoryMarshal.AsBytes(got.AsSpan()).ToArray());
        }
    }
}
