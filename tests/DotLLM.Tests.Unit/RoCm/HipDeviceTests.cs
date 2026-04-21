using DotLLM.RoCm;
using Xunit;

namespace DotLLM.Tests.Unit.RoCm;

/// <summary>
/// Tests for ROCm/HIP device detection and basic device queries.
/// Skip if no AMD GPU is available.
/// </summary>
[Trait("Category", "GPU")]
public class HipDeviceTests
{
    [SkippableFact]
    public void IsAvailable_ReturnsTrue_WhenGpuPresent()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");

        Assert.True(HipDevice.IsAvailable());
    }

    [SkippableFact]
    public void GetDeviceCount_ReturnsAtLeastOne()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");

        int count = HipDevice.GetDeviceCount();
        Assert.True(count >= 1);
    }

    [SkippableFact]
    public void GetDevice_ReturnsValidInfo()
    {
        Skip.IfNot(HipDevice.IsAvailable(), "No ROCm GPU available");

        var device = HipDevice.GetDevice(0);

        Assert.False(string.IsNullOrEmpty(device.Name));
        Assert.True(device.TotalMemoryBytes > 0);
        Assert.True(device.MultiprocessorCount > 0);
    }
}

