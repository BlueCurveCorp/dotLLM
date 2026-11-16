using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Server;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1822: a <c>--device auto</c> CPU fallback must stay visible for the model it happened to. It used to live only on the state, set by
/// the startup load, and every on-demand (chat-triggered) load dropped it. That is the very path where a reload silently landed on the
/// CPU (#1821). It now travels on <see cref="ServerOptions"/>: an on-demand load carries its own warning, a load without one clears it,
/// and a stashed model brings its warning back when it is reactivated.
/// </summary>
public sealed class DeviceFallbackTravelTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("dotllm-1822-fb-").FullName;
    private readonly List<ServerState> _states = new();

    public void Dispose()
    {
        foreach (var s in _states) { try { s.Dispose(); } catch { } }
        try { Directory.Delete(_dir, true); } catch { }
    }

    private string File(string name)
    {
        string path = Path.Combine(_dir, name);
        System.IO.File.WriteAllBytes(path, "GGUF"u8.ToArray().Concat(new byte[96]).ToArray());
        return path;
    }

    private sealed class StubModel : IModel
    {
        public ModelConfig Config { get; } = new() { Architecture = Architecture.Llama, VocabSize = 8, HiddenSize = 8, NumLayers = 1, IntermediateSize = 8, NumAttentionHeads = 1, NumKvHeads = 1, HeadDim = 8, MaxSequenceLength = 8 };
        public long ComputeMemoryBytes => 0;
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => throw new NotSupportedException();
        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) => throw new NotSupportedException();
        public void Dispose() { }
    }

    /// <summary>A state whose loader mimics <c>LoadModel</c> under <c>--device auto</c>: files named <c>*fallback*</c> land on the CPU with a warning.</summary>
    private ServerState NewState(int maxResident = 1)
    {
        var state = new ServerState
        {
            Options = new ServerOptions { Model = "", ModelId = "none", Device = "auto", UsePaged = false, PromptCacheEnabled = false },
            Residency = new ModelResidencyManager { MaxResidentModels = maxResident },
        };
        state.ModelLoader = (path, options) =>
        {
            bool fellBack = path.Contains("fallback", StringComparison.Ordinal);
            var loaded = new ServerState
            {
                Options = options with { ResolvedDevice = fellBack ? "cpu" : "vulkan", DeviceFallbackWarning = null },
                Model = new StubModel(),
                LoadedModelPath = path,
            };
            if (fellBack) loaded.DeviceFallbackWarning = "vulkan: out of device memory";   // as LoadModel sets it
            return loaded;
        };
        _states.Add(state);
        return state;
    }

    [Fact]
    public async Task OnDemandLoad_CarriesItsFallback_ALoadWithoutOneClearsIt()
    {
        var state = NewState();
        Assert.Null(await state.EnsureActiveAsync(File("a-fallback.gguf"), null, CancellationToken.None));
        Assert.Equal("vulkan: out of device memory", state.DeviceFallbackWarning);
        Assert.Equal("cpu", state.ActiveDevice);
        var status = state.GetStatus();
        Assert.True(status.DeviceFallback);
        Assert.Equal("cpu", status.Device);

        Assert.Null(await state.EnsureActiveAsync(File("b.gguf"), null, CancellationToken.None));
        Assert.Null(state.DeviceFallbackWarning);
        Assert.Equal("vulkan", state.ActiveDevice);
        Assert.False(state.GetStatus().DeviceFallback);
    }

    [Fact]
    public async Task StashedModel_BringsItsFallbackBack_OnReactivation()
    {
        var state = NewState(maxResident: 2);
        string a = File("a-fallback.gguf");
        Assert.Null(await state.EnsureActiveAsync(a, null, CancellationToken.None));
        string idA = state.Options.ModelId;
        Assert.Null(await state.EnsureActiveAsync(File("b.gguf"), null, CancellationToken.None));
        Assert.Null(state.DeviceFallbackWarning);

        var stashed = Assert.Single(state.ListResidentModels(), m => !m.IsActive);
        Assert.Equal("cpu", stashed.Device);
        Assert.NotNull(stashed.DeviceFallbackReason);

        Assert.Null(await state.EnsureActiveAsync(idA, null, CancellationToken.None));
        Assert.Equal("vulkan: out of device memory", state.DeviceFallbackWarning);
        Assert.Equal("cpu", state.ActiveDevice);
    }
}
