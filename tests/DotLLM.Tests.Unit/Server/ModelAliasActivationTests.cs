using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.HuggingFace;
using DotLLM.Server;
using DotLLM.Core.Attention;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1821: a request naming the active model by an alias - the short id it was served as, its repo id, its long id, its file path -
/// must hit the active model, not evict and reload it. Before the fix the request string was compared to the resolved long id only, so
/// <c>dotllm serve &lt;short id&gt;</c> plus a client sending that short id reloaded the model on EVERY request (prompt cache lost,
/// <c>--device auto</c> re-run). The loader is replaced by a counting stub, so no real weights are read; every model store is a temp dir.
/// </summary>
[Collection("SequentialFileIO")]   // the model-store env vars are process-global
public sealed class ModelAliasActivationTests : IDisposable
{
    private const string Repo = "acme/Tiny-GGUF";
    private const string Stem = "tiny-Q4_K_M";
    private const string OtherStem = "other-Q8_0";
    private static readonly string[] EnvVars = ["DOTLLM_MODELS_DIR", "DOTLLM_PROFILES_DIR", "OLLAMA_MODELS", "HF_HUB_CACHE"];

    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-1821-" + Guid.NewGuid().ToString("N"));
    private readonly Dictionary<string, string?> _saved = new();
    private readonly List<ServerState> _states = new();
    private readonly string _path;
    private readonly string _otherPath;
    private int _loads;

    public ModelAliasActivationTests()
    {
        foreach (var v in EnvVars) _saved[v] = Environment.GetEnvironmentVariable(v);
        Environment.SetEnvironmentVariable("DOTLLM_MODELS_DIR", Path.Combine(_root, "models"));
        Environment.SetEnvironmentVariable("DOTLLM_PROFILES_DIR", Path.Combine(_root, "profiles"));
        Environment.SetEnvironmentVariable("OLLAMA_MODELS", Path.Combine(_root, "ollama"));
        Environment.SetEnvironmentVariable("HF_HUB_CACHE", Path.Combine(_root, "hub"));
        _path = WriteGguf(Repo, Stem + ".gguf");
        _otherPath = WriteGguf("acme/Other-GGUF", OtherStem + ".gguf");
    }

    public void Dispose()
    {
        foreach (var s in _states) { try { s.Dispose(); } catch { } }
        foreach (var (k, v) in _saved) Environment.SetEnvironmentVariable(k, v);
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); } catch { }
    }

    private string WriteGguf(string repo, string file)
    {
        string path = HubCache.SnapshotFilePath(repo, "c0ffee", file, Path.Combine(_root, "hub"));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, "GGUF"u8.ToArray().Concat(new byte[96]).ToArray());
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

    private ServerState NewState(int maxResident = 1)
    {
        var state = new ServerState
        {
            Options = new ServerOptions { Model = "", ModelId = "none", Device = "cpu", UsePaged = false, PromptCacheEnabled = false },
            Residency = new ModelResidencyManager { MaxResidentModels = maxResident },
        };
        state.ModelLoader = (path, options) =>
        {
            Interlocked.Increment(ref _loads);
            return new ServerState { Options = options, Model = new StubModel(), LoadedModelPath = path };
        };
        _states.Add(state);
        return state;
    }

    /// <summary>The acceptance case: served by its short id, two requests by that short id = one load.</summary>
    [Fact]
    public async Task ShortIdRequestedTwice_LoadsOnce()
    {
        var state = NewState();
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        Assert.Equal(1, _loads);
        Assert.NotEqual(Stem, state.Options.ModelId);   // stored under the long id: the condition the old compare failed on
        var model = state.Model;

        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        Assert.Equal(1, _loads);
        Assert.Same(model, state.Model);
    }

    /// <summary>Every alias of the served model - short id, repo id, long id, path, case-folded - hits it with no reload.</summary>
    [Fact]
    public async Task EveryAlias_HitsTheActiveModel_WithNoReload()
    {
        var state = NewState();
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        var model = state.Model;
        string longId = state.Options.ModelId;

        foreach (string alias in new[] { Stem, Stem.ToUpperInvariant(), longId, _path, _path.Replace('\\', '/'), Repo, Stem + ".gguf" })
        {
            Assert.Null(await state.EnsureActiveAsync(alias, null, CancellationToken.None));
            Assert.True(state.IsActiveModel(alias), alias);
        }
        Assert.Equal(1, _loads);
        Assert.Same(model, state.Model);
    }

    /// <summary>No regression: an unknown id still fails as before (not found), loads nothing and keeps the active model.</summary>
    [Fact]
    public async Task UnknownId_IsNotFound_AndKeepsTheActiveModel()
    {
        var state = NewState();
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        var model = state.Model;

        string? error = await state.EnsureActiveAsync("no-such-model", null, CancellationToken.None);
        Assert.NotNull(error);
        Assert.Contains("Model not found", error);
        Assert.False(state.IsActiveModel("no-such-model"));
        Assert.Equal(1, _loads);
        Assert.Same(model, state.Model);
        Assert.True(state.IsReady);
    }

    /// <summary>No regression: an empty or missing model means "the active one" - no load.</summary>
    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("  ")]
    public async Task EmptyModel_UsesTheActiveModel(string? requested)
    {
        var state = NewState();
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        Assert.Null(await state.EnsureActiveAsync(requested, null, CancellationToken.None));
        Assert.Equal(1, _loads);
        Assert.False(state.IsActiveModel(requested));   // the alias test itself never treats blank as a name
    }

    /// <summary>A different model is still a swap: alias matching must not widen to "anything resolvable".</summary>
    [Fact]
    public async Task DifferentModel_StillSwaps()
    {
        var state = NewState();
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        Assert.Null(await state.EnsureActiveAsync(OtherStem, null, CancellationToken.None));
        Assert.Equal(2, _loads);
        Assert.Equal(_otherPath, state.LoadedModelPath);
        Assert.False(state.IsActiveModel(Stem));
    }

    /// <summary>A profile over the same file is a model of its own (its own device/options): it is not an alias of the plain model.</summary>
    [Fact]
    public async Task ProfileOverTheSameFile_IsNotAnAlias()
    {
        ModelProfileStore.Save("tinyprof", new ModelProfile { From = Stem });
        var state = NewState();
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        Assert.False(state.IsActiveModel("tinyprof"));
        Assert.Null(await state.EnsureActiveAsync("tinyprof", null, CancellationToken.None));
        Assert.Equal(2, _loads);
        Assert.Equal("tinyprof", state.Options.ModelId);
    }

    /// <summary>The residency lookup matches aliases too: a stashed model requested by its short id is reactivated, not reloaded.</summary>
    [Fact]
    public async Task StashedModel_RequestedByShortId_IsReactivatedWithoutReload()
    {
        var state = NewState(maxResident: 2);
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        var first = state.Model;
        Assert.Null(await state.EnsureActiveAsync(OtherStem, null, CancellationToken.None));
        Assert.Equal(2, _loads);
        Assert.Equal(1, state.Residency.StashedCount);

        Assert.Null(await state.EnsureActiveAsync(Repo, null, CancellationToken.None));
        Assert.Equal(2, _loads);
        Assert.Same(first, state.Model);
    }

    /// <summary>An idled-out model requested by its short id reloads from the same file with the options it was served with.</summary>
    [Fact]
    public async Task IdledOutModel_RequestedByShortId_ReloadsTheSameFileAndOptions()
    {
        var state = NewState();
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        var served = state.Options;
        Assert.Single(await state.UnloadAsync(Stem, all: false, CancellationToken.None));   // unload by alias, too
        Assert.Null(state.Model);

        ServerOptions? seen = null;
        var inner = state.ModelLoader;
        state.ModelLoader = (p, o) => { seen = o; return inner(p, o); };
        Assert.Null(await state.EnsureActiveAsync(Stem, null, CancellationToken.None));
        Assert.Equal(2, _loads);
        Assert.Same(served, seen);
        Assert.Equal(_path, state.LoadedModelPath);
    }
}
