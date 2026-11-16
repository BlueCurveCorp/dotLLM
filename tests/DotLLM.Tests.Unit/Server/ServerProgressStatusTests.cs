using System.Globalization;
using System.Net.Http.Json;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Tokenizers;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// #1822 over real HTTP: <c>GET /v1/status</c> shows the in-flight request's prefill progress WHILE its prefill forward is still
/// running, then its decode count; <c>/v1/status</c>, <c>/v1/models</c> and the <c>X-DotLLM-Device</c> headers expose the device and a
/// <c>--device auto</c> CPU fallback. The stub model's prefill forward reports half its layers done through <see cref="ForwardProgress"/>
/// (what a CPU layer loop does) and then blocks until the test has read the status, so the read is provably mid-prefill.
/// </summary>
public sealed class ServerProgressStatusTests : IAsyncLifetime
{
    private const int VocabSize = 16;
    private const int PromptLen = 12;
    private const string Fallback = "vulkan: out of device memory; running on the CPU is ~10x slower";

    private readonly BlockingModel _model = new();
    private WebApplication? _app;
    private ServerState? _state;
    private HttpClient _http = new();

    public async Task InitializeAsync()
    {
        _state = new ServerState
        {
            Options = new ServerOptions { Model = "stub", ModelId = "stub", Device = "auto", ResolvedDevice = "cpu", DeviceFallbackWarning = Fallback },
            Model = _model,
            Config = _model.Config,
            Tokenizer = new StubTokenizer(),
            Generator = new TextGenerator(_model, new StubTokenizer()),
            IsReady = true,
        };
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddDotLLM(_state);
        _app = builder.Build();
        _app.UseRequestLifecycleLog();   // as ServerStartup: gives each POST its req#N tag
        DeviceHeaders.Use(_app, _state);
        _app.MapDotLLMEndpoints(serveUi: false);
        await _app.StartAsync();
        _http = new HttpClient { BaseAddress = new Uri(_app.Urls.First()) };
    }

    public async Task DisposeAsync()
    {
        _model.Release.Set();
        _http.Dispose();
        if (_app is not null) await _app.DisposeAsync();
        _state?.Dispose();
    }

    private async Task<JsonElement> StatusAsync() =>
        JsonDocument.Parse(await _http.GetStringAsync("/v1/status")).RootElement;

    [Fact]
    public async Task Status_ShowsPrefillProgressDuringTheForward_ThenTheDecodeCount_AndTheDevice()
    {
        var idle = await StatusAsync();
        Assert.Equal("cpu", idle.GetProperty("device").GetString());
        Assert.True(idle.GetProperty("device_fallback").GetBoolean());
        Assert.Equal(Fallback, idle.GetProperty("device_fallback_reason").GetString());
        Assert.Equal(JsonValueKind.Null, idle.GetProperty("requests").ValueKind);

        var post = _http.PostAsync("/v1/completions",
            new StringContent("{\"model\":\"stub\",\"prompt\":\"hi\",\"max_tokens\":3,\"temperature\":0}", Encoding.UTF8, "application/json"));

        Assert.True(_model.InPrefill.Wait(TimeSpan.FromSeconds(30)), "the prefill forward never started");
        // The forward is blocked right now, having reported 1 of 2 layers: status must already show half the prompt.
        var mid = await StatusAsync();
        var req = Assert.Single(mid.GetProperty("requests").EnumerateArray());
        Assert.Equal("prefill", req.GetProperty("phase").GetString());
        Assert.Equal(PromptLen, req.GetProperty("prefill_tokens_total").GetInt32());
        Assert.Equal(PromptLen / 2, req.GetProperty("prefill_tokens_done").GetInt32());
        Assert.Equal(0, req.GetProperty("decode_tokens").GetInt32());
        Assert.Equal(PromptLen / 2, mid.GetProperty("prefill_tokens_done").GetInt32());   // top-level mirror of the oldest request
        Assert.Equal(PromptLen, mid.GetProperty("prefill_tokens_total").GetInt32());
        Assert.StartsWith("req#", req.GetProperty("request").GetString() ?? "", StringComparison.Ordinal);

        // Let the prefill finish; the second decode forward blocks too, so the decode count is read mid-generation.
        _model.Release.Set();
        Assert.True(_model.InSecondDecode.Wait(TimeSpan.FromSeconds(30)), "decode never reached its second step");
        var decoding = Assert.Single((await StatusAsync()).GetProperty("requests").EnumerateArray());
        Assert.Equal("decode", decoding.GetProperty("phase").GetString());
        Assert.Equal(PromptLen, decoding.GetProperty("prefill_tokens_done").GetInt32());
        Assert.Equal(2, decoding.GetProperty("decode_tokens").GetInt32());
        _model.DecodeRelease.Set();

        using var response = await post;
        Assert.True(response.IsSuccessStatusCode, await response.Content.ReadAsStringAsync());
        Assert.Equal("cpu", Assert.Single(response.Headers.GetValues(DeviceHeaders.DeviceHeader)));
        Assert.Equal("true", Assert.Single(response.Headers.GetValues(DeviceHeaders.FallbackHeader)));

        var after = await StatusAsync();
        Assert.Equal(JsonValueKind.Null, after.GetProperty("requests").ValueKind);
    }

    [Fact]
    public async Task Models_CarryTheDeviceAndTheFallback()
    {
        var model = Assert.Single(JsonDocument.Parse(await _http.GetStringAsync("/v1/models")).RootElement.GetProperty("data").EnumerateArray());
        Assert.Equal("cpu", model.GetProperty("device").GetString());
        Assert.True(model.GetProperty("device_fallback").GetBoolean());
        Assert.Equal(Fallback, model.GetProperty("device_fallback_reason").GetString());
    }

    [Fact]
    public async Task NoFallback_HasNoFallbackHeader_AndFalseInStatus()
    {
        _state!.DeviceFallbackWarning = null;
        using var response = await _http.GetAsync("/v1/models");
        Assert.Equal("cpu", Assert.Single(response.Headers.GetValues(DeviceHeaders.DeviceHeader)));
        Assert.False(response.Headers.Contains(DeviceHeaders.FallbackHeader));
        var status = await StatusAsync();
        Assert.False(status.GetProperty("device_fallback").GetBoolean());
        Assert.Equal(JsonValueKind.Null, status.GetProperty("device_fallback_reason").ValueKind);
    }

    /// <summary>The production pipeline (<see cref="ServerStartup.BuildApp"/>, what <c>dotllm serve</c> runs) registers the device headers.</summary>
    [Fact]
    public async Task ProductionPipeline_EmitsTheDeviceHeaders()
    {
        await using var app = ServerStartup.BuildApp(_state!, ["--urls", "http://127.0.0.1:0"]);
        await app.StartAsync();
        using var http = new HttpClient { BaseAddress = new Uri(app.Urls.First()) };
        using var response = await http.GetAsync("/v1/models");
        Assert.Equal("cpu", Assert.Single(response.Headers.GetValues(DeviceHeaders.DeviceHeader)));
        Assert.Equal("true", Assert.Single(response.Headers.GetValues(DeviceHeaders.FallbackHeader)));
        await app.StopAsync();
    }

    // ── fakes ──

    private sealed class StubTokenizer : ITokenizer
    {
        public int VocabSize => ServerProgressStatusTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => 15;
        public int[] Encode(string text) => Enumerable.Range(2, PromptLen).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Join(",", tokenIds.ToArray());
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public string DecodeToken(int tokenId) => tokenId.ToString(CultureInfo.InvariantCulture);
        public int CountTokens(string text) => PromptLen;
    }

    /// <summary>
    /// Prefill forward: reports 1 of 2 layers done (as a CPU layer loop would), signals, and blocks until released. The second decode
    /// forward blocks the same way. Always argmaxes token 3 (never EOS).
    /// </summary>
    private sealed class BlockingModel : IModel
    {
        private int _decodeCalls;
        public ManualResetEventSlim InPrefill { get; } = new();
        public ManualResetEventSlim Release { get; } = new();
        public ManualResetEventSlim InSecondDecode { get; } = new();
        public ManualResetEventSlim DecodeRelease { get; } = new();

        public ModelConfig Config { get; } = new()
        {
            VocabSize = VocabSize, NumLayers = 2, NumAttentionHeads = 1, NumKvHeads = 1, HiddenSize = 4, IntermediateSize = 16,
            HeadDim = 4, MaxSequenceLength = 64, Architecture = DotLLM.Core.Configuration.Architecture.Llama,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) => Forward(tokenIds, positions, deviceId, null);

        public unsafe ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
        {
            if (tokenIds.Length > 1)
            {
                ForwardProgress.LayersCompleted(0, 2);
                ForwardProgress.LayersCompleted(1, 2);
                InPrefill.Set();
                Release.Wait(TimeSpan.FromSeconds(30));
            }
            else if (Interlocked.Increment(ref _decodeCalls) == 2)
            {
                InSecondDecode.Set();
                DecodeRelease.Wait(TimeSpan.FromSeconds(30));
            }

            int rows = tokenIds.Length;
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)(rows * VocabSize * sizeof(float)), 64);
            for (int r = 0; r < rows; r++)
            {
                var row = new Span<float>((float*)ptr + r * VocabSize, VocabSize);
                row.Fill(-10f);
                row[3] = 10f;
            }
            if (kvCache != null)
            {
                int kvStride = Config.NumKvHeads * Config.HeadDim;
                nuint bytes = (nuint)(rows * kvStride * sizeof(float));
                nint kPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64), vPtr = (nint)NativeMemory.AlignedAlloc(bytes, 64);
                NativeMemory.Clear((void*)kPtr, bytes);
                NativeMemory.Clear((void*)vPtr, bytes);
                kvCache.Update(new TensorRef(rows, kvStride, DType.Float32, -1, kPtr), new TensorRef(rows, kvStride, DType.Float32, -1, vPtr), positions, 0);
                NativeMemory.AlignedFree((void*)kPtr);
                NativeMemory.AlignedFree((void*)vPtr);
            }
            return new UnmanagedTensor(new TensorShape(rows, VocabSize), DType.Float32, deviceId, ptr);
        }

        public void Dispose() { }
    }
}
