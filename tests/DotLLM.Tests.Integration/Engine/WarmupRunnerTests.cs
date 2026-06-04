using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Integration tests for <see cref="WarmupRunner"/> against SmolLM-135M Q8_0.
/// </summary>
[Collection("SmallModel")]
public sealed class WarmupRunnerTests
{
    private readonly SmallModelFixture _fixture;

    public WarmupRunnerTests(SmallModelFixture fixture) => _fixture = fixture;

    private (TransformerModel model, GgufModelContainer container, DotLLM.Tokenizers.Bpe.BpeTokenizer tokenizer) LoadModel()
    {
        var container = GgufModelContainer.Open(_fixture.FilePath);
        var model = TransformerModel.Load(container);
        var tokenizer = GgufBpeTokenizerFactory.Load(container.Metadata);
        return (model, container, tokenizer);
    }

    [Fact]
    public void Run_CompletesSuccessfully()
    {
        var (model, container, tokenizer) = LoadModel();
        using var _ = container;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new WarmupOptions { Iterations = 1, MaxTokens = 4 };

        WarmupRunner.Run(generator, tokenizer, options);
    }

    [Fact]
    public void Run_WithDisabledOptions_SkipsWarmup()
    {
        var (model, container, tokenizer) = LoadModel();
        using var _ = container;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);

        // Should return immediately without running any inference
        WarmupRunner.Run(generator, tokenizer, WarmupOptions.Disabled);
    }
}
