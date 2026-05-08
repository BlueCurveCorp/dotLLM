using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers.Bpe;
using Xunit;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// End-to-end text generation tests for Qwen2 architecture via <see cref="TextGenerator"/>.
/// Validates that the full pipeline works through the <see cref="TransformerModel"/> interface.
/// </summary>
[Collection("QwenModel")]
public class QwenTextGeneratorTests
{
    private readonly QwenModelFixture _fixture;

    public QwenTextGeneratorTests(QwenModelFixture fixture)
    {
        _fixture = fixture;
    }

    private (TransformerModel model, GgufModelContainer container, BpeTokenizer tokenizer) LoadModel()
    {
        var container = GgufModelContainer.Open(_fixture.FilePath);
        var model = TransformerModel.Load(container);
        var tokenizer = GgufBpeTokenizerFactory.Load(container.Metadata);
        return (model, container, tokenizer);
    }

    [Fact]
    public void GreedyGeneration_ProducesNonEmptyOutput()
    {
        var (model, container, tokenizer) = LoadModel();
        using var _ = container;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var response = generator.Generate("Hello", options);

        Assert.False(string.IsNullOrEmpty(response.Text), "Generated text should not be empty.");
        Assert.True(response.GeneratedTokenCount > 0);
    }

    [Fact]
    public void GreedyGeneration_PredictsParis()
    {
        var (model, container, tokenizer) = LoadModel();
        using var _ = container;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        // Instruct models may emit special/whitespace tokens before the answer,
        // so generate several tokens and check the full text.
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        var response = generator.Generate("The capital of France is", options);

        Assert.True(response.GeneratedTokenIds.Length > 0);
        Assert.Contains("Paris", response.Text);
    }

    [Fact]
    public void Timings_ArePopulated()
    {
        var (model, container, tokenizer) = LoadModel();
        using var _ = container;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 5 };

        var response = generator.Generate("The capital of France is", options);
        var timings = response.Timings;

        Assert.True(timings.PrefillTimeMs > 0);
        Assert.True(timings.PrefillTokensPerSec > 0);

        if (response.GeneratedTokenCount > 1)
        {
            Assert.True(timings.DecodeTimeMs > 0);
            Assert.True(timings.DecodeTokensPerSec > 0);
        }
    }

    [Fact]
    public async Task StreamingOutput_MatchesSynchronousOutput()
    {
        var (model, container, tokenizer) = LoadModel();
        using var _ = container;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = 10 };

        // Synchronous generation
        var syncResponse = generator.Generate("The capital of France is", options);

        // Streaming generation — concatenate all text
        var sb = new StringBuilder();
        await foreach (var token in generator.GenerateStreamingTokensAsync("The capital of France is", options))
            sb.Append(token.Text);

        Assert.Equal(syncResponse.Text, sb.ToString());
    }
}
