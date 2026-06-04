using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.ToolCallParsers;
using System.Text.Json;

namespace DotLLM.Models;

/// <summary>
/// A unified factory for creating chat templates from any supported model container format.
/// Abstracts over GGUF metadata and SafeTensors <c>tokenizer_config.json</c>.
/// </summary>
public static class ChatTemplateFactory
{
    /// <summary>
    /// Attempts to create a <see cref="IChatTemplate"/> from the given model container.
    /// For GGUF containers this reads the embedded Jinja template from GGUF metadata.
    /// For SafeTensors containers this parses <c>tokenizer_config.json</c> in the model directory.
    /// Returns <c>null</c> if no template is available.
    /// </summary>
    /// <param name="container">The loaded model container.</param>
    /// <param name="tokenizer">The tokenizer for the model.</param>
    /// <param name="modelPath">Optional model path (required for SafeTensors).</param>
    public static IChatTemplate? TryCreate(IModelContainer container, BpeTokenizer tokenizer, string? modelPath = null)
    {
        if (container is GgufModelContainer gguf)
        {
            return GgufChatTemplateFactory.TryCreate(gguf.Metadata, tokenizer);
        }

        if (container is SafeTensorsModelContainer)
        {
            if (modelPath is not null)
            {
                string directory = File.Exists(modelPath) 
                    ? Path.GetDirectoryName(modelPath) ?? string.Empty 
                    : modelPath;
                    
                string configPath = Path.Combine(directory, "tokenizer_config.json");
                if (File.Exists(configPath))
                {
                    using var configStream = File.OpenRead(configPath);
                    using var configDoc = JsonDocument.Parse(configStream);
                    var configRoot = configDoc.RootElement;
                    if (configRoot.TryGetProperty("chat_template", out var chatTemplateProp) && chatTemplateProp.ValueKind == JsonValueKind.String)
                    {
                        string templateStr = chatTemplateProp.GetString()!;
                        string bosTokenStr = tokenizer.DecodeToken(tokenizer.BosTokenId);
                        string eosTokenStr = tokenizer.DecodeToken(tokenizer.EosTokenId);
                        return new JinjaChatTemplate(templateStr, bosTokenStr, eosTokenStr);
                    }
                }
            }
        }

        return null;
    }

    /// <summary>
    /// Creates a tool-call parser appropriate for the given model container's architecture.
    /// Returns <c>null</c> if no parser can be inferred.
    /// </summary>
    /// <param name="container">The loaded model container.</param>
    public static IToolCallParser? CreateToolCallParser(IModelContainer container)
    {
        if (container is GgufModelContainer gguf)
        {
            return GgufChatTemplateFactory.CreateToolCallParser(gguf.Metadata, container.Config.Architecture);
        }

        if (container is SafeTensorsModelContainer)
        {
            // For safetensors, tool parsing would depend on the model architecture
            return GgufChatTemplateFactory.CreateToolCallParser(container.Config);
        }

        return null;
    }
}
