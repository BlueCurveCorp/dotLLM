using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Extracts a <see cref="ModelConfig"/> from a Hugging Face config.json file.
/// </summary>
public static class SafeTensorsModelConfigExtractor
{
    /// <summary>
    /// Extracts a <see cref="ModelConfig"/> from the parsed JSON document.
    /// </summary>
    /// <param name="configDoc">The parsed config.json document.</param>
    /// <returns>A populated model configuration.</returns>
    public static ModelConfig Extract(JsonDocument configDoc)
    {
        var root = configDoc.RootElement;

        string archString = GetArchitecture(root);
        Architecture architecture = ParseArchitecture(archString);

        int hiddenSize = root.TryGetProperty("hidden_size", out var hs) ? hs.GetInt32() : 0;
        int numLayers = root.TryGetProperty("num_hidden_layers", out var nl) ? nl.GetInt32() : 0;
        int intermediateSize = root.TryGetProperty("intermediate_size", out var @is) ? @is.GetInt32() : 0;
        int numAttentionHeads = root.TryGetProperty("num_attention_heads", out var nah) ? nah.GetInt32() : 0;
        int numKvHeads = root.TryGetProperty("num_key_value_heads", out var nkvh) ? nkvh.GetInt32() : numAttentionHeads;

        int vocabSize = root.TryGetProperty("vocab_size", out var vs) ? vs.GetInt32() : 0;
        int maxSeqLen = root.TryGetProperty("max_position_embeddings", out var mpe) ? mpe.GetInt32() : 2048;
        float normEps = root.TryGetProperty("rms_norm_eps", out var rne) ? rne.GetSingle() : 1e-5f;
        bool tiedEmbeddings = root.TryGetProperty("tie_word_embeddings", out var twe) && twe.GetBoolean();
        int? slidingWindowSize = root.TryGetProperty("sliding_window", out var sw) ? sw.GetInt32() : null;

        // Head dimension defaults to hidden_size / num_attention_heads
        int headDim = root.TryGetProperty("head_dim", out var hd) ? hd.GetInt32() : 
                      (numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0);

        if (vocabSize == 0 || hiddenSize == 0 || numLayers == 0 || numAttentionHeads == 0)
        {
            throw new InvalidDataException("Missing required configuration fields in config.json.");
        }

        float ropeTheta = root.TryGetProperty("rope_theta", out var rt) ? rt.GetSingle() : 10000.0f;
        var ropeConfig = new DotLLM.Core.PositionEncoding.RoPEConfig
        {
            Type = DotLLM.Core.Configuration.RoPEType.NeoX,
            DimensionCount = headDim,
            Theta = ropeTheta
        };

        return new ModelConfig
        {
            Architecture = architecture,
            VocabSize = vocabSize,
            HiddenSize = hiddenSize,
            IntermediateSize = intermediateSize,
            NumLayers = numLayers,
            NumAttentionHeads = numAttentionHeads,
            NumKvHeads = numKvHeads,
            HeadDim = headDim,
            MaxSequenceLength = maxSeqLen,
            NormEpsilon = normEps,
            RoPEConfig = ropeConfig,
            PositionEncodingType = DotLLM.Core.Configuration.PositionEncodingType.RoPE,
            SlidingWindowSize = slidingWindowSize,
            TiedEmbeddings = tiedEmbeddings
        };
    }

    private static string GetArchitecture(JsonElement root)
    {
        if (root.TryGetProperty("architectures", out var archs) && archs.ValueKind == JsonValueKind.Array && archs.GetArrayLength() > 0)
        {
            return archs[0].GetString() ?? string.Empty;
        }
        return string.Empty;
    }

    private static Architecture ParseArchitecture(string archString)
    {
        return archString switch
        {
            "LlamaForCausalLM" => Architecture.Llama,
            "MistralForCausalLM" => Architecture.Mistral,
            "PhiForCausalLM" or "Phi3ForCausalLM" => Architecture.Phi,
            "Qwen2ForCausalLM" => Architecture.Qwen,
            "DeepseekV2ForCausalLM" => Architecture.DeepSeek,
            _ => throw new InvalidDataException($"Unsupported SafeTensors architecture: '{archString}'.")
        };
    }
}
