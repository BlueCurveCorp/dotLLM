using System.Text.Json;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Creates a <see cref="BpeTokenizer"/> from a Hugging Face tokenizer.json file.
/// </summary>
public static class SafeTensorsTokenizerFactory
{
    /// <summary>
    /// Loads a <see cref="BpeTokenizer"/> from the specified tokenizer.json path.
    /// </summary>
    public static BpeTokenizer Load(string tokenizerJsonPath)
    {
        if (!File.Exists(tokenizerJsonPath))
            throw new FileNotFoundException($"Tokenizer file not found: {tokenizerJsonPath}");

        using var stream = File.OpenRead(tokenizerJsonPath);
        using var doc = JsonDocument.Parse(stream);
        var root = doc.RootElement;

        var model = root.GetProperty("model");
        var vocabElement = model.GetProperty("vocab");
        
        var tokensDict = new Dictionary<string, int>(StringComparer.Ordinal);
        int maxId = -1;
        foreach (var prop in vocabElement.EnumerateObject())
        {
            int id = prop.Value.GetInt32();
            tokensDict[prop.Name] = id;
            if (id > maxId) maxId = id;
        }

        if (root.TryGetProperty("added_tokens", out var addedTokensElement))
        {
            foreach (var tokenElement in addedTokensElement.EnumerateArray())
            {
                int id = tokenElement.GetProperty("id").GetInt32();
                string content = tokenElement.GetProperty("content").GetString()!;
                tokensDict[content] = id;
                if (id > maxId) maxId = id;
            }
        }

        string[] tokens = new string[maxId + 1];
        int[] tokenTypes = new int[maxId + 1];
        Array.Fill(tokenTypes, 1); // 1 = normal

        foreach (var kvp in tokensDict)
        {
            tokens[kvp.Value] = kvp.Key;
        }

        if (root.TryGetProperty("added_tokens", out var addedTokensElement2))
        {
            foreach (var tokenElement in addedTokensElement2.EnumerateArray())
            {
                int id = tokenElement.GetProperty("id").GetInt32();
                bool isSpecial = tokenElement.TryGetProperty("special", out var specProp) && specProp.GetBoolean();
                if (isSpecial)
                {
                    tokenTypes[id] = 3; // 3 = control token
                }
                else
                {
                    tokenTypes[id] = 4; // 4 = user-defined
                }
            }
        }

        var mergesElement = model.GetProperty("merges");
        var mergesList = new List<string>();
        foreach (var merge in mergesElement.EnumerateArray())
        {
            mergesList.Add(merge.GetString()!);
        }
        string[] merges = [.. mergesList];

        int bosId = 1;
        int eosId = 2;

        // Try to read tokenizer_config.json
        string directory = Path.GetDirectoryName(tokenizerJsonPath) ?? string.Empty;
        string configPath = Path.Combine(directory, "tokenizer_config.json");
        string? chatTemplate = null;
        if (File.Exists(configPath))
        {
            using var configStream = File.OpenRead(configPath);
            using var configDoc = JsonDocument.Parse(configStream);
            var configRoot = configDoc.RootElement;
            
            if (configRoot.TryGetProperty("bos_token", out var bosToken))
            {
                string bosStr = bosToken.ValueKind == JsonValueKind.Object ? bosToken.GetProperty("content").GetString()! : bosToken.GetString()!;
                if (tokensDict.TryGetValue(bosStr, out int bId)) bosId = bId;
            }
            if (configRoot.TryGetProperty("eos_token", out var eosToken))
            {
                string eosStr = eosToken.ValueKind == JsonValueKind.Object ? eosToken.GetProperty("content").GetString()! : eosToken.GetString()!;
                if (tokensDict.TryGetValue(eosStr, out int eId)) eosId = eId;
            }
            if (configRoot.TryGetProperty("chat_template", out var chatTemplateProp) && chatTemplateProp.ValueKind == JsonValueKind.String)
            {
                chatTemplate = chatTemplateProp.GetString();
            }
        }

        var tokenizer = BpeTokenizer.CreateTiktoken(tokens, merges, tokenTypes, bosId, eosId, null);
        
        // Expose chat template via a property or just return it? 
        // BpeTokenizer does not have a ChatTemplate property. 
        // But we can store it in the container or we can create a unified ChatTemplateFactory.
        
        return tokenizer;
    }
}
