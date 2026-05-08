using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Tokenizers.Bpe;

namespace DotLLM.Models;

/// <summary>
/// A unified factory for loading tokenizers from any supported model container format.
/// </summary>
public static class TokenizerFactory
{
    /// <summary>
    /// Loads a <see cref="BpeTokenizer"/> based on the provided model container.
    /// </summary>
    /// <param name="container">The loaded model container.</param>
    /// <param name="modelPath">The path to the model directory or file.</param>
    /// <returns>A configured <see cref="BpeTokenizer"/>.</returns>
    public static BpeTokenizer Load(IModelContainer container, string modelPath)
    {
        if (container is GgufModelContainer gguf)
        {
            return GgufBpeTokenizerFactory.Load(gguf.Metadata);
        }
        
        if (container is SafeTensorsModelContainer)
        {
            string directory = File.Exists(modelPath) 
                ? Path.GetDirectoryName(modelPath) ?? string.Empty 
                : modelPath;
                
            string tokenizerJsonPath = Path.Combine(directory, "tokenizer.json");
            return SafeTensorsTokenizerFactory.Load(tokenizerJsonPath);
        }

        throw new NotSupportedException($"Tokenizer loading is not supported for container type {container.GetType().Name}.");
    }
}
