using DotLLM.HuggingFace;

namespace DotLLM.Server;

/// <summary>
/// Whether a requested model string names a loaded model (#1821): its id (<see cref="ServerOptions.ModelId"/>, what
/// <c>/v1/models</c> lists), the argument it was served or loaded as (<see cref="ServerOptions.Model"/> - typically a short id or
/// repo id), its file path, or any other reference that the server's resolver maps to the same file under the same id.
/// </summary>
/// <remarks>
/// Before #1821 the active model was matched by a plain string compare against its resolved long id, so a client using the short id the
/// server was started with never matched: every request evicted and reloaded the same model, discarding its prompt cache and re-running
/// <c>--device auto</c>.
/// </remarks>
internal static class ModelAliases
{
    private static readonly StringComparison PathComparison =
        OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    /// <summary>True when <paramref name="requested"/> names the model loaded as <paramref name="modelId"/>.</summary>
    /// <param name="requested">The model string a request carried. Null/blank never matches here (callers treat it as "the active one").</param>
    /// <param name="modelId">The loaded model's id.</param>
    /// <param name="servedAs">The argument it was served/loaded as (<see cref="ServerOptions.Model"/>), when known.</param>
    /// <param name="loadedPath">The GGUF it was loaded from, when known.</param>
    /// <param name="resolve">The server's model resolver (profiles, issued ids, local stores); may throw on an ambiguous name.</param>
    /// <returns>Whether the request names this model.</returns>
    public static bool Names(string? requested, string modelId, string? servedAs, string? loadedPath, Func<string, string?> resolve)
    {
        if (string.IsNullOrWhiteSpace(requested) || string.IsNullOrEmpty(modelId) || modelId == "none") return false;
        if (string.Equals(requested, modelId, StringComparison.OrdinalIgnoreCase)) return true;

        // A profile is a model of its own (its own device, layers, keep-alive), even over the same file: it names this model only when it
        // IS this model's id, which the compare above already covered.
        if (ModelProfileStore.NormalizeName(requested) is { } n && ModelProfileStore.TryGet(n) is not null) return false;

        if (!string.IsNullOrEmpty(servedAs) && servedAs != "none" && string.Equals(requested, servedAs, StringComparison.OrdinalIgnoreCase))
            return true;
        if (string.IsNullOrEmpty(loadedPath)) return false;

        // Everything else - a file path, a repo id, a bare stem, an issued id - goes through the resolver: a match must resolve to this
        // file AND produce this id (so a profile or ollama name over the same file is not this model).
        string? path;
        try { path = resolve(requested); }
        catch (Exception) { return false; }   // e.g. an ambiguous bare name: not a match; activation reports the error as before
        return path is not null && SamePath(path, loadedPath)
            && string.Equals(ServerStartup.ModelIdFor(requested, path), modelId, StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Whether two paths name the same file (full-path compare; case-insensitive on Windows).</summary>
    private static bool SamePath(string a, string b)
    {
        if (string.Equals(a, b, PathComparison)) return true;
        try { return string.Equals(Path.GetFullPath(a), Path.GetFullPath(b), PathComparison); }
        catch (Exception) { return false; }   // not a valid path
    }
}
