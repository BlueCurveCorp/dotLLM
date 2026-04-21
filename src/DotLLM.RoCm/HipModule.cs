using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// Loads an HSACO code object file into a HIP module and caches kernel function handles.
/// HSACO is pre-compiled native ISA — load latency is effectively zero.
/// </summary>
public sealed class HipModule : IDisposable
{
    private nint _module;
    private readonly Dictionary<string, nint> _functions = [];

    /// <summary>
    /// Loads an HSACO module from a file path.
    /// </summary>
    /// <param name="hsacoPath">Path to the .hsaco file.</param>
    public static HipModule LoadFromFile(string hsacoPath)
    {
        byte[] image = File.ReadAllBytes(hsacoPath);
        return LoadFromBytes(image);
    }

    /// <summary>
    /// Loads an HSACO module from a byte array.
    /// </summary>
    /// <param name="image">HSACO binary bytes.</param>
    public static HipModule LoadFromBytes(byte[] image)
    {
        var module = new HipModule();
        unsafe
        {
            fixed (byte* ptr = image)
            {
                HipApi.hipModuleLoadData(out module._module, (nint)ptr)
                    .ThrowOnError();
            }
        }
        return module;
    }

    /// <summary>
    /// Gets a kernel function handle by name. Caches the result for subsequent calls.
    /// </summary>
    /// <param name="name">The <c>extern "C"</c> kernel function name.</param>
    public nint GetFunction(string name)
    {
        if (!_functions.TryGetValue(name, out nint func))
        {
            HipApi.hipModuleGetFunction(out func, _module, name)
                .ThrowOnError();
            _functions[name] = func;
        }
        return func;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        nint module = Interlocked.Exchange(ref _module, nint.Zero);

        if (module != nint.Zero)
        {
            HipApi.hipModuleUnload(module);
            _functions.Clear();
        }
    }
}
