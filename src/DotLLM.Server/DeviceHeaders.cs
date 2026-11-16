using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace DotLLM.Server;

/// <summary>
/// <c>X-DotLLM-Device</c> and <c>X-DotLLM-Device-Fallback</c> response headers (#1822): which device served the request, and whether
/// <c>--device auto</c> fell back to the CPU, so a client can tell "slow because it is on the CPU" without reading the server log.
/// </summary>
/// <remarks>
/// Set when the response starts, from the model active at that moment. A streaming response commits its headers at its first byte,
/// which for a request that waits for a load can be an SSE keep-alive comment sent before the load finishes: those headers then name
/// the device of the model that was active before the load (or are absent when none was). <c>GET /v1/status</c> and
/// <c>GET /v1/models</c> always report the current device.
/// </remarks>
public static class DeviceHeaders
{
    /// <summary>The device the active model runs on (<c>cpu</c>, <c>vulkan</c>, <c>gpu:0</c>).</summary>
    public const string DeviceHeader = "X-DotLLM-Device";

    /// <summary>Present (<c>true</c>) only when <c>--device auto</c> fell back to the CPU after a faster device failed to load the model.</summary>
    public const string FallbackHeader = "X-DotLLM-Device-Fallback";

    /// <summary>Adds the headers to every <c>/v1/*</c> and <c>/api/*</c> response while a model is ready.</summary>
    /// <param name="app">The application.</param>
    /// <param name="state">The server state.</param>
    public static void Use(IApplicationBuilder app, ServerState state) =>
        app.Use(async (HttpContext context, Func<Task> next) =>
        {
            var path = context.Request.Path;
            if (path.StartsWithSegments("/v1") || path.StartsWithSegments("/api"))
            {
                context.Response.OnStarting(static s =>
                {
                    var (ctx, st) = ((HttpContext, ServerState))s;
                    Apply(ctx.Response.Headers, st);
                    return Task.CompletedTask;
                }, (context, state));
            }
            await next();
        });

    /// <summary>Writes the headers for the current active model into <paramref name="headers"/> (nothing when no model is ready).</summary>
    /// <param name="headers">The response headers.</param>
    /// <param name="state">The server state.</param>
    internal static void Apply(IHeaderDictionary headers, ServerState state)
    {
        if (!state.IsReady || state.Model is null) return;
        headers[DeviceHeader] = state.ActiveDevice;
        if (state.DeviceFallbackWarning is not null)
            headers[FallbackHeader] = "true";
    }
}
