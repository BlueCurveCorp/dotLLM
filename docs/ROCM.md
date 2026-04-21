# ROCm Backend Architecture — dotLLM

## AMD GPU Acceleration from .NET: Alternatives Research

Before committing to a HIP code-object approach (analogous to `nvcc -ptx` → PTX on CUDA), every viable path to AMD GPU compute from C#/.NET was evaluated. Goal: avoid a C/C++ shared library while matching the performance of a native HIP implementation.

### Evaluated Approaches

**ILGPU** (v1.5.3) targets CUDA and CPU; its OpenCL backend is experimental and CPU-bound in practice, and it has no ROCm/HIP backend. Ruled out — no AMD GPU acceleration.

**ManagedCuda** is NVIDIA-only by definition. Ruled out.

**ComputeSharp** (v3.2.0) transpiles C# to HLSL/DirectX 12. Windows-only (DX12 hard dependency). No HIP/ROCm, no FP16 tensor cores. Ruled out.

**Silk.NET** (v2.23.0) exposes Vulkan and OpenCL bindings. Vulkan compute is viable — `VK_AMD_shader_core_properties` and `VK_NV_cooperative_matrix2`-equivalent work is ongoing for AMD. Vulkan path would require GLSL/SPIR-V shaders, not HIP C++, and forfeits rocBLAS entirely. Retains cross-vendor reach but has the same "build your own framework" cost as CUDA's Silk.NET path.

**OpenCL** (via Silk.NET): AMD ships `libOpenCL.so` (via ROCm `opencl-compat`). Full AMD GPU access but: no Matrix Core access from custom kernels, no rocBLAS, inferior ecosystem for LLM workloads. Ruled out on performance grounds.

**HIP Runtime API via P/Invoke** (chosen): AMD's HIP Runtime API is a thin superset of the CUDA Runtime API — 1:1 name-mapping (`cu*` → `hip*`, `__global__` stays). `libamdhip64.so` (Linux) ships with every ROCm driver installation and exposes the full HIP Module API: load code objects, get function handles, launch kernels. rocBLAS provides FP16 GEMM with Matrix Core acceleration via `rocblas_hgemm`. The pattern is structurally identical to the CUDA PTX approach in `docs/CUDA.md` — 100% C# orchestration, no dotLLM-authored `.so`.

**AMD provides no official .NET SDK.** ROCm is a pure Linux-first stack; Windows support via HIP SDK is partial (no rocBLAS on Windows as of ROCm 6.x).

### Conclusion

No pure-C# approach matches native HIP for LLM inference on AMD GPUs. The HIP Module API (`hipModuleLoadData` / `hipModuleLaunchKernel`) replaces the CUDA Driver API (`cuModuleLoadData` / `cuLaunchKernel`) with near-identical semantics. We P/Invoke `libamdhip64.so` and `librocblas.so` directly — no dotLLM-authored shared library, no CMake build system for the runtime path.

### Capability Comparison

| Capability | HIP Module API (chosen) | Vulkan via Silk.NET | OpenCL via Silk.NET |
|---|---|---|---|
| **rocBLAS GEMM** | ✅ Direct calls (Linux + Windows) | ❌ No (manual shaders) | ❌ No |
| **Custom GPU kernels** | ✅ HIP C++ → code object | ✅ GLSL → SPIR-V | ✅ OpenCL C |
| **Matrix Cores (custom)** | ✅ Full access via rocWMMA | ⚠️ Partial (cooperative_matrix ext) | ❌ No |
| **FP16 / BF16** | ✅ Full (CDNA/RDNA3+) | ✅ FP16 (BF16 varies) | ⚠️ FP16 extension-dependent |
| **Flash Attention** | ✅ Native quality (rocWMMA) | ✅ Proven in research | ⚠️ Difficult |
| **Memory management** | ✅ Full control | ✅ Full control | ✅ Full control |
| **Linux + Windows support** | ✅ Full (AMD HIP SDK) | ✅ | ✅ |
| **No C/C++ build system** | ✅ (hipcc only) | ✅ (glslc only) | ✅ |
| **License risk** | None (own code) | MIT (Silk.NET) | None |
| **Multi-vendor GPU** | ❌ AMD only | ✅ Cross-vendor | ✅ Cross-vendor |
| **Perf vs native HIP** | ~98–100% | ~70–90% | ~50–80% |

---

## Chosen Architecture: Code Object Loading via HIP Module API

dotLLM's ROCm backend P/Invokes AMD's **HIP Runtime API** (`libamdhip64.so`) and **rocBLAS** (`librocblas.so`) directly. HIP kernels are written in `.hip` files (a superset of CUDA `.cu`), compiled to **HSACO** (HIP Shared Application Code Object) with a single `hipcc --genco` command — no CMake, no shared library project. HSACO files ship alongside the .NET assemblies as content files, loaded at runtime via `hipModuleLoadData`.

### How It Works

```
┌─────────────────┐   hipcc --genco    ┌────────────────────┐
│  rmsnorm.hip    │ ─────────────────► │ rmsnorm.hsaco      │  (ELF binary, ships with app)
│  rope.hip       │   (one command,    │ rope.hsaco         │
│  attention.hip  │    no build sys)   │ attention.hsaco    │
│  dequant.hip    │                    │ dequant.hsaco      │
└─────────────────┘                    └──────┬─────────────┘
                                              │ loaded at runtime
┌─────────────────────────────────────────────▼─────────────────────┐
│  C# application                                                    │
│                                                                    │
│  [LibraryImport("amdhip64")]   ← AMD HIP runtime (on system)     │
│  hipModuleLoadData(codeObj)    ← loads HSACO into module          │
│  hipModuleGetFunction(module)  ← gets kernel handle               │
│  hipModuleLaunchKernel(func)   ← launches on GPU                 │
│                                                                    │
│  [LibraryImport("rocblas")]    ← AMD rocBLAS (on system)         │
│  rocblas_hgemm(...)            ← Matrix Core FP16 GEMM           │
└────────────────────────────────────────────────────────────────────┘
```

### What Libraries Are Involved

**`libamdhip64.so` / `amdhip64.dll`** — the HIP Runtime library. Installed with every ROCm driver (Linux) or the AMD HIP SDK (Windows). Provides: device enumeration, context management, memory allocation, code object loading, kernel launching, stream management. Semantically equivalent to `libcuda.so`/`nvcuda.dll` — the Module API subset (`hipModule*`, `hipMem*`, `hipStream*`) is the direct counterpart.

**`librocblas.so` / `rocblas.dll`** — rocBLAS. Available on both Linux (ROCm) and Windows (AMD HIP SDK installer, shipped in `%HIP_PATH%\bin`). Provides FP16 GEMM (`rocblas_hgemm`) with automatic Matrix Core usage for CDNA/RDNA3 GPUs when matrix dimensions are multiples of 16 — the single most important operation in LLM inference. Note: on Windows, `rocblas.dll` requires the accompanying `rocblas/` Tensile kernel subdirectory to be co-located or `ROCBLAS_TENSILE_LIBPATH` set.

No dotLLM-authored `.so`/`.dll` is ever created.

### HSACO: AMD's GPU Code Object Format

HSACO (HIP Shared Application Code Object) is an ELF-wrapped binary containing AMD GPU ISA (GFX ISA). It is the AMD counterpart to NVIDIA's PTX/fatbin.

**Critical difference from PTX**: HSACO is **ISA-specific** — each file targets a concrete GPU family (e.g., `gfx1100` for RDNA 3, `gfx90a` for CDNA 2). There is no forward-compatible intermediate representation equivalent to NVIDIA's PTX. Mitigation: `hipcc` can produce a **"fat binary"** bundling HSACO for multiple GFX targets in a single file (`--offload-arch=gfx90a --offload-arch=gfx1100`), loaded via the same `hipModuleLoadData` call — the runtime selects the correct ISA slice.

Key properties:
- **ISA-specific**: must be compiled per GPU family. Fat binary pattern covers multiple targets.
- **Binary format**: ELF-based, not human-readable (unlike PTX text). Diffable via `llvm-objdump`.
- **JIT-free**: no JIT compilation on load — code is already native ISA. Load latency is effectively zero (vs 100–500ms for CUDA PTX JIT).
- **ROCm code object cache**: `~/.cache/comgr` caches compiled intermediate products for reuse.
- **LLVM-based toolchain**: `hipcc` uses Clang/LLVM with the `amdgpu` backend. Standard LLVM IR → AMDGPU ISA pipeline.

### Target GFX Architectures

| GFX Target | GPU Family | Matrix Cores | Notes |
|---|---|---|---|
| `gfx906` | CDNA 1 (MI100 series) | ✅ MFMA | Data-center only |
| `gfx90a` | CDNA 2 (MI200 / MI250) | ✅ MFMA + BF16 | Primary data-center target |
| `gfx942` | CDNA 3 (MI300X) | ✅ MFMA + FP8 | Highest priority DC target |
| `gfx1030` | RDNA 2 (RX 6000) | ❌ No matrix cores | Consumer; rocBLAS GEMM only |
| `gfx1100` | RDNA 3 (RX 7000) | ⚠️ `WMMA` (limited) | Consumer best target |
| `gfx1101/02` | RDNA 3 variants | ⚠️ `WMMA` | Same as gfx1100 path |

For the initial implementation, target `gfx1100` (RDNA 3 baseline, widest consumer reach) and `gfx90a` (CDNA 2, standard HPC target). Add `gfx942` (MI300X) as a high-priority follow-up.

### Comparison with CUDA PTX Approach

| Aspect | CUDA (PTX) | ROCm (HSACO) |
|---|---|---|
| **Compiler** | `nvcc -ptx` | `hipcc --genco` |
| **Output format** | PTX text (arch-independent) | HSACO ELF (GFX-specific) |
| **Forward compatibility** | ✅ PTX runs on SM_50+ | ❌ Per-family HSACO required |
| **Fat binary** | `nvcc -fatbin` | `hipcc --offload-arch=X --offload-arch=Y` |
| **JIT on load** | ~100–500ms per module | None (pre-compiled ISA) |
| **Module API** | `cuModuleLoadData` | `hipModuleLoadData` |
| **Kernel launch API** | `cuLaunchKernel` | `hipModuleLaunchKernel` |
| **GEMM library** | cuBLAS `cublasHgemm` | rocBLAS `rocblas_hgemm` |
| **Matrix Cores** | NVIDIA Tensor Cores | AMD Matrix Cores (MFMA / WMMA) |
| **P/Invoke target** | `libcuda.so` / `nvcuda.dll` | `libamdhip64.so` |
| **Windows support** | ✅ Full | ⚠️ HIP SDK limited (no rocBLAS) |
| **Kernel source** | `.cu` + `extern "C"` | `.hip` + `extern "C"` (identical) |

The HIP source code is **directly translatable from CUDA** — in most cases `hipify-clang` produces a working `.hip` from a `.cu` with no manual edits.

---

## P/Invoke Layer

### HIP Module API Declarations

~25 function declarations against `libamdhip64.so`. Semantically mirrors `CudaDriverApi.cs`. The `hip*` function signatures are deliberately aligned with the CUDA Driver API.

```csharp
// src/DotLLM.RoCm/Interop/HipApi.cs
using System.Runtime.InteropServices;

namespace DotLLM.RoCm.Interop;

/// <summary>
/// Minimal P/Invoke declarations against AMD's HIP Runtime API (Module API subset).
/// libamdhip64.so — installed with ROCm driver.
/// All functions return hipError_t (int): 0 = hipSuccess, non-zero = error.
/// </summary>
internal static partial class HipApi
{
    private const string LibName = "amdhip64";

    // ── Initialization ──────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipInit(uint flags);

    // ── Device ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipGetDevice(out int device);

    [LibraryImport(LibName)]
    internal static partial int hipGetDeviceCount(out int count);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceGetName(
        [MarshalAs(UnmanagedType.LPArray)] byte[] name, int len, int device);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceTotalMem(out nuint bytes, int device);

    [LibraryImport(LibName)]
    internal static partial int hipDeviceGetAttribute(
        out int value, int attribute, int device);

    // ── Context ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipCtxCreate(out nint ctx, uint flags, int device);

    [LibraryImport(LibName)]
    internal static partial int hipCtxDestroy(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int hipCtxSetCurrent(nint ctx);

    [LibraryImport(LibName)]
    internal static partial int hipCtxGetCurrent(out nint ctx);

    // ── Module (code object loading) ────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipModuleLoadData(out nint module, nint imagePtr);

    [LibraryImport(LibName)]
    internal static partial int hipModuleGetFunction(
        out nint function, nint module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [LibraryImport(LibName)]
    internal static partial int hipModuleUnload(nint module);

    // ── Kernel launch ───────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipModuleLaunchKernel(
        nint function,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, nint stream,
        nint kernelParams, nint extra);

    // ── Memory ──────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipMalloc(out nint devicePtr, nuint size);

    [LibraryImport(LibName)]
    [SuppressGCTransition] // trivially short — just hipFree
    internal static partial int hipFree(nint devicePtr);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyHtoD(nint dst, nint src, nuint size);
    // hipMemcpyKind: 1 = hipMemcpyHostToDevice

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoH(nint dst, nint src, nuint size);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoD(nint dst, nint src, nuint size);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyHtoDAsync(
        nint dst, nint src, nuint size, nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipMemcpyDtoHAsync(
        nint dst, nint src, nuint size, nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipMemset(nint dst, int value, nuint size);

    // ── Streams ─────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial int hipStreamCreate(out nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipStreamDestroy(nint stream);

    [LibraryImport(LibName)]
    internal static partial int hipStreamSynchronize(nint stream);

    // ── Error ───────────────────────────────────────────────────────

    [LibraryImport(LibName)]
    internal static partial nint hipGetErrorName(int error);

    [LibraryImport(LibName)]
    internal static partial nint hipGetErrorString(int error);
}
```

> **Note on `hipMemcpy` vs explicit direction overloads**: HIP exposes `hipMemcpy(dst, src, size, kind)` where `kind` is `hipMemcpyKind`. The declarations above use the explicit-direction overloads (`hipMemcpyHtoD`, `hipMemcpyDtoH`, `hipMemcpyDtoD`) which map directly to the CUDA Driver API equivalents and avoid the additional kind parameter — same pattern as the CUDA backend.

### rocBLAS Declarations

```csharp
// src/DotLLM.RoCm/Interop/RocBlasApi.cs
using System.Runtime.InteropServices;

namespace DotLLM.RoCm.Interop;

/// <summary>
/// Minimal rocBLAS P/Invoke. librocblas.so — installed with ROCm.
/// rocblas_status: 0 = rocblas_status_success.
/// </summary>
internal static partial class RocBlasApi
{
    private const string LibName = "rocblas";

    [LibraryImport(LibName)]
    internal static partial int rocblas_create_handle(out nint handle);

    [LibraryImport(LibName)]
    internal static partial int rocblas_destroy_handle(nint handle);

    [LibraryImport(LibName)]
    internal static partial int rocblas_set_stream(nint handle, nint stream);

    [LibraryImport(LibName)]
    internal static partial int rocblas_set_math_mode(nint handle, int mathMode);
    // rocblas_math_mode: rocblas_default_math = 0, rocblas_xf32_xdl_math_op = 1

    // FP16 GEMM — uses Matrix Cores automatically on CDNA/RDNA3 GPUs
    // Row-major trick identical to CUDA backend: swap A↔B, both ROCBLAS_OPERATION_NONE
    [LibraryImport(LibName)]
    internal static partial int rocblas_hgemm(
        nint handle,
        int transA, int transB,        // rocblas_operation: 0=none, 1=T, 2=C
        int m, int n, int k,
        in ushort alpha,               // rocblas_half passed as ushort
        nint A, int lda,
        nint B, int ldb,
        in ushort beta,
        nint C, int ldc);

    // GemmEx — mixed precision (FP16 input, FP32 accumulate)
    [LibraryImport(LibName)]
    internal static partial int rocblas_gemm_ex(
        nint handle,
        int transA, int transB,
        int m, int n, int k,
        nint alpha,
        nint A, int aType, int lda,
        nint B, int bType, int ldb,
        nint beta,
        nint C, int cType, int ldc,
        nint D, int dType, int ldd,
        int computeType,
        int algo,
        int solutionIndex,
        uint flags);
    // rocblas_datatype: rocblas_datatype_f16_r=150, rocblas_datatype_f32_r=151
    // rocblas_compute_type: rocblas_compute_f32=300
}
```

### Error Handling

```csharp
// src/DotLLM.RoCm/Interop/HipException.cs
namespace DotLLM.RoCm.Interop;

public sealed class HipException : Exception
{
    public int ErrorCode { get; }

    public HipException(int errorCode, string message)
        : base($"HIP error {errorCode}: {message}")
    {
        ErrorCode = errorCode;
    }
}

// src/DotLLM.RoCm/Interop/HipErrorHelper.cs
using System.Runtime.InteropServices;

namespace DotLLM.RoCm.Interop;

internal static class HipErrorHelper
{
    internal static void ThrowOnError(this int result)
    {
        if (result == 0) return; // hipSuccess

        string message = "Unknown HIP error";
        nint strPtr = HipApi.hipGetErrorString(result);
        if (strPtr != 0)
            message = Marshal.PtrToStringAnsi(strPtr) ?? message;

        throw new HipException(result, message);
    }
}
```

---

## HIP Kernel Conventions

HIP kernels mirror CUDA conventions with direct `__global__` syntax. `extern "C"` linkage prevents C++ name mangling, enabling `hipModuleGetFunction` lookup by name:

```cpp
// native/kernels/rmsnorm.hip
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__ void rmsnorm_f16(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    const int n,
    const float eps)
{
    // Standard warp-reduction RMS normalization
    // FP16 in/out, FP32 accumulation for numerical stability
    // One block per row, warp shuffle for reduction (__shfl_down_sync → __shfl_down)
}
```

**Key HIP-vs-CUDA source differences:**

| CUDA C++ | HIP C++ | Notes |
|---|---|---|
| `#include <cuda_fp16.h>` | `#include <hip/hip_fp16.h>` | Header path changes |
| `__shfl_down_sync(mask, v, d)` | `__shfl_down(v, d)` | HIP warp sync is implicit |
| `wmma::` namespace | `rocwmma::` namespace | Matrix Core access via rocWMMA |
| `__syncthreads()` | `__syncthreads()` | Identical |
| `threadIdx`, `blockIdx` | `threadIdx`, `blockIdx` | Identical |
| `atomicAdd(float*)` | `atomicAdd(float*)` | Identical |

`hipify-clang` automates the majority of these transformations. Verify output manually for `wmma`/`__shfl_sync` patterns.

### Compiling to HSACO

```bash
# Single GFX target:
hipcc --genco --offload-arch=gfx1100 -o rmsnorm_gfx1100.hsaco rmsnorm.hip

# Fat binary (multiple GFX targets — recommended for distribution):
hipcc --genco \
    --offload-arch=gfx906 \
    --offload-arch=gfx90a \
    --offload-arch=gfx942 \
    --offload-arch=gfx1030 \
    --offload-arch=gfx1100 \
    -o rmsnorm.hsaco rmsnorm.hip
```

The resulting `.hsaco` is an ELF file with one code section per target. `hipModuleLoadData` selects the correct ISA slice automatically based on the active device's GFX target string.

### Kernel Launch from C#

Structurally identical to the CUDA backend — kernel arguments passed as an array of pointers:

```csharp
public void LaunchRmsNorm(
    nint input, nint weight, nint output,
    int hiddenSize, float eps,
    uint rows, nint stream)
{
    unsafe
    {
        nint inputArg  = input;
        nint weightArg = weight;
        nint outputArg = output;
        int  nArg      = hiddenSize;
        float epsArg   = eps;

        void*[] args = [&inputArg, &weightArg, &outputArg, &nArg, &epsArg];

        fixed (void** argsPtr = args)
        {
            HipApi.hipModuleLaunchKernel(
                _rmsnormFunc,
                gridDimX: rows, gridDimY: 1, gridDimZ: 1,
                blockDimX: 256, blockDimY: 1, blockDimZ: 1,
                sharedMemBytes: 0,
                stream: stream,
                kernelParams: (nint)argsPtr,
                extra: 0).ThrowOnError();
        }
    }
}
```

### Module Loading

```csharp
// src/DotLLM.RoCm/HipModule.cs
public sealed class HipModule : IDisposable
{
    private nint _module;
    private readonly Dictionary<string, nint> _functions = new();

    public static HipModule LoadFromFile(string hsacoPath)
    {
        byte[] image = File.ReadAllBytes(hsacoPath);
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

    public void Dispose()
    {
        if (_module != 0)
        {
            HipApi.hipModuleUnload(_module);
            _module = 0;
        }
    }
}
```

---

## Build System

### Kernel Compilation Script

```bash
#!/bin/bash
# native/build_rocm.sh — Compile all .hip kernels to HSACO fat binaries

# Target GPU families — adjust to match your distribution targets
GFX_TARGETS=(gfx906 gfx90a gfx942 gfx1030 gfx1100)

OUT_DIR="$(dirname "$0")/hsaco"
mkdir -p "$OUT_DIR"

# Build --offload-arch flags
ARCH_FLAGS=""
for target in "${GFX_TARGETS[@]}"; do
    ARCH_FLAGS="$ARCH_FLAGS --offload-arch=$target"
done

for hip_file in "$(dirname "$0")"/kernels/*.hip; do
    base=$(basename "$hip_file" .hip)

    hipcc --genco $ARCH_FLAGS \
          -O3 \
          --fast-math \
          -o "$OUT_DIR/$base.hsaco" \
          "$hip_file"

    echo "  $base.hip → $base.hsaco"
done
```

### .NET Integration

HSACO files are included as content files, identical to the PTX pattern:

```xml
<!-- src/DotLLM.RoCm/DotLLM.RoCm.csproj -->
<ItemGroup>
    <Content Include="..\..\native\hsaco\*.hsaco" Link="hsaco\%(Filename)%(Extension)">
        <CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory>
    </Content>
</ItemGroup>
```

Or as embedded resources for single-file deployment:

```xml
<ItemGroup>
    <EmbeddedResource Include="..\..\native\hsaco\*.hsaco" Link="hsaco\%(Filename)%(Extension)" />
</ItemGroup>
```

### NativeLibrary Resolution

```csharp
// src/DotLLM.RoCm/Interop/HipLibraryResolver.cs
using System.Reflection;
using System.Runtime.InteropServices;

namespace DotLLM.RoCm.Interop;

internal static class HipLibraryResolver
{
    internal static void Register()
    {
        NativeLibrary.SetDllImportResolver(
            typeof(HipLibraryResolver).Assembly,
            ResolveHipLibrary);
    }

    private static nint ResolveHipLibrary(
        string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName == "amdhip64")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // AMD HIP SDK installs amdhip64.dll into %HIP_PATH%\bin
                // Default: C:\Program Files\AMD\ROCm\<ver>\bin
                foreach (var candidate in new[]
                {
                    "amdhip64.dll",
                    @"C:\Program Files\AMD\ROCm\6.1\bin\amdhip64.dll",
                    @"C:\Program Files\AMD\ROCm\6.0\bin\amdhip64.dll",
                })
                {
                    if (NativeLibrary.TryLoad(candidate, out nint h))
                        return h;
                }
            }
            else
            {
                // ROCm installs libamdhip64.so into /opt/rocm/lib
                foreach (var candidate in new[]
                {
                    "libamdhip64.so",
                    "/opt/rocm/lib/libamdhip64.so",
                    "/opt/rocm-6.0/lib/libamdhip64.so",
                })
                {
                    if (NativeLibrary.TryLoad(candidate, out nint h))
                        return h;
                }
            }
        }

        if (libraryName == "rocblas")
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                // rocblas.dll ships in %HIP_PATH%\bin alongside amdhip64.dll
                // Requires rocblas/ Tensile subdirectory co-located or ROCBLAS_TENSILE_LIBPATH set
                foreach (var candidate in new[]
                {
                    "rocblas.dll",
                    @"C:\Program Files\AMD\ROCm\6.1\bin\rocblas.dll",
                    @"C:\Program Files\AMD\ROCm\6.0\bin\rocblas.dll",
                })
                {
                    if (NativeLibrary.TryLoad(candidate, out nint h))
                        return h;
                }
            }
            else
            {
                foreach (var candidate in new[]
                {
                    "librocblas.so",
                    "/opt/rocm/lib/librocblas.so",
                    "/opt/rocm-6.0/lib/librocblas.so",
                })
                {
                    if (NativeLibrary.TryLoad(candidate, out nint h))
                        return h;
                }
            }
        }

        return 0; // fall through to default resolution
    }
}
```

> **Linux library path**: `/opt/rocm/lib` is not in the default `LD_LIBRARY_PATH` on all distros. Users must either add it (via `/etc/ld.so.conf.d/rocm.conf` + `ldconfig`) or the resolver's explicit path probing handles it.
>
> **Windows `ROCBLAS_TENSILE_LIBPATH`**: `rocblas.dll` depends on a `rocblas/` subdirectory of Tensile kernel files co-located with the DLL (in `%HIP_PATH%\bin\rocblas\`). If moved, set `ROCBLAS_TENSILE_LIBPATH` to the new directory. The `DotLLM.RoCm` README should document both platform requirements.

---

## No JIT Overhead: HSACO Load Latency

Unlike CUDA PTX (100–500ms JIT per module), HSACO is pre-compiled native ISA — `hipModuleLoadData` is an ELF parse + GPU upload, typically completing in **<10ms** per module. There is no equivalent to CUDA's `~/.nv/ComputeCache` because no JIT step occurs.

The ROCm code object manager (`comgr`) does maintain a cache at `~/.cache/comgr` for intermediate compilation products used during `hipcc` invocations, but this is irrelevant at runtime.

This is a strict advantage over the CUDA PTX path: **cold-start model initialization is faster on ROCm when using pre-compiled HSACO**. The tradeoff is the requirement to ship ISA-specific binaries per GFX family rather than a single architecture-agnostic PTX file.

---

## rocBLAS Row-Major Convention

rocBLAS, like cuBLAS, uses column-major layout (Fortran convention). The identical transposition trick from the CUDA backend applies:

To compute `C = A × B` (all row-major), call `rocblas_hgemm` with swapped operands:
- Pass `B` as first matrix, `A` as second matrix
- Set `transA = ROCBLAS_OPERATION_NONE`, `transB = ROCBLAS_OPERATION_NONE`
- rocBLAS computes `C_colmajor = B_colmajor × A_colmajor`
- Because `X_colmajor ≡ X^T_rowmajor`, this yields the correct row-major result

This is the same well-proven trick used by llama.cpp's ROCm backend and every HIP inference engine.

**Matrix Core activation**: `rocblas_hgemm` uses Matrix Cores automatically on CDNA (MI series) when dimensions are multiples of 16. On RDNA 3 (`gfx1100`), `WMMA` instructions are used internally by rocBLAS with multiples of 16. Enabling `rocblas_xf32_xdl_math_op` via `rocblas_set_math_mode` is analogous to `CUBLAS_TENSOR_OP_MATH`.

---

## Kernel Catalog

All kernels compiled to HSACO fat binaries, loaded via `hipModuleLoadData`, launched via `hipModuleLaunchKernel`. Function names are **identical to the CUDA backend** — shared kernel catalog simplifies `IBackend` dispatch.

**FP16 pipeline (primary):**

| Kernel | File | Function Name | Block Size | Grid Size | Shared Mem |
|---|---|---|---|---|---|
| RMS Norm | `rmsnorm.hip` | `rmsnorm_f16` | 256 | rows | Warp reduction |
| Fused Add+RmsNorm | `fused_add_rmsnorm.hip` | `fused_add_rmsnorm_f16` | 256 | rows | Warp reduction |
| Per-Head RmsNorm | `per_head_rmsnorm.hip` | `per_head_rmsnorm_f16` | 256 | heads × seqLen | Warp reduction |
| RoPE | `rope.hip` | `rope_f16` | 256 | seqLen × numHeads | None |
| Attention | `attention.hip` | `attention_f16` | 256 | numHeads × seqQ | Per-head scores |
| SwiGLU | `swiglu.hip` | `swiglu_f16` | 256 | ceil(n/256) | None |
| Add | `add.hip` | `add_f16` | 256 | ceil(n/256) | None |
| Bias Add | `bias_add.hip` | `bias_add_f16` | 256 | ceil(n/256) | None |
| Softmax | `softmax.hip` | `softmax_f16` | 256 | rows | Warp reduction |
| Embedding (F32) | `embedding.hip` | `embedding_lookup_f32` | 256 | seqLen | None |
| Embedding (F16) | `embedding.hip` | `embedding_lookup_f16` | 256 | seqLen | None |
| Embedding (Q8_0) | `embedding.hip` | `embedding_lookup_q8_0` | 256 | seqLen | None |

**Dequantization (quantized weights → FP16 scratch):**

| Kernel | File | Function Name |
|---|---|---|
| Dequant Q8_0 | `dequant.hip` | `dequant_q8_0_f16` |
| Dequant Q4_0 | `dequant.hip` | `dequant_q4_0_f16` |
| Dequant Q5_0 | `dequant.hip` | `dequant_q5_0_f16` |
| Dequant Q4_K | `dequant.hip` | `dequant_q4_k_f16` |
| Dequant Q5_K | `dequant.hip` | `dequant_q5_k_f16` |
| Dequant Q6_K | `dequant.hip` | `dequant_q6_k_f16` |

**Quantized GEMV (decode path — operate directly on quantized weights):**

| Kernel | File | Function Name |
|---|---|---|
| Q8_0 GEMV | `quantized_gemv.hip` | `quantized_gemv_q8_0` |
| Q4_K GEMV | `quantized_gemv.hip` | `quantized_gemv_q4_k` |
| Q6_K GEMV | `quantized_gemv.hip` | `quantized_gemv_q6_k` |

**Conversion:**

| Kernel | File | Function Name |
|---|---|---|
| FP16→FP32 | `convert.hip` | `convert_f16_to_f32` |
| FP32→FP16 | `convert.hip` | `convert_f32_to_f16` |

GEMM/GEMV for prefill uses rocBLAS (`rocblas_hgemm` / `rocblas_gemm_ex`) directly — no custom kernel needed.

The source of truth for kernel correctness is the llama.cpp ROCm backend (`ggml-rocm/`); cross-reference there before diverging from established numerical patterns.

---

## `IBackend` Integration

The ROCm backend lives in `DotLLM.RoCm` and implements `IBackend` from `DotLLM.Core`. It is consumed identically to `DotLLM.Cuda`:

```csharp
// Conceptual — mirrors CudaBackend registration
services.AddSingleton<IBackend>(sp =>
{
    var rocmBackend = new RoCmBackend(deviceIndex: 0);
    rocmBackend.Initialize(); // hipInit + context create + module load + rocBLAS init
    return rocmBackend;
});
```

The `DotLLM.RoCm` project is a separate NuGet package per CLAUDE.md §Design Philosophy point 4: *"separate NuGet packages per backend (CPU, CUDA, ROCm)"*.

---

## Prerequisites

### Runtime Requirements

- **AMD GPU**: RDNA 2 (`gfx1030`) or newer for consumer; CDNA 2 (`gfx90a`) or newer for data-center/Matrix Core acceleration. Minimum: any GFX9 or GFX10 family GPU.
- **ROCm / HIP SDK version**: ROCm 5.7+ on Linux; AMD HIP SDK 5.5+ on Windows. Recommended: ROCm/HIP SDK 6.1+ for stable `gfx1100` and `gfx942` support.
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+) **or** Windows 10/11 (via AMD HIP SDK installer). Both are fully supported including rocBLAS.
- **`libamdhip64.so` (Linux)**: installed with ROCm or via `rocm-dev` package. Typically in `/opt/rocm/lib`.
- **`amdhip64.dll` (Windows)**: installed by the [AMD HIP SDK installer](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html). Typically in `C:\Program Files\AMD\ROCm\<ver>\bin`.
- **`librocblas.so` (Linux)**: installed via `rocblas` or `rocm-libs` package. Typically in `/opt/rocm/lib`.
- **`rocblas.dll` (Windows)**: bundled in the HIP SDK installer. Located in `%HIP_PATH%\bin`. Requires the `rocblas/` Tensile kernel subdirectory alongside it.

### Build Requirements (kernel compilation only)

- **ROCm 5.7+ (Linux) / AMD HIP SDK 5.5+ (Windows)**: provides `hipcc` compiler. Only needed to recompile `.hip` → `.hsaco` files. Pre-compiled HSACO fat binaries can be distributed without ROCm/HIP SDK on end-user machines.
- **No CMake, no C/C++ compiler beyond `hipcc`/clang-offload**.
- **`hipify-clang`** (optional): required only if translating `.cu` → `.hip` source. Ships with ROCm and the HIP SDK.

### Checking Available GPU and GFX Target

```bash
# Identify GFX target for current GPU (needed to select correct HSACO slice or compile flags)
rocminfo | grep "gfx"
# or:
/opt/rocm/bin/rocm_agent_enumerator
```

---

## Future Work

- **Flash Attention**: replace naive attention kernel with tiled flash attention using `rocWMMA` (AMD's `wmma`-equivalent for CDNA Matrix Cores). Reference: `flash-attention` ROCm fork and llama.cpp `ggml-rocm/` flash attention kernels.
- **Fused quantized GEMM for prefill**: rock equivalent of Marlin/llama.cpp MMQ — dequant-in-register fused matmul to eliminate per-projection dequant→scratch overhead. Decode path already uses custom quantized GEMV.
- **Multi-stream pipelining** (mirrors CUDA Step 32): overlap H2D transfer with compute across layers using multiple `hipStream_t` handles.
- **RCCL integration** (mirrors NCCL Step 51): multi-GPU tensor parallelism. RCCL (`librccl.so`) is AMD's drop-in equivalent to NCCL — same P/Invoke pattern, no shared library needed.
- **Windows HIP SDK**: rocBLAS ships in the AMD HIP SDK Windows installer (`rocblas.dll` in `%HIP_PATH%\bin`) and is fully functional. Windows support is production-grade from HIP SDK 5.5+. The `HipLibraryResolver` handles path discovery on both platforms.
- **GFX11 (RDNA 3) WMMA kernels**: `gfx1100` exposes `WMMA` (Wave Matrix Multiply Accumulate) instructions analogous to CUDA `wmma`. Custom attention kernels can exploit these directly for consumer AMD GPU acceleration when rocBLAS Matrix Cores are unavailable.
- **FP8 support (MI300X / gfx942)**: CDNA 3 native FP8 (`__hip_fp8_e4m3_fnuz`) enables quantized prefill without dequant overhead. Requires ROCm 6.0+.
- **NVRTC equivalent (hipRTC)**: AMD provides `hipRTC` (`libhiprtc.so`) for runtime compilation of `.hip` source to code objects — eliminates the `hipcc` build step entirely, enabling kernel updates without recompiling. API mirrors NVRTC.
