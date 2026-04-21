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

    /// <summary>
    /// FP16 GEMM — C = alpha * op(A) * op(B) + beta * C, all FP16.
    /// Matrix Cores used automatically on CDNA/RDNA3 when dims are multiples of 16.
    /// Row-major trick: compute C^T = B^T @ A^T via swapped args.
    /// </summary>
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

    /// <summary>
    /// Mixed-precision GEMM — FP16 input, FP32 accumulate.
    /// </summary>
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

    // ── rocBLAS constants ──

    internal const int ROCBLAS_OPERATION_NONE = 0;
    internal const int ROCBLAS_OPERATION_TRANSPOSE = 1;
    internal const int ROCBLAS_OPERATION_CONJUGATE_TRANSPOSE = 2;

    internal const int ROCBLAS_DEFAULT_MATH = 0;
    internal const int ROCBLAS_XF32_XDL_MATH_OP = 1;

    internal const int ROCBLAS_DATATYPE_F16_R = 150;
    internal const int ROCBLAS_DATATYPE_F32_R = 151;

    internal const int ROCBLAS_COMPUTE_F32 = 300;

    internal const int ROCBLAS_GEMM_DEFAULT = -1;
}
