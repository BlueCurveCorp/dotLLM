using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// rocBLAS GEMM/GEMV wrappers for linear projections.
/// Weight matrices are FP16, stored as [outputDim, inputDim] (row-major).
/// Input/output are FP16. FP32 accumulation via rocblas_gemm_ex.
/// </summary>
public static class HipGemm
{
    /// <summary>
    /// Linear projection: Y_f16[m, n] = X_f16[m, k] x W_f16^T.
    /// FP32 accumulation, FP16 output.
    /// Row-major trick: swap AxB -> BxA via swapped operands.
    /// </summary>
    public static unsafe void LinearF16(nint handle, nint xF16, nint wF16, nint yF16, int m, int k, int n, nint stream)
    {
        RocBlasApi.rocblas_set_stream(handle, stream).ThrowOnRocBlasError();

        float alpha = 1.0f;
        float beta = 0.0f;

        RocBlasApi.rocblas_gemm_ex(
            handle,
            RocBlasApi.ROCBLAS_OPERATION_TRANSPOSE,
            RocBlasApi.ROCBLAS_OPERATION_NONE,
            n, m, k,
            (nint)(&alpha),
            wF16, RocBlasApi.ROCBLAS_DATATYPE_F16_R, k,
            xF16, RocBlasApi.ROCBLAS_DATATYPE_F16_R, k,
            (nint)(&beta),
            yF16, RocBlasApi.ROCBLAS_DATATYPE_F16_R, n,
            yF16, RocBlasApi.ROCBLAS_DATATYPE_F16_R, n,
            RocBlasApi.ROCBLAS_COMPUTE_F32,
            RocBlasApi.ROCBLAS_GEMM_DEFAULT,
            0, 0
        ).ThrowOnRocBlasError();
    }

    /// <summary>
    /// GEMV for single token: y_f16[n] = W_f16[n,k] x x_f16[k].
    /// </summary>
    public static unsafe void GemvF16(nint handle, nint wF16, nint xF16, nint yF16, int n, int k, nint stream) => LinearF16(handle, xF16, wF16, yF16, 1, k, n, stream);
}
