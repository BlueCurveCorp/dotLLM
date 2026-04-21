using DotLLM.Core.Configuration;
using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// Loads all HSACO kernel modules and provides typed launch methods for each kernel.
/// </summary>
public sealed unsafe class HipKernels : IDisposable
{
    private const int BlockSize = 256;
    private const int MaxDequantGridSize = 256;

    private readonly HipModule _rmsnormModule;
    private readonly HipModule _ropeModule;
    private readonly HipModule _swigluModule;
    private readonly HipModule _addModule;
    private readonly HipModule _softmaxModule;
    private readonly HipModule _embeddingModule;
    private readonly HipModule _attentionModule;
    private readonly HipModule _biasAddModule;
    private readonly HipModule _perHeadRmsNormModule;
    private readonly HipModule _convertModule;
    private readonly HipModule _dequantModule;
    private readonly HipModule _quantizedGemvModule;
    private readonly HipModule _fusedAddRmsNormModule;
    private readonly HipModule _rmsnormF32InModule;
    private readonly HipModule _addF32Module;
    private readonly HipModule _embeddingF32OutModule;
    private readonly HipModule _ropeF32Module;
    private readonly HipModule _attentionF32Module;
    private readonly HipModule _swigluF32Module;
    private readonly HipModule _biasAddF32Module;
    private readonly HipModule _perHeadRmsNormF32Module;
    private readonly HipModule _rmsnormF32Module;
    private readonly HipModule _quantizedGemvF32InModule;

    private readonly nint _rmsnormFunc;
    private readonly nint _rmsnormF32Func;
    private readonly nint _quantizedGemvQ8_0F32InFunc;
    private readonly nint _fusedAddRmsNormFunc;
    private readonly nint _rmsnormF32InF16OutFunc;
    private readonly nint _addF32Func;
    private readonly nint _addF32F16Func;
    private readonly nint _embeddingF32OutF32Func;
    private readonly nint _embeddingF32OutF16Func;
    private readonly nint _embeddingF32OutQ8_0Func;
    private readonly nint _ropeF32Func;
    private readonly nint _attentionF32Func;
    private readonly nint _swigluF32Func;
    private readonly nint _biasAddF32Func;
    private readonly nint _perHeadRmsNormF32Func;
    private readonly nint _ropeFunc;
    private readonly nint _swigluFunc;
    private readonly nint _addFunc;
    private readonly nint _softmaxFunc;
    private readonly nint _embeddingF32Func;
    private readonly nint _embeddingF16Func;
    private readonly nint _embeddingQ8_0Func;
    private readonly nint _attentionFunc;
    private readonly nint _biasAddFunc;
    private readonly nint _perHeadRmsNormFunc;
    private readonly nint _convertF16ToF32Func;
    private readonly nint _convertF32ToF16Func;
    private readonly nint _quantizedGemvQ8_0Func;
    private readonly nint _quantizedGemvQ4_KFunc;
    private readonly nint _quantizedGemvQ5_0Func;
    private readonly nint _quantizedGemvQ5_KFunc;
    private readonly nint _quantizedGemvQ6_KFunc;
    private readonly nint _dequantQ8_0Func;
    private readonly nint _dequantQ4_0Func;
    private readonly nint _dequantQ5_0Func;
    private readonly nint _dequantQ4_KFunc;
    private readonly nint _dequantQ5_KFunc;
    private readonly nint _dequantQ6_KFunc;
    private readonly HipModule? _quantKvModule;
    private readonly nint _quantKvQ8_0Func;
    private readonly nint _quantKvQ4_0Func;

    /// <summary>
    /// Loads all HSACO modules from the specified directory.
    /// </summary>
    public HipKernels(string hsacoDir)
    {
        _rmsnormModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "rmsnorm.hsaco"));
        _ropeModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "rope.hsaco"));
        _swigluModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "swiglu.hsaco"));
        _addModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "add.hsaco"));
        _softmaxModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "softmax.hsaco"));
        _embeddingModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "embedding.hsaco"));
        _attentionModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "attention.hsaco"));
        _biasAddModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "bias_add.hsaco"));
        _perHeadRmsNormModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "per_head_rmsnorm.hsaco"));
        _convertModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "convert.hsaco"));
        _dequantModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "dequant.hsaco"));
        _quantizedGemvModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "quantized_gemv.hsaco"));
        _fusedAddRmsNormModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "fused_add_rmsnorm.hsaco"));
        _rmsnormF32InModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "rmsnorm_f32in.hsaco"));
        _addF32Module = HipModule.LoadFromFile(Path.Combine(hsacoDir, "add_f32.hsaco"));
        _embeddingF32OutModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "embedding_f32out.hsaco"));
        _ropeF32Module = HipModule.LoadFromFile(Path.Combine(hsacoDir, "rope_f32.hsaco"));
        _attentionF32Module = HipModule.LoadFromFile(Path.Combine(hsacoDir, "attention_f32.hsaco"));
        _swigluF32Module = HipModule.LoadFromFile(Path.Combine(hsacoDir, "swiglu_f32.hsaco"));
        _biasAddF32Module = HipModule.LoadFromFile(Path.Combine(hsacoDir, "bias_add_f32.hsaco"));
        _perHeadRmsNormF32Module = HipModule.LoadFromFile(Path.Combine(hsacoDir, "per_head_rmsnorm_f32.hsaco"));
        _rmsnormF32Module = HipModule.LoadFromFile(Path.Combine(hsacoDir, "rmsnorm_f32.hsaco"));
        _quantizedGemvF32InModule = HipModule.LoadFromFile(Path.Combine(hsacoDir, "quantized_gemv_f32in.hsaco"));

        _rmsnormFunc = _rmsnormModule.GetFunction("rmsnorm_f16");
        _rmsnormF32Func = _rmsnormF32Module.GetFunction("rmsnorm_f32");
        _quantizedGemvQ8_0F32InFunc = _quantizedGemvF32InModule.GetFunction("quantized_gemv_q8_0_f32in");
        _fusedAddRmsNormFunc = _fusedAddRmsNormModule.GetFunction("fused_add_rmsnorm_f16");
        _rmsnormF32InF16OutFunc = _rmsnormF32InModule.GetFunction("rmsnorm_f32in_f16out");
        _addF32Func = _addF32Module.GetFunction("add_f32");
        _addF32F16Func = _addF32Module.GetFunction("add_f32_f16");
        _embeddingF32OutF32Func = _embeddingF32OutModule.GetFunction("embedding_lookup_f32_f32out");
        _embeddingF32OutF16Func = _embeddingF32OutModule.GetFunction("embedding_lookup_f16_f32out");
        _embeddingF32OutQ8_0Func = _embeddingF32OutModule.GetFunction("embedding_lookup_q8_0_f32out");
        _ropeF32Func = _ropeF32Module.GetFunction("rope_f32");
        _attentionF32Func = _attentionF32Module.GetFunction("attention_f32");
        _swigluF32Func = _swigluF32Module.GetFunction("swiglu_f32");
        _biasAddF32Func = _biasAddF32Module.GetFunction("bias_add_f32");
        _perHeadRmsNormF32Func = _perHeadRmsNormF32Module.GetFunction("per_head_rmsnorm_f32");
        _ropeFunc = _ropeModule.GetFunction("rope_f16");
        _swigluFunc = _swigluModule.GetFunction("swiglu_f16");
        _addFunc = _addModule.GetFunction("add_f16");
        _softmaxFunc = _softmaxModule.GetFunction("softmax_f16");
        _embeddingF32Func = _embeddingModule.GetFunction("embedding_lookup_f32");
        _embeddingF16Func = _embeddingModule.GetFunction("embedding_lookup_f16");
        _embeddingQ8_0Func = _embeddingModule.GetFunction("embedding_lookup_q8_0");
        _attentionFunc = _attentionModule.GetFunction("attention_f16");
        _biasAddFunc = _biasAddModule.GetFunction("bias_add_f16");
        _perHeadRmsNormFunc = _perHeadRmsNormModule.GetFunction("per_head_rmsnorm_f16");
        _convertF16ToF32Func = _convertModule.GetFunction("convert_f16_to_f32");
        _convertF32ToF16Func = _convertModule.GetFunction("convert_f32_to_f16");
        _quantizedGemvQ8_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q8_0");
        _quantizedGemvQ4_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q4_k");
        _quantizedGemvQ5_0Func = _quantizedGemvModule.GetFunction("quantized_gemv_q5_0");
        _quantizedGemvQ5_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q5_k");
        _quantizedGemvQ6_KFunc = _quantizedGemvModule.GetFunction("quantized_gemv_q6_k");
        _dequantQ8_0Func = _dequantModule.GetFunction("dequant_q8_0_f16");
        _dequantQ4_0Func = _dequantModule.GetFunction("dequant_q4_0_f16");
        _dequantQ5_0Func = _dequantModule.GetFunction("dequant_q5_0_f16");
        _dequantQ4_KFunc = _dequantModule.GetFunction("dequant_q4_k_f16");
        _dequantQ5_KFunc = _dequantModule.GetFunction("dequant_q5_k_f16");
        _dequantQ6_KFunc = _dequantModule.GetFunction("dequant_q6_k_f16");

        string quantKvPath = Path.Combine(hsacoDir, "quant_kv.hsaco");
        if (File.Exists(quantKvPath))
        {
            _quantKvModule = HipModule.LoadFromFile(quantKvPath);
            _quantKvQ8_0Func = _quantKvModule.GetFunction("quant_f16_to_q8_0");
            _quantKvQ4_0Func = _quantKvModule.GetFunction("quant_f16_to_q4_0");
        }
    }

    public void LaunchRmsNorm(nint input, nint weight, nint output, int hiddenSize, float eps, int rows, nint stream)
    {
        nint iArg = input, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float eArg = eps;
        void** args = stackalloc void*[] { &iArg, &wArg, &oArg, &nArg, &eArg };
        HipApi.hipModuleLaunchKernel(_rmsnormFunc, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchFusedAddRmsNorm(nint residual, nint x, nint weight, nint output, int hiddenSize, float eps, int rows, nint stream)
    {
        nint rArg = residual, xArg = x, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float eArg = eps;
        void** args = stackalloc void*[] { &rArg, &xArg, &wArg, &oArg, &nArg, &eArg };
        HipApi.hipModuleLaunchKernel(_fusedAddRmsNormFunc, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchRmsNormF32(nint input, nint weight, nint output, int hiddenSize, float eps, int rows, nint stream)
    {
        nint iArg = input, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float eArg = eps;
        void** args = stackalloc void*[] { &iArg, &wArg, &oArg, &nArg, &eArg };
        HipApi.hipModuleLaunchKernel(_rmsnormF32Func, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchQuantizedGemvF32In(nint quantWeight, nint xF32, nint yF32, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = xF32, yArg = yF32;
        int nArg = n, kArg = k;
        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
        HipApi.hipModuleLaunchKernel(_quantizedGemvQ8_0F32InFunc, (uint)n, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchRmsNormF32In(nint input, nint weight, nint output, int hiddenSize, float eps, int rows, nint stream)
    {
        nint iArg = input, wArg = weight, oArg = output;
        int nArg = hiddenSize;
        float eArg = eps;
        void** args = stackalloc void*[] { &iArg, &wArg, &oArg, &nArg, &eArg };
        HipApi.hipModuleLaunchKernel(_rmsnormF32InF16OutFunc, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchAddF32(nint a, nint b, nint output, int n, nint stream)
    {
        nint aArg = a, bArg = b, oArg = output;
        int nArg = n;
        void** args = stackalloc void*[] { &aArg, &bArg, &oArg, &nArg };
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_addF32Func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchAddF32F16(nint aF32, nint bF16, nint outputF32, int n, nint stream)
    {
        nint aArg = aF32, bArg = bF16, oArg = outputF32;
        int nArg = n;
        void** args = stackalloc void*[] { &aArg, &bArg, &oArg, &nArg };
        uint grid = (uint)((n + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_addF32F16Func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchEmbeddingLookupF32(nint embedTable, QuantizationType embedDtype, nint tokenIds, nint output, int seqLen, int hiddenSize, nint stream)
    {
        nint tArg = embedTable, iArg = tokenIds, oArg = output;
        int sArg = seqLen, hArg = hiddenSize;
        nint func = embedDtype switch
        {
            QuantizationType.F32 => _embeddingF32OutF32Func,
            QuantizationType.F16 => _embeddingF32OutF16Func,
            QuantizationType.Q8_0 => _embeddingF32OutQ8_0Func,
            _ => throw new NotSupportedException($"FP32 embedding lookup not supported for {embedDtype}.")
        };
        void** args = stackalloc void*[] { &tArg, &iArg, &oArg, &sArg, &hArg };
        HipApi.hipModuleLaunchKernel(func, (uint)seqLen, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchRoPEF32(nint q, nint k, nint positions, int seqLen, int numHeads, int numKvHeads, int headDim, int ropeDim, float theta, int ropeType, nint stream)
    {
        nint qArg = q, kArg = k, pArg = positions;
        int slArg = seqLen, nhArg = numHeads, nkArg = numKvHeads, hdArg = headDim, rdArg = ropeDim, rtArg = ropeType;
        float tArg = theta;
        void** args = stackalloc void*[] { &qArg, &kArg, &pArg, &slArg, &nhArg, &nkArg, &hdArg, &rdArg, &tArg, &rtArg };
        int pairs = seqLen * Math.Max(numHeads, numKvHeads) * (ropeDim / 2);
        uint grid = (uint)((pairs + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_ropeF32Func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchAttentionF32(nint q, nint k, nint v, nint output, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset, int slidingWindow, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, oArg = output;
        int sqArg = seqQ, skArg = seqKv, nhArg = numHeads, nkArg = numKvHeads, hdArg = headDim, poArg = positionOffset, swArg = slidingWindow;
        void** args = stackalloc void*[] { &qArg, &kArg, &vArg, &oArg, &sqArg, &skArg, &nhArg, &nkArg, &hdArg, &poArg, &swArg };
        int blocks = seqQ * numHeads;
        const int TileKv = 256;
        uint sharedMem = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));
        HipApi.hipModuleLaunchKernel(_attentionF32Func, (uint)blocks, 1, 1, BlockSize, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchSwiGLUF32(nint gate, nint up, nint output, int n, int seqLen, nint stream)
    {
        nint gArg = gate, uArg = up, oArg = output;
        int nArg = n, sArg = seqLen;
        void** args = stackalloc void*[] { &gArg, &uArg, &oArg, &nArg, &sArg };
        uint grid = (uint)((n * seqLen + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_swigluF32Func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchBiasAddF32(nint output, nint biasF16, int dim, int seqLen, nint stream)
    {
        nint oArg = output, bArg = biasF16;
        int dArg = dim, sArg = seqLen;
        void** args = stackalloc void*[] { &oArg, &bArg, &dArg, &sArg };
        uint grid = (uint)((dim * seqLen + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_biasAddF32Func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchPerHeadRmsNormF32(nint qk, nint weightF16, float eps, int numHeads, int headDim, int seqLen, nint stream)
    {
        nint qArg = qk, wArg = weightF16;
        float eArg = eps;
        int nhArg = numHeads, hdArg = headDim, sArg = seqLen;
        void** args = stackalloc void*[] { &qArg, &wArg, &eArg, &nhArg, &hdArg, &sArg };
        HipApi.hipModuleLaunchKernel(_perHeadRmsNormF32Func, (uint)(seqLen * numHeads), 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchRoPE(nint q, nint k, nint positions, int seqLen, int numHeads, int numKvHeads, int headDim, int ropeDim, float theta, int ropeType, nint stream)
    {
        nint qArg = q, kArg = k, pArg = positions;
        int slArg = seqLen, nhArg = numHeads, nkArg = numKvHeads, hdArg = headDim, rdArg = ropeDim, rtArg = ropeType;
        float tArg = theta;
        void** args = stackalloc void*[] { &qArg, &kArg, &pArg, &slArg, &nhArg, &nkArg, &hdArg, &rdArg, &tArg, &rtArg };
        int pairs = seqLen * Math.Max(numHeads, numKvHeads) * (ropeDim / 2);
        uint grid = (uint)((pairs + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_ropeFunc, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchSwiGLU(nint gate, nint up, nint output, int n, int seqLen, nint stream)
    {
        nint gArg = gate, uArg = up, oArg = output;
        int nArg = n, sArg = seqLen;
        void** args = stackalloc void*[] { &gArg, &uArg, &oArg, &nArg, &sArg };
        uint grid = (uint)((n / 2 + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_swigluFunc, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchAdd(nint a, nint b, nint output, int n, nint stream)
    {
        nint aArg = a, bArg = b, oArg = output;
        int nArg = n;
        void** args = stackalloc void*[] { &aArg, &bArg, &oArg, &nArg };
        uint grid = (uint)((n / 2 + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_addFunc, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchSoftmax(nint input, nint output, int rows, int cols, nint stream)
    {
        nint iArg = input, oArg = output;
        int rArg = rows, cArg = cols;
        void** args = stackalloc void*[] { &iArg, &oArg, &rArg, &cArg };
        HipApi.hipModuleLaunchKernel(_softmaxFunc, (uint)rows, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchEmbeddingLookup(nint embedTable, QuantizationType embedDtype, nint tokenIds, nint output, int seqLen, int hiddenSize, nint stream)
    {
        nint tArg = embedTable, iArg = tokenIds, oArg = output;
        int sArg = seqLen, hArg = hiddenSize;
        nint func = embedDtype switch
        {
            QuantizationType.F32 => _embeddingF32Func,
            QuantizationType.F16 => _embeddingF16Func,
            QuantizationType.Q8_0 => _embeddingQ8_0Func,
            _ => throw new NotSupportedException($"Embedding type {embedDtype} not supported on GPU.")
        };
        void** args = stackalloc void*[] { &tArg, &iArg, &oArg, &sArg, &hArg };
        HipApi.hipModuleLaunchKernel(func, (uint)seqLen, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchAttention(nint q, nint k, nint v, nint output, int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim, int positionOffset, int slidingWindow, nint stream)
    {
        nint qArg = q, kArg = k, vArg = v, oArg = output;
        int sqArg = seqQ, skArg = seqKv, nhArg = numHeads, nkArg = numKvHeads, hdArg = headDim, poArg = positionOffset, swArg = slidingWindow;
        void** args = stackalloc void*[] { &qArg, &kArg, &vArg, &oArg, &sqArg, &skArg, &nhArg, &nkArg, &hdArg, &poArg, &swArg };
        int blocks = seqQ * numHeads;
        const int TileKv = 256;
        uint sharedMem = (uint)((headDim + TileKv + headDim + 32) * sizeof(float));
        HipApi.hipModuleLaunchKernel(_attentionFunc, (uint)blocks, 1, 1, BlockSize, 1, 1, sharedMem, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchBiasAdd(nint output, nint bias, int dim, int seqLen, nint stream)
    {
        nint oArg = output, bArg = bias;
        int dArg = dim, sArg = seqLen;
        void** args = stackalloc void*[] { &oArg, &bArg, &dArg, &sArg };
        uint grid = (uint)((dim * seqLen / 2 + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_biasAddFunc, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchPerHeadRmsNorm(nint qk, nint weight, float eps, int numHeads, int headDim, int seqLen, nint stream)
    {
        nint qArg = qk, wArg = weight;
        float eArg = eps;
        int nhArg = numHeads, hdArg = headDim, sArg = seqLen;
        void** args = stackalloc void*[] { &qArg, &wArg, &eArg, &nhArg, &hdArg, &sArg };
        int blocks = seqLen * numHeads;
        HipApi.hipModuleLaunchKernel(_perHeadRmsNormFunc, (uint)blocks, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchConvertF16ToF32(nint src, nint dst, int n, nint stream)
    {
        nint sArg = src, dArg = dst;
        int nArg = n;
        void** args = stackalloc void*[] { &sArg, &dArg, &nArg };
        uint grid = (uint)((n / 2 + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_convertF16ToF32Func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchConvertF32ToF16(nint src, nint dst, int n, nint stream)
    {
        nint sArg = src, dArg = dst;
        int nArg = n;
        void** args = stackalloc void*[] { &sArg, &dArg, &nArg };
        uint grid = (uint)((n / 2 + BlockSize - 1) / BlockSize);
        HipApi.hipModuleLaunchKernel(_convertF32ToF16Func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void LaunchQuantizedGemv(nint quantWeight, QuantizationType qt, nint x, nint y, int n, int k, nint stream)
    {
        nint wArg = quantWeight, xArg = x, yArg = y;
        int nArg = n, kArg = k;
        nint func = qt switch
        {
            QuantizationType.Q8_0 => _quantizedGemvQ8_0Func,
            QuantizationType.Q4_K => _quantizedGemvQ4_KFunc,
            QuantizationType.Q5_0 => _quantizedGemvQ5_0Func,
            QuantizationType.Q5_K => _quantizedGemvQ5_KFunc,
            QuantizationType.Q6_K => _quantizedGemvQ6_KFunc,
            _ => 0
        };

        if (func == 0)
        {
            throw new NotSupportedException($"Quantized GEMV not supported for {qt}.");
        }

        void** args = stackalloc void*[] { &wArg, &xArg, &yArg, &nArg, &kArg };
        HipApi.hipModuleLaunchKernel(func, (uint)n, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public static bool HasQuantizedGemv(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_K or QuantizationType.Q5_0 or QuantizationType.Q5_K or QuantizationType.Q6_K;

    public void LaunchDequantToF16(nint src, QuantizationType srcDtype, nint dst, int totalElements, nint stream)
    {
        nint sArg = src, dArg = dst;

        switch (srcDtype)
        {
            case QuantizationType.F16:
                HipApi.hipMemcpyDtoD(dst, src, (nuint)(totalElements * 2)).ThrowOnError();
                return;
            case QuantizationType.F32:
                LaunchConvertF32ToF16(src, dst, totalElements, stream);
                return;
            case QuantizationType.Q8_0:
            case QuantizationType.Q4_0:
            case QuantizationType.Q5_0:
            case QuantizationType.Q4_K:
            case QuantizationType.Q5_K:
            case QuantizationType.Q6_K:
            {
                int blockCount = totalElements / (srcDtype == QuantizationType.Q4_K || srcDtype == QuantizationType.Q5_K || srcDtype == QuantizationType.Q6_K ? 256 : 32);
                int bArg = blockCount;
                void** args = stackalloc void*[] { &sArg, &dArg, &bArg };
                uint grid = (uint)System.Math.Min(blockCount + BlockSize / 8 - 1, MaxDequantGridSize);
                nint func = srcDtype switch
                {
                    QuantizationType.Q8_0 => _dequantQ8_0Func,
                    QuantizationType.Q4_0 => _dequantQ4_0Func,
                    QuantizationType.Q5_0 => _dequantQ5_0Func,
                    QuantizationType.Q4_K => _dequantQ4_KFunc,
                    QuantizationType.Q5_K => _dequantQ5_KFunc,
                    QuantizationType.Q6_K => _dequantQ6_KFunc,
                    _ => 0
                };

                if (func != 0)
                    HipApi.hipModuleLaunchKernel(func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
                return;
            }
            default:
                throw new NotSupportedException($"GPU dequantization not supported for {srcDtype}.");
        }
    }

    public unsafe void LaunchQuantKv(nint src, nint dst, int elementCount, KvCacheDType dtype, nint stream)
    {
        if (_quantKvModule == null)
        {
            throw new InvalidOperationException("KV-cache quantization kernels not available.");
        }

        int blockCount = elementCount / 32;

        nint sArg = src, dArg = dst;

        int bArg = blockCount;

        void** args = stackalloc void*[] { &sArg, &dArg, &bArg };

        uint grid = (uint)((blockCount + BlockSize - 1) / BlockSize);

        nint func = dtype switch
        {
            KvCacheDType.Q8_0 => _quantKvQ8_0Func,
            KvCacheDType.Q4_0 => _quantKvQ4_0Func,
            _ => throw new NotSupportedException($"KV quantization not supported for {dtype}")
        };

        HipApi.hipModuleLaunchKernel(func, grid, 1, 1, BlockSize, 1, 1, 0, stream, (nint)args, 0).ThrowOnError();
    }

    public void Dispose()
    {
        _rmsnormModule.Dispose();
        _ropeModule.Dispose();
        _swigluModule.Dispose();
        _addModule.Dispose();
        _softmaxModule.Dispose();
        _embeddingModule.Dispose();
        _attentionModule.Dispose();
        _biasAddModule.Dispose();
        _perHeadRmsNormModule.Dispose();
        _convertModule.Dispose();
        _dequantModule.Dispose();
        _quantizedGemvModule.Dispose();
        _fusedAddRmsNormModule.Dispose();
        _rmsnormF32InModule.Dispose();
        _addF32Module.Dispose();
        _embeddingF32OutModule.Dispose();
        _ropeF32Module.Dispose();
        _attentionF32Module.Dispose();
        _swigluF32Module.Dispose();
        _biasAddF32Module.Dispose();
        _perHeadRmsNormF32Module.Dispose();
        _rmsnormF32Module.Dispose();
        _quantizedGemvF32InModule.Dispose();
        _quantKvModule?.Dispose();
    }
}
