using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// GPU-resident quantized KV-cache with dual-region storage (quantized buffer + FP16 window).
/// </summary>
public sealed class HipQuantizedKvCache : IQuantizedKvCache
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int Q4_0BlockBytes = 18;

    private readonly nint[] _keysQuant;
    private readonly nint[] _valuesQuant;
    private readonly nint[]? _keysWindow;
    private readonly nint[]? _valuesWindow;
    private readonly int _numLayers, _kvStride, _maxSeqLen, _windowSize, _keyQuantRowBytes, _valueQuantRowBytes;
    private readonly int[] _layerQuantizedLength;
    private int _currentLength, _quantizedLength;
    private nint _kScratch, _vScratch;

    public int CurrentLength => _currentLength;
    public int MaxLength => _maxSeqLen;
    public int QuantizedLength => _quantizedLength;
    public int WindowLength => _windowSize > 0 ? Math.Min(_currentLength, _windowSize) : 0;
    public int WindowCapacity => _windowSize;
    public KvCacheDType KeyDType { get; }
    public KvCacheDType ValueDType { get; }
    public int KeyQuantizedRowBytes => _keyQuantRowBytes;
    public int ValueQuantizedRowBytes => _valueQuantRowBytes;
    public long AllocatedBytes { get; }

    public HipQuantizedKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen, KvCacheConfig config)
    {
        _numLayers = numLayers;
        _kvStride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _windowSize = config.MixedPrecisionWindowSize;
        KeyDType = config.KeyDType;
        ValueDType = config.ValueDType;

        if (_kvStride % BlockSize != 0)
        {
            throw new ArgumentException($"kvStride ({_kvStride}) must be a multiple of {BlockSize} for quantization.");
        }

        _keyQuantRowBytes = ComputeQuantRowBytes(_kvStride, config.KeyDType);
        _valueQuantRowBytes = ComputeQuantRowBytes(_kvStride, config.ValueDType);
        _layerQuantizedLength = new int[numLayers];
        _keysQuant = new nint[numLayers];
        _valuesQuant = new nint[numLayers];
        long totalBytes = 0;

        for (int i = 0; i < numLayers; i++)
        {
            HipApi.hipMalloc(out _keysQuant[i], (nuint)((long)maxSeqLen * _keyQuantRowBytes)).ThrowOnError();
            HipApi.hipMalloc(out _valuesQuant[i], (nuint)((long)maxSeqLen * _valueQuantRowBytes)).ThrowOnError();
            totalBytes += (long)((_keyQuantRowBytes + _valueQuantRowBytes) * maxSeqLen);
        }

        if (_windowSize > 0)
        {
            _keysWindow = new nint[numLayers];
            _valuesWindow = new nint[numLayers];
            nuint wBytes = (nuint)((long)_windowSize * _kvStride * sizeof(ushort));
            for (int i = 0; i < numLayers; i++)
            {
                HipApi.hipMalloc(out _keysWindow[i], wBytes).ThrowOnError();
                HipApi.hipMalloc(out _valuesWindow[i], wBytes).ThrowOnError();
                totalBytes += (long)(wBytes * 2);
            }
        }

        long scrBytes = (long)maxSeqLen * _kvStride * sizeof(ushort);
        HipApi.hipMalloc(out _kScratch, (nuint)scrBytes).ThrowOnError();
        HipApi.hipMalloc(out _vScratch, (nuint)scrBytes).ThrowOnError();
        totalBytes += scrBytes * 2;
        AllocatedBytes = totalBytes;
    }

    internal void UpdateDevice(nint keysDevice, nint valuesDevice, ReadOnlySpan<int> positions, int seqLen,
                                int layerIndex, nint stream, HipKernels kernels)
    {
        long fp16RowBytes = (long)_kvStride * sizeof(ushort);
        int maxPos = positions[0];

        for (int i = 1; i < seqLen; i++)
        {
            if (positions[i] > maxPos)
            {
                maxPos = positions[i];
            }
        }

        int newLength = maxPos + 1;

        if (_windowSize > 0)
        {
            int prevQ = _layerQuantizedLength[layerIndex];
            int newQ = Math.Max(0, newLength - _windowSize);
            for (int ep = prevQ; ep < newQ; ep++)
            {
                int ri = ep % _windowSize;
                kernels.LaunchQuantKv(_keysWindow![layerIndex] + (nint)(ri * fp16RowBytes),
                    _keysQuant[layerIndex] + (nint)((long)ep * _keyQuantRowBytes), _kvStride, KeyDType, stream);

                kernels.LaunchQuantKv(_valuesWindow![layerIndex] + (nint)(ri * fp16RowBytes),
                    _valuesQuant[layerIndex] + (nint)((long)ep * _valueQuantRowBytes), _kvStride, ValueDType, stream);
            }
            _layerQuantizedLength[layerIndex] = newQ;

            for (int i = 0; i < seqLen; i++)
            {
                int ri = positions[i] % _windowSize;
                HipApi.hipMemcpyDtoDAsync(_keysWindow![layerIndex] + (nint)(ri * fp16RowBytes), keysDevice + (nint)(i * fp16RowBytes), (nuint)fp16RowBytes, stream).ThrowOnError();

                HipApi.hipMemcpyDtoDAsync(_valuesWindow![layerIndex] + (nint)(ri * fp16RowBytes), valuesDevice + (nint)(i * fp16RowBytes), (nuint)fp16RowBytes, stream).ThrowOnError();
            }
            _quantizedLength = newQ;
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];

                kernels.LaunchQuantKv(keysDevice + (nint)(i * fp16RowBytes), _keysQuant[layerIndex] + (nint)((long)pos * _keyQuantRowBytes), _kvStride, KeyDType, stream);

                kernels.LaunchQuantKv(valuesDevice + (nint)(i * fp16RowBytes), _valuesQuant[layerIndex] + (nint)((long)pos * _valueQuantRowBytes), _kvStride, ValueDType, stream);
            }
            _quantizedLength = newLength;
        }
        _currentLength = newLength;
    }

    internal (nint kPtr, nint vPtr) PrepareAttentionScratch(int layerIndex, nint stream, HipKernels kernels)
    {
        long fp16RowBytes = (long)_kvStride * sizeof(ushort);

        if (_quantizedLength > 0)
        {
            int totalEl = _quantizedLength * _kvStride;

            kernels.LaunchDequantToF16(_keysQuant[layerIndex], KeyDType == KvCacheDType.Q8_0 ? Core.Configuration.QuantizationType.Q8_0 : Core.Configuration.QuantizationType.Q4_0, _kScratch, totalEl, stream);

            kernels.LaunchDequantToF16(_valuesQuant[layerIndex], ValueDType == KvCacheDType.Q8_0 ? Core.Configuration.QuantizationType.Q8_0 : Core.Configuration.QuantizationType.Q4_0, _vScratch, totalEl, stream);
        }

        int wLen = WindowLength;

        if (wLen > 0 && _keysWindow != null && _valuesWindow != null)
        {
            int ri = _quantizedLength % _windowSize;

            if (ri + wLen <= _windowSize)
            {
                long bb = (long)wLen * fp16RowBytes;
                HipApi.hipMemcpyDtoDAsync(_kScratch + (nint)(_quantizedLength * fp16RowBytes),
                    _keysWindow[layerIndex] + (nint)(ri * fp16RowBytes), (nuint)bb, stream).ThrowOnError();

                HipApi.hipMemcpyDtoDAsync(_vScratch + (nint)(_quantizedLength * fp16RowBytes), _valuesWindow[layerIndex] + (nint)(ri * fp16RowBytes), (nuint)bb, stream).ThrowOnError();
            }
            else
            {
                int tl = _windowSize - ri, hl = wLen - tl;
                long tb = (long)tl * fp16RowBytes;
                nint kOff = _kScratch + (nint)(_quantizedLength * fp16RowBytes);
                nint vOff = _vScratch + (nint)(_quantizedLength * fp16RowBytes);
                HipApi.hipMemcpyDtoDAsync(kOff, _keysWindow[layerIndex] + (nint)(ri * fp16RowBytes), (nuint)tb, stream).ThrowOnError();
                HipApi.hipMemcpyDtoDAsync(vOff, _valuesWindow[layerIndex] + (nint)(ri * fp16RowBytes), (nuint)tb, stream).ThrowOnError();

                if (hl > 0)
                {
                    long hb = (long)hl * fp16RowBytes;
                    HipApi.hipMemcpyDtoDAsync(kOff + (nint)tb, _keysWindow[layerIndex], (nuint)hb, stream).ThrowOnError();
                    HipApi.hipMemcpyDtoDAsync(vOff + (nint)tb, _valuesWindow[layerIndex], (nuint)hb, stream).ThrowOnError();
                }
            }
        }
        return (_kScratch, _vScratch);
    }

    public nint GetQuantizedKeysPtr(int layerIndex) => _keysQuant[layerIndex];
    public nint GetQuantizedValuesPtr(int layerIndex) => _valuesQuant[layerIndex];
    public nint GetWindowKeysPtr(int layerIndex) => _keysWindow != null ? _keysWindow[layerIndex] : 0;
    public nint GetWindowValuesPtr(int layerIndex) => _valuesWindow != null ? _valuesWindow[layerIndex] : 0;

    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use UpdateDevice().");
    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use UpdateDevice().");
    public ITensor GetKeys(int layerIndex) => throw new NotSupportedException("Use PrepareAttentionScratch().");
    public ITensor GetValues(int layerIndex) => throw new NotSupportedException("Use PrepareAttentionScratch().");
    public TensorRef GetKeysRef(int layerIndex) => new(WindowLength, _kvStride, DType.Float16, 0, _keysWindow != null ? _keysWindow[layerIndex] : 0);
    public TensorRef GetValuesRef(int layerIndex) => new(WindowLength, _kvStride, DType.Float16, 0, _valuesWindow != null ? _valuesWindow[layerIndex] : 0);

    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }
        _currentLength = length;

        if (_quantizedLength > length)
        {
            _quantizedLength = length;
        }
    }

    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_keysQuant[i] != nint.Zero)
            {
                HipApi.hipFree(_keysQuant[i]); _keysQuant[i] = 0;
            }

            if (_valuesQuant[i] != nint.Zero)
            {
                HipApi.hipFree(_valuesQuant[i]); _valuesQuant[i] = 0;
            }

            if (_keysWindow != null && _keysWindow[i] != nint.Zero)
            {
                HipApi.hipFree(_keysWindow[i]); _keysWindow[i] = 0;
            }

            if (_valuesWindow != null && _valuesWindow[i] != nint.Zero)
            {
                HipApi.hipFree(_valuesWindow[i]); _valuesWindow[i] = 0;
            }
        }

        if (_kScratch != nint.Zero)
        {
            HipApi.hipFree(_kScratch); _kScratch = 0;
        }

        if (_vScratch != nint.Zero)
        {
            HipApi.hipFree(_vScratch); _vScratch = 0;
        }
    }

    private static int ComputeQuantRowBytes(int kvStride, KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.F32 => kvStride * sizeof(ushort),
        KvCacheDType.Q8_0 => kvStride / BlockSize * Q8_0BlockBytes,
        KvCacheDType.Q4_0 => kvStride / BlockSize * Q4_0BlockBytes,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype))
    };
}
