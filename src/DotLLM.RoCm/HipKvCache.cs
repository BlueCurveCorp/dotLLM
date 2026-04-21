using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;
using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// GPU-resident KV-cache storing FP16 key and value vectors per layer via HIP.
/// </summary>
public sealed class HipKvCache : IKvCache
{
    private readonly nint[] _keys;
    private readonly nint[] _values;
    private readonly int _numLayers;
    private readonly int _kvStride;
    private readonly int _maxSeqLen;
    private int _currentLength;

    public int CurrentLength => _currentLength;
    public int MaxLength => _maxSeqLen;

    public HipKvCache(int numLayers, int numKvHeads, int headDim, int maxSeqLen)
    {
        _numLayers = numLayers;
        _kvStride = numKvHeads * headDim;
        _maxSeqLen = maxSeqLen;
        _keys = new nint[numLayers];
        _values = new nint[numLayers];

        long bytesPerLayer = (long)maxSeqLen * _kvStride * sizeof(ushort);
        for (int i = 0; i < numLayers; i++)
        {
            HipApi.hipMalloc(out _keys[i], (nuint)bytesPerLayer).ThrowOnError();
            HipApi.hipMalloc(out _values[i], (nuint)bytesPerLayer).ThrowOnError();
        }
    }

    internal void UpdateDevice(nint keysDevice, nint valuesDevice, ReadOnlySpan<int> positions, int seqLen, int layerIndex, nint stream)
    {
        long rowBytes = (long)_kvStride * sizeof(ushort);
        bool contiguous = seqLen > 0;
        for (int i = 0; i < seqLen; i++)
        {
            if ((uint)positions[i] >= (uint)_maxSeqLen)
            {
                throw new ArgumentOutOfRangeException(nameof(positions), $"Position {positions[i]} at index {i} exceeds max KV-cache length {_maxSeqLen}.");
            }

            if (i > 0 && positions[i] != positions[i - 1] + 1)
            {
                contiguous = false;
            }
        }

        if (contiguous && seqLen > 1)
        {
            long bulkBytes = (long)seqLen * rowBytes;
            nint kDst = _keys[layerIndex] + (nint)(positions[0] * rowBytes);
            nint vDst = _values[layerIndex] + (nint)(positions[0] * rowBytes);
            HipApi.hipMemcpyDtoDAsync(kDst, keysDevice, (nuint)bulkBytes, stream).ThrowOnError();
            HipApi.hipMemcpyDtoDAsync(vDst, valuesDevice, (nuint)bulkBytes, stream).ThrowOnError();
        }
        else
        {
            for (int i = 0; i < seqLen; i++)
            {
                int pos = positions[i];
                nint kSrc = keysDevice + (nint)(i * rowBytes);
                nint vSrc = valuesDevice + (nint)(i * rowBytes);
                nint kDst = _keys[layerIndex] + (nint)(pos * rowBytes);
                nint vDst = _values[layerIndex] + (nint)(pos * rowBytes);
                HipApi.hipMemcpyDtoDAsync(kDst, kSrc, (nuint)rowBytes, stream).ThrowOnError();
                HipApi.hipMemcpyDtoDAsync(vDst, vSrc, (nuint)rowBytes, stream).ThrowOnError();
            }
        }

        int maxPos = positions[0];
        for (int i = 1; i < seqLen; i++)
        {
            if (positions[i] > maxPos)
            {
                maxPos = positions[i];
            }
        }

        int newLength = maxPos + 1;

        if (newLength > _currentLength)
        {
            _currentLength = newLength;
        }
    }

    internal nint GetKeysPtr(int layerIndex) => _keys[layerIndex];

    internal nint GetValuesPtr(int layerIndex) => _values[layerIndex];


    public void Update(ITensor keys, ITensor values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use UpdateDevice().");

    public void Update(TensorRef keys, TensorRef values, ReadOnlySpan<int> positions, int layerIndex)
        => throw new NotSupportedException("Use UpdateDevice().");

    public ITensor GetKeys(int layerIndex) => throw new NotSupportedException("Use GetKeysPtr().");

    public ITensor GetValues(int layerIndex) => throw new NotSupportedException("Use GetValuesPtr().");

    public TensorRef GetKeysRef(int layerIndex) => new(_currentLength, _kvStride, DType.Float16, 0, _keys[layerIndex]);

    public TensorRef GetValuesRef(int layerIndex) => new(_currentLength, _kvStride, DType.Float16, 0, _values[layerIndex]);

    public void Rollback(int length)
    {
        if ((uint)length > (uint)_currentLength)
        {
            throw new ArgumentOutOfRangeException(nameof(length));
        }

        _currentLength = length;
    }

    public void Dispose()
    {
        for (int i = 0; i < _numLayers; i++)
        {
            if (_keys[i] != nint.Zero)
            {
                HipApi.hipFree(_keys[i]);
                _keys[i] = nint.Zero;
            }
            if (_values[i] != nint.Zero)
            {
                HipApi.hipFree(_values[i]);
                _values[i] = nint.Zero;
            }
        }
    }
}
