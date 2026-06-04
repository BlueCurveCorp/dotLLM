using System.Numerics;
using DotLLM.RoCm.Interop;

namespace DotLLM.RoCm;

/// <summary>
/// Pre-allocated GPU scratch buffers for the ROCm forward pass.
/// </summary>
internal sealed class HipForwardState : IDisposable
{
    private readonly int _hiddenSize;
    private readonly int _numHeads;
    private readonly int _numKvHeads;
    private readonly int _headDim;
    private readonly int _intermediateSize;
    private readonly int _vocabSize;
    private int _currentSeqLen;

    public long AllocatedBytes { get; private set; }

    public nint HiddenState;
    public nint Residual;
    public nint NormOutput;
    public nint Q;
    public nint K;
    public nint V;
    public nint AttnOutput;
    public nint FfnGate;
    public nint FfnUp;
    public nint SiluOutput;
    public nint LogitsF16;
    public nint LogitsF32;
    public nint GemmOutputF16;
    public nint DequantScratch;
    public nint TokenIdsDevice;
    public nint PositionsDevice;

    public HipForwardState(int hiddenSize, int numHeads, int numKvHeads, int headDim, int intermediateSize, int vocabSize)
    {
        _hiddenSize = hiddenSize;
        _numHeads = numHeads;
        _numKvHeads = numKvHeads;
        _headDim = headDim;
        _intermediateSize = intermediateSize;
        _vocabSize = vocabSize;

        LogitsF16 = AllocDevice((long)vocabSize * sizeof(ushort));
        LogitsF32 = AllocDevice((long)vocabSize * sizeof(float));

        long maxProj = Math.Max(
            (long)Math.Max(numHeads * headDim, numKvHeads * headDim) * hiddenSize,
            (long)intermediateSize * hiddenSize);
        DequantScratch = AllocDevice(maxProj * sizeof(ushort));

        EnsureCapacity(1);
    }

    public void EnsureCapacity(int seqLen)
    {
        if (seqLen <= _currentSeqLen) return;

        int cap = (int)BitOperations.RoundUpToPowerOf2((uint)seqLen);
        FreeSequenceBuffers();

        int h2 = sizeof(ushort);
        HiddenState = AllocDevice((long)cap * _hiddenSize * h2);
        Residual = AllocDevice((long)cap * _hiddenSize * h2);
        NormOutput = AllocDevice((long)cap * _hiddenSize * h2);
        Q = AllocDevice((long)cap * _numHeads * _headDim * h2);
        K = AllocDevice((long)cap * _numKvHeads * _headDim * h2);
        V = AllocDevice((long)cap * _numKvHeads * _headDim * h2);
        AttnOutput = AllocDevice((long)cap * _numHeads * _headDim * h2);
        FfnGate = AllocDevice((long)cap * _intermediateSize * h2);
        FfnUp = AllocDevice((long)cap * _intermediateSize * h2);
        SiluOutput = AllocDevice((long)cap * _intermediateSize * h2);

        long maxLayer = (long)cap * Math.Max(Math.Max(_numHeads * _headDim, _intermediateSize), _hiddenSize);
        long maxLmHead = _vocabSize;
        GemmOutputF16 = AllocDevice(Math.Max(maxLayer, maxLmHead) * h2);
        TokenIdsDevice = AllocDevice((long)cap * sizeof(int));
        PositionsDevice = AllocDevice((long)cap * sizeof(int));

        _currentSeqLen = cap;
    }

    private nint AllocDevice(long bytes)
    {
        HipApi.hipMalloc(out nint ptr, (nuint)bytes).ThrowOnError();
        AllocatedBytes += bytes;
        return ptr;
    }

    private void FreeIfNonZero(ref nint ptr)
    {
        if (ptr != 0) { HipApi.hipFree(ptr); ptr = 0; }
    }

    private void FreeSequenceBuffers()
    {
        FreeIfNonZero(ref HiddenState); FreeIfNonZero(ref Residual); FreeIfNonZero(ref NormOutput);
        FreeIfNonZero(ref Q); FreeIfNonZero(ref K); FreeIfNonZero(ref V);
        FreeIfNonZero(ref AttnOutput); FreeIfNonZero(ref FfnGate); FreeIfNonZero(ref FfnUp);
        FreeIfNonZero(ref SiluOutput); FreeIfNonZero(ref GemmOutputF16);
        FreeIfNonZero(ref TokenIdsDevice); FreeIfNonZero(ref PositionsDevice);
    }

    public void Dispose()
    {
        FreeSequenceBuffers();
        FreeIfNonZero(ref LogitsF16);
        FreeIfNonZero(ref LogitsF32);
        FreeIfNonZero(ref DequantScratch);
        _currentSeqLen = 0;
    }
}
