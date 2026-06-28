size_t getScratchBufferSize_gemm(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    return maxBatch * maxHidden * 2 / 64 * 66 + 4096;
}
