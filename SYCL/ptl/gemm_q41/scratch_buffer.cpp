size_t getScratchBufferSize_gemm_q41(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    return (maxBatch * maxHidden) * sizeof(fp16) + 64 * maxHidden * sizeof(float);
}