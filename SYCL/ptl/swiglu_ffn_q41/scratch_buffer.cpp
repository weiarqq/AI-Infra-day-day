size_t getScratchBufferSize_swiglu_ffn(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    return maxBatch * maxHidden * 2 * sizeof(fp16) + 64 * maxHidden * sizeof(float);
}