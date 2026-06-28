size_t getScratchBufferSize_ffn(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    return maxBatch * maxHidden * sizeof(sycl::half) * 3;
}