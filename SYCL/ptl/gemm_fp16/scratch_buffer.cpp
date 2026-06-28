
size_t getVisionScratchBufferSize_gemm(uint32_t maxBatch, uint32_t head_num, uint32_t maxHidden)
{
    //size_t temp1 = head_num * maxBatch * maxBatch;
    size_t temp1 = 0;
    size_t temp2 = maxBatch * maxHidden;
    size_t count = temp1 > temp2? temp1: temp2;
    return count * sizeof(sycl::half);
}
