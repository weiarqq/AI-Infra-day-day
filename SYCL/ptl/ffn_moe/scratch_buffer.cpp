size_t getScratchBufferSize_ffnmoe(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    // hard coded for Qwen3 A3B
    uint32_t total_experts = 128;
    uint32_t selected_experts = 8;
    uint32_t input_len = 2048;
    uint32_t hidden_len = 768;
    uint32_t token_len = maxBatch;
    return      token_len * input_len * sizeof(fp16) // input fp16
            +   total_experts * input_len * sizeof(fp16) // router weights fp16
            +   token_len * total_experts * sizeof(float) // router outputs
            +   token_len * selected_experts * sizeof(uint32_t) // indexes
            +   token_len * selected_experts * sizeof(float) // weights
            +   token_len * selected_experts * sizeof(uint32_t) // scattered to offsets
            +   token_len * selected_experts * input_len * sizeof(fp16)  // scattered inputs
            +   token_len * selected_experts * input_len * sizeof(fp16)  // scattered outputs
            +   token_len * hidden_len * sizeof(fp16)  // up result
            +   token_len * hidden_len * sizeof(fp16)  // gate result
            +   token_len * hidden_len * sizeof(fp16)  // silu result
            ;
}