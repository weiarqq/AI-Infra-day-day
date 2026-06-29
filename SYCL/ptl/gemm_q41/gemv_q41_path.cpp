


bool runGemv_Q41Weights_L3(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt)
{
    if (batch == 1)
    {
        return runLinearQ41_L3(*q, output_len, input_len, batch, weights, inputs, scales, zps, outputs); //SYCL/ptl/gemm_q41/llama.esimd.gemv-slim.q41.l3.cpp
    }
    else
    {
        return runSlimGemmQ41_L3(q, inputs, weights, scales, zps, outputs, batch, input_len, output_len, input_precision, output_precision, shuffleTt);
    }
}


bool runGemv_Q41Weights_L5(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt)
{
    return runSlimGemmQ41_L5(q, inputs, weights, scales, zps, outputs, batch, input_len, output_len, input_precision, output_precision, shuffleTt);
}