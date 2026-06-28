
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

void shuffle_Q40Weights_group128_L1(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        sycl::half d; // delta
        uint8_t qs[128 / 2]; // nibbles / quants
    } block_q4_0;
    block_q4_0* t = (block_q4_0*)input;
    char* p = (char*)output;
    sycl::half* h = (sycl::half*)(p + input_len * output_len / 2);

    for (int i = 0; i < input_len * output_len / 128; i++) {
        memcpy(p, t[i].qs, 128 / 2);
        p += (128 / 2);
        *h = t[i].d;
        h++;
    }
}