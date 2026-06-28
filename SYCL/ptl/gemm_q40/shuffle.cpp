
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

void shuffle_Q40Weights_group128_L2(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        sycl::half d; // delta
        uint8_t qs[128 / 2]; // nibbles / quants
    } block_q4_0;
    block_q4_0* t = (block_q4_0*)input;
    char* p = (char*)output;
    sycl::half* h = (sycl::half*)(p + input_len * output_len / 2);

    for (int i = 0; i < input_len * output_len / 128; i++) {
        // memcpy(p, t[i].qs, QK4_0 / 2);
        for (int j = 0; j < 128 / 2; j += 16) {
            int8_t shuffle[32];
            for (int k = 0; k < 16; k++) {
                uint8_t hi = t[i].qs[j + k] >> 4;
                uint8_t lo = t[i].qs[j + k] & 0x0f;
                shuffle[k] = lo - 8;
                shuffle[k + 16] = hi - 8;
            }

            for (int k = 0; k < 16; k++) {
                p[j + k] = ((shuffle[2 * k + 1] & 0x0f) << 4) | (shuffle[2 * k] & 0x0f);
            }
        }

        p += (128 / 2);

        int vv = i / (input_len / 128);
        int hh = i % (input_len / 128);
        h[hh * output_len + vv] = t[i].d;
    }
}