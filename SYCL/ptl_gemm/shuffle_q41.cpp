

void shuffle_Q41Weights_group32_L3(uint8_t * input, uint8_t * output, uint32_t input_len, uint32_t output_len) {
    typedef struct {
        union {
            struct {
                sycl::half d;  // delta
                sycl::half m;  // min
            };

            uint32_t dm;
        };

        uint8_t qs[32 / 2];  // nibbles / quants for 32 elements
    } block_q4_1_32;

    block_q4_1_32 * t = (block_q4_1_32 *) input;
    uint8_t *       p = (uint8_t *) output;
    sycl::half *    h = (sycl::half *) (p + input_len * output_len / 2);
    sycl::half * z = (sycl::half *) (p + input_len * output_len / 2 + input_len * output_len / 32 * sizeof(sycl::half));

    for (int i = 0; i < input_len * output_len / 32; i++) {
        memcpy(p, t[i].qs, 32 / 2);
        p += (32 / 2);
    }

    uint32_t idx = 0;
    for (int j = 0; j < output_len / 32; j++) {
        for (int k = 0; k < input_len / 32; k++) {
            for (int i = 0; i < 32; i++) {
                h[idx] = t[(j * 32 + i) * (input_len / 32) + k].d;
                z[idx] = t[(j * 32 + i) * (input_len / 32) + k].m;
                ++idx;
            }
        }
    }
}
