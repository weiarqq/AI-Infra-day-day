void shuffle_Q41Weights_group32_L1(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        union {
            struct {
                sycl::half d; // delta
                sycl::half m; // min
            };
            uint32_t dm;
        };
        uint8_t qs[32 / 2]; // nibbles / quants
    } block_q4_1;

    block_q4_1* t = (block_q4_1*)input;
    uint8_t* p = (uint8_t*)output;
    sycl::half* h = (sycl::half*)(p + input_len * output_len / 2);
    sycl::half* z = (sycl::half*)(p + input_len * output_len / 2 + input_len * output_len / 32 * sizeof(sycl::half));

    for (int i = 0; i < input_len * output_len / 32; i++) {
        memcpy(p, t[i].qs, 32 / 2);

        p += (32 / 2);

        h[i] = t[i].d;
        z[i] = t[i].m;
    }
}