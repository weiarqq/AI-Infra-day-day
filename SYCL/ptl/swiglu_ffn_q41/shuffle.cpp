






void shuffle_Q41Weights_group32_L3(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        union {
            struct {
                sycl::half d; // delta
                sycl::half m; // min
            };
            uint32_t dm;
        };
        uint8_t qs[32 / 2]; // nibbles / quants for 32 elements
    } block_q4_1_32;

    block_q4_1_32* t = (block_q4_1_32*)input;
    uint8_t* p = (uint8_t *)output;
    sycl::half* h = (sycl::half*)(p + input_len * output_len / 2);
    sycl::half* z = (sycl::half*)(p + input_len * output_len / 2 + input_len * output_len / 32 * sizeof(sycl::half));

    for (int i = 0; i < input_len * output_len / 32; i++)
    {
        memcpy(p, t[i].qs, 32 / 2);
        p += (32 / 2);
    }

    uint32_t idx = 0;
    for (int j = 0; j < output_len/32; j++)
    {
        for (int k = 0; k < input_len/32; k ++)
        {
            for (int i = 0; i < 32; i++)
            {
                h[idx] = t[(j * 32 + i) * (input_len/32) + k].d;
                z[idx] = t[(j * 32 + i) * (input_len/32) + k].m;
                ++idx;
            }
        }

    }
}




void shuffle_swiglu_Q41Weights_group32_L3(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        union {
            struct {
                sycl::half d; // delta
                sycl::half m; // min
            };
            uint32_t dm;
        };
        uint8_t qs[32 / 2]; // nibbles / quants for 32 elements
    } block_q4_1_32;

    uint8_t *temp = new uint8_t[(input_len / 32) * output_len * sizeof(block_q4_1_32)];

    uint32_t copy_block_size = 16 * (input_len/32) * sizeof(block_q4_1_32);
    uint32_t gate_offset = (output_len / 2) * (input_len / 32) * sizeof(block_q4_1_32);

    uint32_t num_block = output_len / 2 / 16;

    for (int j = 0; j < num_block; j++)
    {
        memcpy(temp + j * 2 * copy_block_size, input + j * copy_block_size, copy_block_size);
        memcpy(temp + (j * 2 + 1 ) * copy_block_size, input + gate_offset + j * copy_block_size, copy_block_size);
    }

    shuffle_Q41Weights_group32_L3(temp, output, input_len, output_len);

    delete[] temp;
}



void shuffle_Q41Weights_group32_L5(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        union {
            struct {
                sycl::half d; // delta
                sycl::half m; // min
            };
            uint32_t dm;
        };
        uint8_t qs[32 / 2]; // nibbles / quants for 32 elements
    } block_q4_1_32;

    block_q4_1_32* t = (block_q4_1_32*)input;
    uint8_t* p = (uint8_t *)output;
    sycl::half* h = (sycl::half*)(p + input_len * output_len / 2);
    sycl::half* z = (sycl::half*)(p + input_len * output_len / 2 + input_len * output_len / 32 * sizeof(sycl::half));

    for (int j = 0; j < output_len / 16; j ++)
    {
        for (int jj = 0; jj < 16; jj++)
        {
            for (int k = 0; k < input_len / 32; k ++)
            {
                uint32_t idx = (j * 16 + jj) * input_len /32 + k;
                for (int kk = 0; kk < 8; kk ++)
                {
                    p[j * 16 * input_len / 2 + k * 16 * 16 + kk * 16 * 2 + jj * 2] = t[idx].qs[kk * 2];
                    p[j * 16 * input_len / 2 + k * 16 * 16 + kk * 16 * 2 + jj * 2 + 1] = t[idx].qs[kk * 2 + 1];
                }

            }
        }
    }

    for (int j = 0; j < output_len; j ++)
    {
        for (int k = 0; k < input_len / 32; k ++)
        {
            h[k * output_len + j] = t[j * input_len/32 + k].d;
            z[k * output_len + j] = t[j * input_len/32 + k].m;
        }
    }
}



void shuffle_swiglu_Q41Weights_group32_L5(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        union {
            struct {
                sycl::half d; // delta
                sycl::half m; // min
            };
            uint32_t dm;
        };
        uint8_t qs[32 / 2]; // nibbles / quants for 32 elements
    } block_q4_1_32;

    uint8_t *temp = new uint8_t[(input_len / 32) * output_len * sizeof(block_q4_1_32)];

    uint32_t copy_block_size = 16 * (input_len/32) * sizeof(block_q4_1_32);
    uint32_t gate_offset = (output_len / 2) * (input_len / 32) * sizeof(block_q4_1_32);

    uint32_t num_block = output_len / 2 / 16;

    for (int j = 0; j < num_block; j++)
    {
        memcpy(temp + j * 2 * copy_block_size, input + j * copy_block_size, copy_block_size);
        memcpy(temp + (j * 2 + 1 ) * copy_block_size, input + gate_offset + j * copy_block_size, copy_block_size);
    }

    shuffle_Q41Weights_group32_L5(temp, output, input_len, output_len);

    delete[] temp;
}