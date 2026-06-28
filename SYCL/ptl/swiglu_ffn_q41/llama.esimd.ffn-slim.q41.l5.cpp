// Copyright (C) 2024 - 2026 Intel Corporation
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you ("License"). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit
// this software or the related documents without Intel's prior written
// permission.

// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly stated
// in the License.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <map>

#include <windows.h>

using namespace sycl;
using fp16 = sycl::half;
#define DEVICE_MEM_ALIGNMENT 4096;


using namespace std;
using namespace sycl::ext::intel::esimd;

#define GROUP_SIZE 128

extern "C" bool runSwigluFFnVec_Q41_L5(sycl::queue* q, uint8_t *up, uint8_t *down, uint8_t *input, uint8_t *output, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, uint8_t *scratch_buffer);


// import from other file
extern "C" bool runSlimGemmQ41_L5(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt);

bool runSwiglu_L5(queue & q, float *input, float *output, uint32_t token_len, uint32_t hidden_len)
{
    int groups = (hidden_len + 127)/128;

    sycl::range<1> GlobalRange(groups);
    sycl::range<1> LocalRange(1);
    sycl::nd_range<1> RangeCommon(GlobalRange, LocalRange);

    sycl::event e;
    try {
        e = q.submit([&](handler& cgh) {
            cgh.parallel_for(RangeCommon, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                //int hh = ndi.get_local_id(0);
                int h = ndi.get_group(0);
                //int range = ndi.get_local_range(0);

                simd<float, 256> data;
                simd<float, 128> gate;
                simd<float, 128> up;
                //int offset = hh * 128 + h * input_len;
                uint32_t input_offset = h * 256;
                uint32_t output_offset = h * 128;

                for (int lp = 0; lp < token_len; lp ++)
                {
                    data = block_load<float, 256>(input + input_offset);
                #pragma unroll
                    for (int j = 0; j < 8; j++)
                    {
                        gate.select<16, 1>(j * 16) = data.select<16, 1>(j * 32);
                        up.select<16, 1>(j * 16) = data.select<16, 1>(j * 32 + 16);
                    }

                    //simd<float, 128> temp = up;
                    simd<float, 128> temp = -1.0 * gate;
                    temp = pow<float, 128, float>(2.718f, temp);
                    temp = temp + 1.0;
                    temp = 1.0/temp;
                    temp = temp * gate;
                    temp = temp * up;
                    block_store<float, 128>(output + output_offset, temp);

                    input_offset += 2*hidden_len;
                    output_offset += hidden_len;
                }

                });
            });
    }
    catch (sycl::exception const& e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

}

bool runSwigluFFnVec_Q41_L5(sycl::queue* q, uint8_t *up, uint8_t *down, uint8_t *input, uint8_t *output, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, uint8_t *scratch_buffer)
{
    uint8_t *s_up = up + (input_len * hidden_len * 2)/2;
    uint8_t *z_up = s_up + (input_len * hidden_len * 2) / 32 * sizeof(fp16);
    uint8_t *s_down = down + (input_len * hidden_len)/2;
    uint8_t *z_down = s_down + (input_len * hidden_len) / 32 * sizeof(fp16);
    uint8_t *upResult = scratch_buffer;
    uint8_t *swigluResult = upResult + token_len * 2 * hidden_len * sizeof(float);
    runSlimGemmQ41_L5(q, input, up, s_up, z_up, upResult, token_len, input_len, hidden_len * 2, input_precision, 0,  nullptr);
    runSwiglu_L5(*q, (float *)upResult, (float *)swigluResult, token_len, hidden_len);
    runSlimGemmQ41_L5(q, swigluResult, down, s_down, z_down, output, token_len, hidden_len, input_len, 0, output_precision, nullptr);

    return true;
}