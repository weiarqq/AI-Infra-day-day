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

#include <stdint.h>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using namespace sycl::ext::intel::esimd;

typedef sycl::half fp16;

#define FP32_MIN (-3.4e+38F)
#define GGML_TYPE_I32 26
#define GGML_TYPE_I64 27

extern "C" void __declspec(dllexport) StoreCacheQ80_xve(void* stream, uint8_t* cache, const void* inputs, const void* indexes, uint32_t token_len, uint32_t head_num, uint32_t head_dim, int input_precision, int index_precision);

template <typename IT, typename DT>
void StoreCacheQ80_xve_impl(void* stream, uint8_t* cache_data, const IT* inputs, const DT* indexes, uint32_t token_len, uint32_t head_num, uint32_t head_dim)
{
    uint32_t cacheline_size = head_dim * head_num / 32 * 34;

    uint32_t threadx = head_dim * head_num / 128;
    uint32_t thready = token_len;

    sycl::range<2> GlobalRange(threadx, thready);
    sycl::range<2> LocalRange(1, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    sycl::queue* q = (sycl::queue*)stream;

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            uint32_t h = ndi.get_global_id(0);
            uint32_t j = ndi.get_global_id(1);

            uint32_t input_offset = j * head_num * head_dim + h * 128;
            uint32_t index = (uint32_t)indexes[j];
            uint32_t output_quant_offset = index * cacheline_size + h * 128;
            uint32_t output_scale_offset = index * cacheline_size + head_num * head_dim + h * 4 * sizeof(fp16); // offset in bytes

            simd<float, 128> inputData = block_load<IT, 128>(inputs + input_offset);
            simd<float, 128> absData = abs<float, 128>(inputData);
            simd<int8_t, 128> quantData;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                float max_value = hmax<float, float, 32>(absData.select<32, 1>(k * 32));
                float d = max_value / 127.0f;
                float id = (d == 0.0f) ? 0.0f : 1.0f / d;

                *(fp16*)(cache_data + output_scale_offset + k * sizeof(fp16)) = (fp16)d;

                simd<float, 32> temp = inputData.select<32, 1>(k * 32) * id;

                quantData.select<32, 1>(k * 32) = rnde<float, 32>(temp);
            }

            block_store<int8_t, 128>((int8_t*)cache_data + output_quant_offset, quantData);
        });
    });
}

void StoreCacheQ80_xve(void* stream, uint8_t* cache, const void* inputs, const void* indexes, uint32_t token_len, uint32_t head_num, uint32_t head_dim, int input_precision, int index_precision)
{
    if (input_precision == 0 && index_precision == GGML_TYPE_I64) {
        StoreCacheQ80_xve_impl(stream, cache, (float*)inputs, (int64_t*)indexes, token_len, head_num, head_dim);
    } else if (input_precision == 0 && index_precision == GGML_TYPE_I32) {
        StoreCacheQ80_xve_impl(stream, cache, (float*)inputs, (int32_t*)indexes, token_len, head_num, head_dim);
    } else if (input_precision == 1 && index_precision == GGML_TYPE_I64) {
        StoreCacheQ80_xve_impl(stream, cache, (fp16*)inputs, (int64_t*)indexes, token_len, head_num, head_dim);
    } else if (input_precision == 1 && index_precision == GGML_TYPE_I32) {
        StoreCacheQ80_xve_impl(stream, cache, (fp16*)inputs, (int32_t*)indexes, token_len, head_num, head_dim);
    }
}