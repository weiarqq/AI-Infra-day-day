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

#define GGML_TYPE_I32 26
#define GGML_TYPE_I64 27

extern "C" void __declspec(dllexport) StoreCacheFp16_xve(void* stream, fp16* cache, const void* inputs, const void* indexes, uint32_t token_len, uint32_t head_num, uint32_t head_dim, int input_precision, int index_precision);

template <typename IT, typename DT>
void StoreCacheFp16_xve_impl(void* stream, fp16* cache_data, const IT* inputs, const DT* indexes, uint32_t token_len, uint32_t head_num, uint32_t head_dim)
{
    const uint32_t row_size = head_num * head_dim;
    const uint32_t cacheline_size = row_size * sizeof(fp16);

    uint32_t threadx = row_size / 128;
    uint32_t thready = token_len;

    sycl::range<2> GlobalRange(threadx, thready);
    sycl::range<2> LocalRange(1, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);

    sycl::queue* q = (sycl::queue*)stream;

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            uint32_t h = ndi.get_global_id(0);
            uint32_t j = ndi.get_global_id(1);

            uint32_t input_offset = j * row_size + h * 128;
            uint32_t index = (uint32_t)indexes[j];
            uint32_t output_offset = index * cacheline_size + h * 128 * sizeof(fp16);

            simd<fp16, 128> inputData = block_load<IT, 128>(inputs + input_offset);
            block_store<fp16, 128>((fp16*)((uint8_t*)cache_data + output_offset), inputData);
        });
    });
}

void StoreCacheFp16_xve(void* stream, fp16* cache, const void* inputs, const void* indexes, uint32_t token_len, uint32_t head_num, uint32_t head_dim, int input_precision, int index_precision)
{
    if ((head_num * head_dim) % 128 != 0) {
        return;
    }

    if (input_precision == 0 && index_precision == GGML_TYPE_I64) {
        StoreCacheFp16_xve_impl(stream, cache, (float*)inputs, (int64_t*)indexes, token_len, head_num, head_dim);
    } else if (input_precision == 0 && index_precision == GGML_TYPE_I32) {
        StoreCacheFp16_xve_impl(stream, cache, (float*)inputs, (int32_t*)indexes, token_len, head_num, head_dim);
    } else if (input_precision == 1 && index_precision == GGML_TYPE_I64) {
        StoreCacheFp16_xve_impl(stream, cache, (fp16*)inputs, (int64_t*)indexes, token_len, head_num, head_dim);
    } else if (input_precision == 1 && index_precision == GGML_TYPE_I32) {
        StoreCacheFp16_xve_impl(stream, cache, (fp16*)inputs, (int32_t*)indexes, token_len, head_num, head_dim);
    }
}