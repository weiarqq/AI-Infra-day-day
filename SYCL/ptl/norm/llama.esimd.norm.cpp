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

extern "C" bool runNormFusion(sycl::queue *q, uint8_t* inputs, uint8_t *outputs, float *scale, float *bias, uint32_t token_len, uint32_t input_len, uint32_t input_stride, uint32_t norm_type, uint32_t input_precision, uint32_t output_precision);


template<typename inT, typename outT>
ESIMD_INLINE void runNormalNorm_blk128_esimd(uint8_t* inputs, uint8_t* outputs, float *scale, float *bias, uint32_t token_len, uint32_t input_len, uint32_t input_stride, bool has_scale, bool has_bias, nd_item<1>& ndi)
{
    slm_init(sizeof(float) * 64 * 2);
    constexpr uint32_t slmOffsetMean = 0;
    constexpr uint32_t slmOffsetVar = sizeof(float)*64;

    uint32_t localRange = ndi.get_local_range(0);
    uint32_t h = ndi.get_group(0);
    uint32_t hh = ndi.get_local_id(0);

    simd<float, 128> zeros = 0.0;
    if (hh == 0)
    {
        slm_block_store<float, 128>(0, zeros);
    }
    barrier();

    uint32_t offsetInRow = hh * 64;
    simd<float, 128> input_data;
    input_data.select<64, 1>(0) = block_load<inT, 64>((inT *)inputs + h * input_stride + offsetInRow);
    if (offsetInRow + 64 * localRange < input_len)
    {
        input_data.select<64, 1>(64) = block_load<inT, 64>((inT *)inputs + h * input_stride + offsetInRow + 64 * localRange);
    }
    else
    {
        input_data.select<64, 1>(64) = 0.0;
    }

    simd<float, 128> scale_data = 1.0;
    if (has_scale)
    {
        scale_data.select<64, 1>(0) = block_load<float, 64>(scale + offsetInRow);
        if (offsetInRow + 64 * localRange < input_len)
        {
            scale_data.select<64, 1>(64) = block_load<float, 64>(scale + offsetInRow + 64 * localRange);
        }
    }

    simd<float, 128> bias_data = 0.0;
    if (has_bias)
    {
        bias_data.select<64, 1>(0) = block_load<float, 64>(bias + offsetInRow);
        if (offsetInRow + 64 * localRange < input_len)
        {
            bias_data.select<64, 1>(64) = block_load<float, 64>(bias + offsetInRow + 64 * localRange);
        }
    }

    float mean = sycl::ext::intel::esimd::detail::sum<float, float, 128>(input_data);
    simd<float, 128> squares = input_data * input_data;
    float var = sycl::ext::intel::esimd::detail::sum<float, float, 128>(squares);

    slm_block_store<float, 1>(slmOffsetMean + hh * sizeof(float), mean);
    slm_block_store<float, 1>(slmOffsetVar + hh * sizeof(float), var);

    barrier();
    simd<float, 64> acc_vec = slm_block_load<float, 64>(slmOffsetMean);
    mean = sycl::ext::intel::esimd::detail::sum<float, float, 64>(acc_vec);
    mean = mean / input_len;
    acc_vec = slm_block_load<float, 64>(slmOffsetVar);
    var = sycl::ext::intel::esimd::detail::sum<float, float, 64>(acc_vec);
    var = var / input_len;
    var = var - mean * mean;

    float inv = std::sqrt(var + 1e-6);
    inv = 1.0/inv;
    simd<float, 128> output_data = (input_data - mean) * inv;

    if (has_scale)
    {
        output_data = output_data * scale_data;
    }
    if (has_bias)
    {
        output_data = output_data + bias_data;
    }

    block_store<outT, 64>((outT*)outputs + h * input_stride + offsetInRow, output_data.select<64, 1>(0));

    if (offsetInRow + 64 * localRange < input_len)
    {
        block_store<outT, 64>((outT*)outputs + h * input_stride + offsetInRow + 64 * localRange, output_data.select<64, 1>(64));
    }
}

template<typename inT, typename outT>
ESIMD_INLINE void runRMSNorm_blk128_esimd(uint8_t* inputs, uint8_t* outputs, float *scale, float *bias, uint32_t token_len, uint32_t input_len, uint32_t input_stride, bool has_scale, bool has_bias, nd_item<1>& ndi)
{
    slm_init(sizeof(float) * 64);

    uint32_t localRange = ndi.get_local_range(0);
    uint32_t h = ndi.get_group(0);
    uint32_t hh = ndi.get_local_id(0);

    simd<float, 64> zeros = 0.0;
    if (hh == 0)
    {
        slm_block_store<float, 64>(0, zeros);
    }
    barrier();

    uint32_t offsetInRow = hh * 64;
    simd<float, 128> input_data;
    input_data.select<64, 1>(0) = block_load<inT, 64>((inT *)inputs + h * input_stride + offsetInRow);
    if (offsetInRow + 64 * localRange < input_len)
    {
        input_data.select<64, 1>(64) = block_load<inT, 64>((inT *)inputs + h * input_stride + offsetInRow + 64 * localRange);
    }
    else
    {
        input_data.select<64, 1>(64) = 0.0;
    }

    simd<float, 128> scale_data = 1.0;
    if (has_scale)
    {
        scale_data.select<64, 1>(0) = block_load<float, 64>(scale + offsetInRow);
        if (offsetInRow + 64 * localRange < input_len)
        {
            scale_data.select<64, 1>(64) = block_load<float, 64>(scale + offsetInRow + 64 * localRange);
        }
    }

    simd<float, 128> bias_data = 0.0;
    if (has_bias)
    {
        bias_data.select<64, 1>(0) = block_load<float, 64>(bias + offsetInRow);
        if (offsetInRow + 64 * localRange < input_len)
        {
            bias_data.select<64, 1>(64) = block_load<float, 64>(bias + offsetInRow + 64 * localRange);
        }
    }

    simd<float, 128> squares = input_data * input_data;
    float var = sycl::ext::intel::esimd::detail::sum<float, float, 128>(squares);

    slm_block_store<float, 1>(hh * sizeof(float), var);

    barrier();
    simd<float, 64> acc_vec = slm_block_load<float, 64>(0);
    var = sycl::ext::intel::esimd::detail::sum<float, float, 64>(acc_vec);
    var = var / input_len;

    float inv = std::sqrt(var + 1e-6);
    inv = 1.0/inv;
    simd<float, 128> output_data = input_data * inv;

    if (has_scale)
    {
        output_data = output_data * scale_data;
    }
    if (has_bias)
    {
        output_data = output_data + bias_data;
    }

    block_store<outT, 64>((outT*)outputs + h * input_stride + offsetInRow, output_data.select<64, 1>(0));

    if (offsetInRow + 64 * localRange < input_len)
    {
        block_store<outT, 64>((outT*)outputs + h * input_stride + offsetInRow + 64 * localRange, output_data.select<64, 1>(64));
    }
}

template<typename inT, typename outT>
static inline sycl::event runNormFusion_impl(sycl::queue *q, uint8_t* inputs, uint8_t *outputs, float *scale, float *bias, uint32_t token_len, uint32_t input_len, uint32_t input_stride, uint32_t norm_type)
{
    sycl::event e;
    int groups = token_len;
    int localRange = (input_len + 127)/128;

    sycl::range<1> GlobalRange(groups * localRange);
    sycl::range<1> LocalRange(localRange);
    sycl::nd_range<1> RangeV2(GlobalRange, localRange);

    bool has_scale = (scale != nullptr);
    bool has_bias = (bias != nullptr);

    if (norm_type == 0)
    {
        e = q->submit([&](handler& cgh) {
            cgh.parallel_for(RangeV2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    runNormalNorm_blk128_esimd<inT, outT>(inputs, outputs, scale, bias, token_len, input_len, input_stride, has_scale, has_bias, ndi);
                });
            });
    }
    else
    {
      e = q->submit([&](handler& cgh) {
          cgh.parallel_for(RangeV2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                  runRMSNorm_blk128_esimd<inT, outT>(inputs, outputs, scale, bias, token_len, input_len, input_stride, has_scale, has_bias, ndi);
              });
          });

    }
    return e;
}

bool runNormFusion(sycl::queue *q, uint8_t* inputs, uint8_t *outputs, float *scale, float *bias, uint32_t token_len, uint32_t input_len, uint32_t input_stride, uint32_t norm_type, uint32_t input_precision, uint32_t output_precision)
{
    if (input_len % 64 > 0)
    {
        printf("Error: runNormFusion only supports 64x dimension, current dimension is %d\n", input_len);
        return false;
    }
    sycl::event e;
    try{
        if (input_precision == 0 && output_precision == 0)
        {
            e = runNormFusion_impl<float, float>(q, inputs, outputs, scale, bias, token_len, input_len, input_stride, norm_type);
        }
        else if (input_precision == 0 && output_precision == 1)
        {
            e = runNormFusion_impl<float, fp16>(q, inputs, outputs, scale, bias, token_len, input_len, input_stride, norm_type);
        }
        if (input_precision == 1 && output_precision == 0)
        {
            e = runNormFusion_impl<fp16, float>(q, inputs, outputs, scale, bias, token_len, input_len, input_stride, norm_type);
        }
        if (input_precision == 1 && output_precision == 1)
        {
            e = runNormFusion_impl<fp16, fp16>(q, inputs, outputs, scale, bias, token_len, input_len, input_stride, norm_type);
        }

    } catch (sycl::exception const& e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }
}
