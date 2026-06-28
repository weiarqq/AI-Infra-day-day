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

#include <windows.h>

#include <map>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;
using fp16 = sycl::half;
#define DEVICE_MEM_ALIGNMENT 4096;

#define FP32_MAX (1.7e+38)
#define FP32_MIN (-1.7e+38)

using namespace std;
using namespace sycl::ext::intel::esimd;

extern "C" bool runSlimGemmQ41_L4_XMX(sycl::queue * q,
                                      uint8_t *     inputs,
                                      uint8_t *     weights,
                                      uint8_t *     scales,
                                      uint8_t *     zps,
                                      uint8_t *     outputs,
                                      unsigned      batch,
                                      unsigned      input_len,
                                      unsigned      output_len,
                                      unsigned      input_precision,
                                      unsigned      output_precision,
                                      uint8_t *     shuffleTt);

template <uint32_t batchnum>
ESIMD_INLINE void slimGemmGroup32Block128BatchN_L4_XMX(float *      inputs,
                                                       uint8_t *    weights,
                                                       fp16 *       scales,
                                                       fp16 *       zps,
                                                       float *      outputs,
                                                       uint32_t     input_len,
                                                       uint32_t     output_len,
                                                       nd_item<1> & ndi) {
    slm_init(sizeof(float) * 64 * batchnum * 16);  // 声明共享内存

    constexpr uint32_t outputPerGroup      = 16;
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

    uint32_t h          = ndi.get_group(0);
    uint32_t localId    = ndi.get_local_id(0);
    uint32_t localRange = ndi.get_local_range(0);

    uint32_t           inputOffset  = localId * 128;
    uint32_t           weightOffset = h * outputPerGroup * input_len / 2 + localId * 64;
    //权重偏移 权重是int4 输入权重类型是int8 所以除以2， localId处理128个int4 即 64个int8
    simd<uint32_t, 16> scatterWeightsOffset(baseOffsetInc16);
    scatterWeightsOffset = scatterWeightsOffset * input_len / 2;
    scatterWeightsOffset += weightOffset;
    // scatterWeightsOffset 相当于滑动窗口，weightoffset相当于该滑动窗口的起点
    //simd<float, batchnum * 256> readBuffer;
    simd<fp16, batchnum * 128> inputData;

    simd<float, batchnum * 8> accResult0 = 0.0;
    simd<float, batchnum * 8> accResult1 = 0.0;

#pragma unroll
    for (int j = 0; j < batchnum; j++) {
        inputData.template select<128, 1>(j * 128) = block_load<float, 128>(inputs + inputOffset);
        inputOffset += input_len;
    }
    //inputData = readBuffer;

    simd<uint8_t, 1024> weightsBuffer;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        //weightsBuffer.select<256, 1>(j*256) = block_load<uint8_t, 256>(weights + weightOffset);
        weightsBuffer.template bit_cast_view<uint32_t>().select<64, 1>(j * 64) =
            __ESIMD_ENS::lsc_gather<uint32_t,
                                    4,
                                    __ESIMD_ENS::lsc_data_size::u32,
                                    __ESIMD_ENS::cache_hint::cached,
                                    __ESIMD_ENS::cache_hint::cached,
                                    16,
                                    uint32_t>((uint32_t *) weights, scatterWeightsOffset);
        scatterWeightsOffset += 16;
    }

    simd<fp16, 64> scaleData;
    simd<fp16, 64> zpsData;

    uint32_t scaleOffset = h * 16 * input_len / 32 + localId * 16 * 4;
    // 32个元素一个block -> input_len/32
    // work item 负责 128个k维 128/32个元素 =4个scale output channel 方向 负责16个，所以 16*4
#pragma unroll
    for (int j = 0; j < 4; j++) {
        scaleData.select<16, 1>(j * 16) = block_load<fp16, 16>(scales + scaleOffset);
        zpsData.select<16, 1>(j * 16)   = block_load<fp16, 16>(zps + scaleOffset);
        scaleOffset += 16;
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
        simd<uint8_t, 128> shuffledWeights0;
        simd<uint8_t, 128> shuffledWeights1;
#pragma unroll
        for (int k = 0; k < 4; k++) {
            shuffledWeights0.template bit_cast_view<uint16_t>().select<8, 1>(16 * k) =
                weightsBuffer.template bit_cast_view<uint16_t>().select<8, 2>(j * 128 + 32 * k);
            shuffledWeights0.template bit_cast_view<uint16_t>().select<8, 1>(16 * k + 8) =
                weightsBuffer.template bit_cast_view<uint16_t>().select<8, 2>(j * 128 + 32 * k + 1);
            shuffledWeights1.template bit_cast_view<uint16_t>().select<8, 1>(16 * k) =
                weightsBuffer.template bit_cast_view<uint16_t>().select<8, 2>(j * 128 + 32 * k + 16);
            shuffledWeights1.template bit_cast_view<uint16_t>().select<8, 1>(16 * k + 8) =
                weightsBuffer.template bit_cast_view<uint16_t>().select<8, 2>(j * 128 + 32 * k + 17);
        }
        simd<fp16, 128> weightsDataLow0 = shuffledWeights0 & 0x0f;
        simd<fp16, 128> weightsDataLow1 = shuffledWeights1 & 0x0f;

#pragma unroll
        for (int k = 0; k < 8; k++) {
            weightsDataLow0.select<8, 2>(16 * k) =
                weightsDataLow0.select<8, 2>(16 * k) * scaleData.select<8, 1>(j * 16) + zpsData.select<8, 1>(j * 16);
            weightsDataLow0.select<8, 2>(16 * k + 1) =
                weightsDataLow0.select<8, 2>(16 * k + 1) * scaleData.select<8, 1>(j * 16) +
                zpsData.select<8, 1>(j * 16);
            weightsDataLow1.select<8, 2>(16 * k) =
                weightsDataLow1.select<8, 2>(16 * k) * scaleData.select<8, 1>(j * 16 + 8) +
                zpsData.select<8, 1>(j * 16 + 8);
            weightsDataLow1.select<8, 2>(16 * k + 1) =
                weightsDataLow1.select<8, 2>(16 * k + 1) * scaleData.select<8, 1>(j * 16 + 8) +
                zpsData.select<8, 1>(j * 16 + 8);
        }

        simd<fp16, batchnum * 16> inputDataLow;
#pragma unroll
        for (int k = 0; k < batchnum; k++) {
            inputDataLow.template select<16, 1>(k * 16) = inputData.template select<16, 1>(k * 128 + j * 32);
        }

        accResult0 = xmx::dpas<8, batchnum, float, float, fp16, fp16>(accResult0, weightsDataLow0, inputDataLow);
        accResult1 = xmx::dpas<8, batchnum, float, float, fp16, fp16>(accResult1, weightsDataLow1, inputDataLow);

        simd<fp16, 128> weightsDataHigh0 = shuffledWeights0 >> 4;
        simd<fp16, 128> weightsDataHigh1 = shuffledWeights1 >> 4;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            weightsDataHigh0.select<8, 2>(16 * k) =
                weightsDataHigh0.select<8, 2>(16 * k) * scaleData.select<8, 1>(j * 16) + zpsData.select<8, 1>(j * 16);
            weightsDataHigh0.select<8, 2>(16 * k + 1) =
                weightsDataHigh0.select<8, 2>(16 * k + 1) * scaleData.select<8, 1>(j * 16) +
                zpsData.select<8, 1>(j * 16);
            weightsDataHigh1.select<8, 2>(16 * k) =
                weightsDataHigh1.select<8, 2>(16 * k) * scaleData.select<8, 1>(j * 16 + 8) +
                zpsData.select<8, 1>(j * 16 + 8);
            weightsDataHigh1.select<8, 2>(16 * k + 1) =
                weightsDataHigh1.select<8, 2>(16 * k + 1) * scaleData.select<8, 1>(j * 16 + 8) +
                zpsData.select<8, 1>(j * 16 + 8);
        }

        simd<fp16, batchnum * 16> inputDataHigh;
#pragma unroll
        for (int k = 0; k < batchnum; k++) {
            inputDataHigh.template select<16, 1>(k * 16) = inputData.template select<16, 1>(k * 128 + j * 32 + 16);
        }

        accResult0 = xmx::dpas<8, batchnum, float, float, fp16, fp16>(accResult0, weightsDataHigh0, inputDataHigh);
        accResult1 = xmx::dpas<8, batchnum, float, float, fp16, fp16>(accResult1, weightsDataHigh1, inputDataHigh);
    }

    slm_block_store<float, batchnum * 8>(sizeof(float) * localId * batchnum * 16, accResult0);
    slm_block_store<float, batchnum * 8>(sizeof(float) * localId * batchnum * 16 + batchnum * 8 * sizeof(float),
                                         accResult1);

    barrier();

    if (localId < batchnum) {
        simd<float, 16> outputData      = 0.0;
        uint32_t        slmOutputOffset = localId * 8 * sizeof(float);
#pragma unroll
        for (int j = 0; j < localRange; j++) {
            outputData.select<8, 1>(0) += slm_block_load<float, 8>(slmOutputOffset);
            slmOutputOffset += batchnum * 8 * sizeof(float);
            outputData.select<8, 1>(8) += slm_block_load<float, 8>(slmOutputOffset);
            slmOutputOffset += batchnum * 8 * sizeof(float);
        }

        block_store<float, 16>(outputs + localId * output_len + h * outputPerGroup, outputData);
    }
}

bool runSlimGemmQ41_L4_XMX(sycl::queue * q,
                           uint8_t *     inputs,
                           uint8_t *     weights,
                           uint8_t *     scales,
                           uint8_t *     zps,
                           uint8_t *     outputs,
                           unsigned      batch,
                           unsigned      input_len,
                           unsigned      output_len,
                           unsigned      input_precision,
                           unsigned      output_precision,
                           uint8_t *     shuffleTt) {
    uint32_t global_thread = (output_len + 15) / 16;
    uint32_t local_thread  = (input_len + 127) / 128;

    sycl::range<1>    GlobalRange(global_thread * local_thread);
    sycl::range<1>    LocalRange(local_thread);
    sycl::nd_range<1> Range_128(GlobalRange, LocalRange);

    sycl::event e;

    switch (batch) {
        case 1:
            e = q->submit([&](handler & cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                    slimGemmGroup32Block128BatchN_L4_XMX<1>((float *) inputs,
                                                            (uint8_t *) weights,
                                                            (fp16 *) scales,
                                                            (fp16 *) zps,
                                                            (float *) outputs,
                                                            input_len,
                                                            output_len,
                                                            ndi);
                });
            });
            break;
        case 2:
            e = q->submit([&](handler & cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                    slimGemmGroup32Block128BatchN_L4_XMX<2>((float *) inputs,
                                                            (uint8_t *) weights,
                                                            (fp16 *) scales,
                                                            (fp16 *) zps,
                                                            (float *) outputs,
                                                            input_len,
                                                            output_len,
                                                            ndi);
                });
            });
            break;
        case 3:
            e = q->submit([&](handler & cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                    slimGemmGroup32Block128BatchN_L4_XMX<3>((float *) inputs,
                                                            (uint8_t *) weights,
                                                            (fp16 *) scales,
                                                            (fp16 *) zps,
                                                            (float *) outputs,
                                                            input_len,
                                                            output_len,
                                                            ndi);
                });
            });
            break;
        case 4:
            e = q->submit([&](handler & cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                    slimGemmGroup32Block128BatchN_L4_XMX<4>((float *) inputs,
                                                            (uint8_t *) weights,
                                                            (fp16 *) scales,
                                                            (fp16 *) zps,
                                                            (float *) outputs,
                                                            input_len,
                                                            output_len,
                                                            ndi);
                });
            });
            break;
        case 5:
            e = q->submit([&](handler & cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                    slimGemmGroup32Block128BatchN_L4_XMX<5>((float *) inputs,
                                                            (uint8_t *) weights,
                                                            (fp16 *) scales,
                                                            (fp16 *) zps,
                                                            (float *) outputs,
                                                            input_len,
                                                            output_len,
                                                            ndi);
                });
            });
            break;
        case 6:
            e = q->submit([&](handler & cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                    slimGemmGroup32Block128BatchN_L4_XMX<6>((float *) inputs,
                                                            (uint8_t *) weights,
                                                            (fp16 *) scales,
                                                            (fp16 *) zps,
                                                            (float *) outputs,
                                                            input_len,
                                                            output_len,
                                                            ndi);
                });
            });
            break;
        case 7:
            e = q->submit([&](handler & cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                    slimGemmGroup32Block128BatchN_L4_XMX<7>((float *) inputs,
                                                            (uint8_t *) weights,
                                                            (fp16 *) scales,
                                                            (fp16 *) zps,
                                                            (float *) outputs,
                                                            input_len,
                                                            output_len,
                                                            ndi);
                });
            });
            break;
        case 8:
            e = q->submit([&](handler & cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
                    slimGemmGroup32Block128BatchN_L4_XMX<8>((float *) inputs,
                                                            (uint8_t *) weights,
                                                            (fp16 *) scales,
                                                            (fp16 *) zps,
                                                            (float *) outputs,
                                                            input_len,
                                                            output_len,
                                                            ndi);
                });
            });
            break;
        default:
            printf("Error! Q41 GEMV only supports k <= 8 right now.\n");
    }
}
