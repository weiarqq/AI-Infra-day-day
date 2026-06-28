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

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <map>

#include <windows.h>

using namespace sycl;
using fp16 = sycl::half;
#define DEVICE_MEM_ALIGNMENT 4096;


using namespace std;
using namespace sycl::ext::intel::esimd;


extern "C" bool runSlimGemmQ41_L5(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt);

template<uint32_t batchnum>
ESIMD_INLINE void slimGemmGroup32Block128BatchN_L5(float* inputs, uint8_t* weights, fp16* scales, fp16* zps, float* outputs, uint32_t input_len, uint32_t output_len, nd_item<1>& ndi)
{
    slm_init(sizeof(float) * 64 * batchnum * 16);

    constexpr uint32_t outputPerGroup = 16;

    uint32_t h = ndi.get_group(0);
    uint32_t localId = ndi.get_local_id(0);
    uint32_t localRange = ndi.get_local_range(0);

    uint32_t inputOffset = localId * 32;
    uint32_t weightOffset = h * outputPerGroup * input_len/2 + 256 * localId;
    simd<float, batchnum * 128> readData;
    simd<fp16, batchnum * 128> inputData;

    simd<float, batchnum * 16> accResult = 0.0;

#pragma unroll
    for (int j = 0; j < batchnum * 4; j++)
    {
        readData.template select<32, 1>(j * 32) = block_load<float, 32>(inputs + inputOffset);
        inputOffset += 32 * localRange;
    }
    inputData = readData / 4.0; // to handle granite overflow issue

    simd<uint8_t, 1024> weightsBuffer;
#pragma unroll
    for (int j = 0; j < 4; j ++)
    {
        weightsBuffer.select<256, 1>(j*256) = block_load<uint8_t, 256>(weights + weightOffset);
        weightOffset += 256 * localRange;
    }

    simd<fp16, 64> scaleData;
    simd<fp16, 64> zpsData;

    uint32_t scaleOffset = localId  * output_len + h * outputPerGroup;
#pragma unroll
    for (int j = 0; j < 4; j ++)
    {
        scaleData.select<16, 1>(j * 16) = block_load<fp16, 16>(scales + scaleOffset);
        zpsData.select<16, 1>(j * 16) = block_load<fp16, 16>(zps + scaleOffset);
        scaleOffset += localRange * output_len;
    }

    scaleData *= 4;
    zpsData *= 4;
    
#pragma unroll
    for(int j = 0; j < 4; j ++)
    {
        simd<fp16, 256> weightsDataLow = weightsBuffer.select<256, 1>(j * 256) & 0x0f;
        

#pragma unroll
        for (int k = 0; k < 8; k ++)
        {
            weightsDataLow.select<16, 2>(32 * k) = weightsDataLow.select<16, 2>(32 * k) * scaleData.select<16, 1>(j * 16) + zpsData.select<16, 1>(j * 16);
            weightsDataLow.select<16, 2>(32 * k + 1) = weightsDataLow.select<16, 2>(32 * k + 1) * scaleData.select<16, 1>(j * 16) + zpsData.select<16, 1>(j * 16);
        }

        simd<fp16, batchnum * 16> inputDataLow;
#pragma unroll
        for (int k = 0; k < batchnum; k++)
        {
            inputDataLow.template select<16, 1>(k * 16) = inputData.template select<16,  1>(k * 128 + j * 32);
        }

        accResult = xmx::dpas<8, batchnum, float, float, fp16, fp16>(accResult, weightsDataLow, inputDataLow);


        simd<fp16, 256> weightsDataHigh = weightsBuffer.select<256, 1>(j * 256) >> 4;
#pragma unroll
        for (int k = 0; k < 8; k ++)
        {
            weightsDataHigh.select<16, 2>(32 * k) = weightsDataHigh.select<16, 2>(32 * k) * scaleData.select<16, 1>(j * 16) + zpsData.select<16, 1>(j * 16);
            weightsDataHigh.select<16, 2>(32 * k + 1) = weightsDataHigh.select<16, 2>(32 * k + 1) * scaleData.select<16, 1>(j * 16) + zpsData.select<16, 1>(j * 16);
        }

        simd<fp16, batchnum * 16> inputDataHigh;
#pragma unroll
        for (int k = 0; k < batchnum; k++)
        {
            inputDataHigh.template select<16, 1>(k * 16) = inputData.template select<16,  1>(k * 128 + j * 32 + 16);
        }
        accResult = xmx::dpas<8, batchnum, float, float, fp16, fp16>(accResult, weightsDataHigh, inputDataHigh);
    }

    slm_block_store<float, batchnum * 16>(sizeof(float)* localId * batchnum * 16,  accResult);

    barrier();

    if (localId < batchnum)
    {
        simd<float, 16> outputData = 0.0;
        uint32_t slmOutputOffset = localId * 16 * sizeof(float);
#pragma unroll
        for (int j = 0; j < localRange; j++)
        {
            outputData += slm_block_load<float, 16>(slmOutputOffset);
            slmOutputOffset += batchnum * 16 * sizeof(float);
        }

        block_store<float, 16>(outputs + localId * output_len + h * outputPerGroup, outputData);
    }
}


template<uint32_t batchnum>
ESIMD_INLINE void slimGemmGroup32Block256BatchN_L5(float* inputs, uint8_t* weights, fp16* scales, fp16* zps, float* outputs, uint32_t input_len, uint32_t output_len, nd_item<1>& ndi)
{
    slm_init(sizeof(float) * 64 * batchnum * 16);

    constexpr uint32_t outputPerGroup = 16;

    uint32_t h = ndi.get_group(0);
    uint32_t localId = ndi.get_local_id(0);
    uint32_t localRange = ndi.get_local_range(0);

    uint32_t inputOffset = localId * 32;
    uint32_t weightOffset = h * outputPerGroup * input_len/2 + 256 * localId;
    simd<float, batchnum * 256> readData;
    simd<fp16, batchnum * 256> inputData;

    simd<float, batchnum * 16> accResult = 0.0;

#pragma unroll
    for (int j = 0; j < batchnum * 8; j++)
    {
        readData.template select<32, 1>(j * 32) = block_load<float, 32>(inputs + inputOffset);
        inputOffset += 32 * localRange;
    }
    inputData = readData / 4; // to handle granite overflow issue

    simd<uint8_t, 2048> weightsBuffer;
#pragma unroll
    for (int j = 0; j < 8; j ++)
    {
        weightsBuffer.select<256, 1>(j*256) = block_load<uint8_t, 256>(weights + weightOffset);
        weightOffset += 256 * localRange;
    }

    simd<fp16, 128> scaleData;
    simd<fp16, 128> zpsData;

    uint32_t scaleOffset = localId  * output_len + h * outputPerGroup;
#pragma unroll
    for (int j = 0; j < 8; j ++)
    {
        scaleData.select<16, 1>(j * 16) = block_load<fp16, 16>(scales + scaleOffset);
        zpsData.select<16, 1>(j * 16) = block_load<fp16, 16>(zps + scaleOffset);
        scaleOffset += localRange * output_len;
    }

    scaleData *= 4;
    zpsData *= 4;

#pragma unroll
    for(int j = 0; j < 8; j ++)
    {
        simd<fp16, 256> weightsDataLow = weightsBuffer.select<256, 1>(j * 256) & 0x0f;
        

#pragma unroll
        for (int k = 0; k < 8; k ++)
        {
            weightsDataLow.select<16, 2>(32 * k) = weightsDataLow.select<16, 2>(32 * k) * scaleData.select<16, 1>(j * 16) + zpsData.select<16, 1>(j * 16);
            weightsDataLow.select<16, 2>(32 * k + 1) = weightsDataLow.select<16, 2>(32 * k + 1) * scaleData.select<16, 1>(j * 16) + zpsData.select<16, 1>(j * 16);
        }

        simd<fp16, batchnum * 16> inputDataLow;
#pragma unroll
        for (int k = 0; k < batchnum; k++)
        {
            inputDataLow.template select<16, 1>(k * 16) = inputData.template select<16,  1>(k * 256 + j * 32);
        }

        accResult = xmx::dpas<8, batchnum, float, float, fp16, fp16>(accResult, weightsDataLow, inputDataLow);


        simd<fp16, 256> weightsDataHigh = weightsBuffer.select<256, 1>(j * 256) >> 4;
#pragma unroll
        for (int k = 0; k < 8; k ++)
        {
            weightsDataHigh.select<16, 2>(32 * k) = weightsDataHigh.select<16, 2>(32 * k) * scaleData.select<16, 1>(j * 16) + zpsData.select<16, 1>(j * 16);
            weightsDataHigh.select<16, 2>(32 * k + 1) = weightsDataHigh.select<16, 2>(32 * k + 1) * scaleData.select<16, 1>(j * 16) + zpsData.select<16, 1>(j * 16);
        }

        simd<fp16, batchnum * 16> inputDataHigh;
#pragma unroll
        for (int k = 0; k < batchnum; k++)
        {
            inputDataHigh.template select<16, 1>(k * 16) = inputData.template select<16,  1>(k * 256 + j * 32 + 16);
        }
        accResult = xmx::dpas<8, batchnum, float, float, fp16, fp16>(accResult, weightsDataHigh, inputDataHigh);
    }

    slm_block_store<float, batchnum * 16>(sizeof(float)* localId * batchnum * 16,  accResult);

    barrier();

    if (localId < batchnum)
    {
        simd<float, 16> outputData = 0.0;
        uint32_t slmOutputOffset = localId * 16 * sizeof(float);
#pragma unroll
        for (int j = 0; j < localRange; j++)
        {
            outputData += slm_block_load<float, 16>(slmOutputOffset);
            slmOutputOffset += batchnum * 16 * sizeof(float);
        }

        block_store<float, 16>(outputs + localId * output_len + h * outputPerGroup, outputData);
    }
}


bool runSlimGemmQ41_L5(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt)
{
    if (input_len % 256 != 0)
    {
        printf("Error! current Q41 GEMV only supports 256x input_length, input_length %d\n", input_len);
        return false;
    }

    sycl::event e;
    int groups = (output_len + 15) / 16;

    uint32_t localRange_128 = (input_len + 127) / 128;
    uint32_t localRange_256 = (input_len + 255) / 256;

    sycl::range<1> GlobalRange_128(groups * localRange_128);
    sycl::range<1> LocalRange_128(localRange_128);
    sycl::nd_range<1> Range_128(GlobalRange_128, localRange_128);

    sycl::range<1> GlobalRange_256(groups * localRange_256);
    sycl::range<1> LocalRange_256(localRange_256);
    sycl::nd_range<1> Range_256(GlobalRange_256, LocalRange_256);
    switch (batch)
    {
        case 1:
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range_256, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    slimGemmGroup32Block256BatchN_L5<1>((float*)inputs, (uint8_t*)weights, (fp16*)scales, (fp16*)zps, (float*)outputs, input_len, output_len, ndi);
                });
            });
            break;
        case 2:
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range_256, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    slimGemmGroup32Block256BatchN_L5<2>((float*)inputs, (uint8_t*)weights, (fp16*)scales, (fp16*)zps, (float*)outputs, input_len, output_len, ndi);
                });
            });
            break;
        case 3:
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range_256, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    slimGemmGroup32Block256BatchN_L5<3>((float*)inputs, (uint8_t*)weights, (fp16*)scales, (fp16*)zps, (float*)outputs, input_len, output_len, ndi);
                });
            });
            break;
        case 4:
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range_256, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    slimGemmGroup32Block256BatchN_L5<4>((float*)inputs, (uint8_t*)weights, (fp16*)scales, (fp16*)zps, (float*)outputs, input_len, output_len, ndi);
                });
            });
            break;
        case 5:
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    slimGemmGroup32Block128BatchN_L5<5>((float*)inputs, (uint8_t*)weights, (fp16*)scales, (fp16*)zps, (float*)outputs, input_len, output_len, ndi);
                });
            });
            break;
        case 6:
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    slimGemmGroup32Block128BatchN_L5<6>((float*)inputs, (uint8_t*)weights, (fp16*)scales, (fp16*)zps, (float*)outputs, input_len, output_len, ndi);
                });
            });
            break;
        case 7:
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    slimGemmGroup32Block128BatchN_L5<7>((float*)inputs, (uint8_t*)weights, (fp16*)scales, (fp16*)zps, (float*)outputs, input_len, output_len, ndi);
                });
            });
            break;
        case 8:
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range_128, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                    slimGemmGroup32Block128BatchN_L5<8>((float*)inputs, (uint8_t*)weights, (fp16*)scales, (fp16*)zps, (float*)outputs, input_len, output_len, ndi);
                });
            });
            break;
        default:
            printf("Error! Q41 GEMV only supports k <= 8 right now.\n");
    }

    return true;
}