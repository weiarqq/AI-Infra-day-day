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

extern "C" bool runQ41FfnSwiglu_L3(sycl::queue* q, uint8_t *up, uint8_t *down, uint8_t *input, uint8_t *output, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, uint8_t *scratch_buffer);


ESIMD_INLINE void shuffle_input(float *inputs, fp16 *outputs, uint32_t token_len, uint32_t input_len, nd_item<2>& ndi)
{
    const uint32_t blockWidth = 4096;
    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    uint32_t localRange = ndi.get_local_range(0);
    int localLinearId = ndi.get_local_linear_id();

    uint32_t alignedTokenLen = (token_len + 255) / 256 * 256;

    uint32_t curBlockWidth = input_len - h * blockWidth;
    curBlockWidth = curBlockWidth > blockWidth ? blockWidth : curBlockWidth;

    simd<float, 1024> inputData = 0;
    simd<fp16, 1024> outputData = 0;
    
    uint32_t v_blk_idx = v >> 4;
    uint32_t v_inn_idx = v & 0x0F;
    uint32_t inputOffset = v * 16 * input_len + h * blockWidth + localLinearId * 64;
    uint32_t outputOffset = v_blk_idx * 256 * input_len + h * 256 * blockWidth +  localLinearId * 256 * 64 + v_inn_idx * 16 * 16;
    uint32_t readOffset = inputOffset;
    uint32_t writeOffset = outputOffset;

    if (h * blockWidth + localLinearId * 64 < input_len)
    {
#pragma unroll
        for (int j = 0; j < 16; j++)
        {
            if (v * 16 + j < token_len)
            {
                inputData.select<64, 1>(j * 64) = block_load<float, 64>(inputs + readOffset);
                readOffset += input_len;
            }
        }

#pragma unroll
        for (int j = 0; j < 16 && v*16+j < token_len; j++)
        {
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                outputData.select<16, 1>(k * 256 + j * 16) = inputData.select<16, 1>(j * 64 + k * 16);
            }
        }

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            block_store<fp16, 256>(outputs + writeOffset, outputData.select<256, 1>(j * 256));
            writeOffset += 16 * 256;
        }
    }

    inputData = 0.0;
    outputData = 0.0;
    inputOffset = v * 16 * input_len + h * blockWidth + (localLinearId + localRange)* 64;
    outputOffset = v_blk_idx * 256 * input_len + h * 256 * blockWidth +  (localRange + localLinearId) * 256 * 64 + v_inn_idx * 16 * 16;
    readOffset = inputOffset;
    writeOffset = outputOffset;
    if (h * blockWidth + (localRange + localLinearId) * 64 < input_len)
    {
#pragma unroll
        for (int j = 0; j < 16; j++)
        {
            if (v * 16 + j < token_len)
            {
                inputData.select<64, 1>(j * 64) = block_load<float, 64>(inputs + readOffset);
                readOffset += input_len;
            }
        }

#pragma unroll
        for (int j = 0; j < 16 && v*16+j < token_len; j++)
        {
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                outputData.select<16, 1>(k * 256 + j * 16) = inputData.select<16, 1>(j * 64 + k * 16);
            }
        }

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            block_store<fp16, 256>(outputs + writeOffset, outputData.select<256, 1>(j * 256));
            writeOffset += 16 * 256;
        }
    }

}

ESIMD_INLINE void swiglu_q41weights_xmx16(fp16 *inputs, uint8_t *weights, fp16 *scales, fp16 *zps, fp16 *outputs, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, nd_item<2>& ndi)
{
    //slm_init(256 * 128 * sizeof(fp16) * 2);
    constexpr uint32_t slmOffsetScale = 0;
    constexpr uint32_t slmOffsetZps = 256 * 128 * sizeof(fp16);
    const uint32_t localRange = 32;

    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int localLinearId = ndi.get_local_linear_id();
    int hh = localLinearId & 0x07;
    int vv = localLinearId >> 3;

    if (v * 256 + vv * 64 >= token_len)
    {
        return;
    }

    uint32_t globalOffsetInput = v * 256 * input_len;
    uint32_t globalOffsetWeight = h * 256 * input_len;

    uint32_t localOffsetInput = globalOffsetInput + vv * 64 * 16;
    uint32_t localOffsetWeight = globalOffsetWeight/2 + hh * 32 * 16;
    uint32_t localOffsetWeightY = h *256 + hh * 32;
    uint32_t localOffsetWeightX = 0;
    

    uint32_t loopStep = 32;
    uint32_t loopNum = input_len / loopStep;

    uint32_t scaleOffset = (h * 256 + hh * 32) * input_len / 32;

    simd<fp16, 32> scales_buf;
    simd<fp16, 32> zps_buf;

    simd<fp16, 128> AData0_tik;
    simd<fp16, 128> AData1_tik;
    simd<fp16, 128> AData2_tik;
    simd<fp16, 128> AData3_tik;
    simd<fp16, 128> AData4_tik;
    simd<fp16, 128> AData5_tik;
    simd<fp16, 128> AData6_tik;
    simd<fp16, 128> AData7_tik;

    simd<uint8_t, 512> BData;
    simd<uint8_t, 256> BData0;
    simd<uint8_t, 256> BData1;

    simd<fp16, 256> BData0_tik;
    simd<fp16, 256> BData1_tik;

    simd<fp16, 128> AData0_tok;
    simd<fp16, 128> AData1_tok;
    simd<fp16, 128> AData2_tok;
    simd<fp16, 128> AData3_tok;
    simd<fp16, 128> AData4_tok;
    simd<fp16, 128> AData5_tok;
    simd<fp16, 128> AData6_tok;
    simd<fp16, 128> AData7_tok;


    simd<fp16, 256> BData0_tok;
    simd<fp16, 256> BData1_tok;

    simd<float, 128> CData00 = 0.0;
    simd<float, 128> CData10 = 0.0;
    simd<float, 128> CData20 = 0.0;
    simd<float, 128> CData30 = 0.0;
    simd<float, 128> CData40 = 0.0;
    simd<float, 128> CData50 = 0.0;
    simd<float, 128> CData60 = 0.0;
    simd<float, 128> CData70 = 0.0;

    simd<float, 128> CData01 = 0.0;
    simd<float, 128> CData11 = 0.0;
    simd<float, 128> CData21 = 0.0;
    simd<float, 128> CData31 = 0.0;
    simd<float, 128> CData41 = 0.0;
    simd<float, 128> CData51 = 0.0;
    simd<float, 128> CData61 = 0.0;
    simd<float, 128> CData71 = 0.0;

    AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
    AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
    AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
    AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
    AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
    AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
    AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
    AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

    BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)weights, input_len/2 - 1, 2 * hidden_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);

    scales_buf = block_load<fp16, 32>(scales + scaleOffset);
    zps_buf = block_load<fp16, 32>(zps + scaleOffset);
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
    }

    localOffsetInput += 16 * 256;
    localOffsetWeightX += 4;
    scaleOffset += 32;

    BData0_tik = BData0 & 0x0f;
    BData1_tik = BData1 & 0x0f;

    simd<fp16, 32> s;
    simd<fp16, 32> z;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s  + z;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s  + z;
    }

    for (int i = 0; i < loopNum - 1; i++)
    {
        AData0_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

        BData0_tok = BData0 >> 4;
        BData1_tok = BData1 >> 4;

        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tok.select<32, 1>(j * 32) = BData0_tok.select<32, 1>(j * 32) * s  + z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tok.select<32, 1>(j * 32) = BData1_tok.select<32, 1>(j * 32) * s  + z;
        }

        scales_buf = block_load<fp16, 32>(scales + scaleOffset);
        zps_buf = block_load<fp16, 32>(zps + scaleOffset);
        
        localOffsetInput += 16 * 256;

        CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tik, AData0_tik);
        CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tik, AData1_tik);
        CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tik, AData2_tik);
        CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tik, AData3_tik);
        CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tik, AData4_tik);
        CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tik, AData5_tik);
        CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tik, AData6_tik);
        CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tik, AData7_tik);

        CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tik, AData0_tik);
        CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tik, AData1_tik);
        CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tik, AData2_tik);
        CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tik, AData3_tik);
        CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tik, AData4_tik);
        CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tik, AData5_tik);
        CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tik, AData6_tik);
        CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tik, AData7_tik);


        AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

        BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)weights, input_len/2 - 1, 2 * hidden_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
        }

        localOffsetInput += 16 * 256;
        localOffsetWeightX += 4;
        scaleOffset += 32;

        BData0_tik = BData0 & 0x0f;
        BData1_tik = BData1 & 0x0f;
        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s  + z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s  + z;
        }

        CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tok, AData0_tok);
        CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tok, AData1_tok);
        CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tok, AData2_tok);
        CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tok, AData3_tok);
        CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tok, AData4_tok);
        CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tok, AData5_tok);
        CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tok, AData6_tok);
        CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tok, AData7_tok);

        CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tok, AData0_tok);
        CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tok, AData1_tok);
        CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tok, AData2_tok);
        CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tok, AData3_tok);
        CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tok, AData4_tok);
        CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tok, AData5_tok);
        CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tok, AData6_tok);
        CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tok, AData7_tok);


    }

    AData0_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
    AData1_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
    AData2_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
    AData3_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
    AData4_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
    AData5_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
    AData6_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
    AData7_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

    BData0_tok = BData0 >> 4;
    BData1_tok = BData1 >> 4;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tok.select<32, 1>(j * 32) = BData0_tok.select<32, 1>(j * 32) * s  + z;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tok.select<32, 1>(j * 32) = BData1_tok.select<32, 1>(j * 32) * s  + z;
    }
    localOffsetInput += 16 * 256;

    CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tik, AData0_tik);
    CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tik, AData1_tik);
    CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tik, AData2_tik);
    CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tik, AData3_tik);
    CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tik, AData4_tik);
    CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tik, AData5_tik);
    CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tik, AData6_tik);
    CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tik, AData7_tik);

    CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tik, AData0_tik);
    CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tik, AData1_tik);
    CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tik, AData2_tik);
    CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tik, AData3_tik);
    CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tik, AData4_tik);
    CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tik, AData5_tik);
    CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tik, AData6_tik);
    CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tik, AData7_tik);

    CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tok, AData0_tok);
    CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tok, AData1_tok);
    CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tok, AData2_tok);
    CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tok, AData3_tok);
    CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tok, AData4_tok);
    CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tok, AData5_tok);
    CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tok, AData6_tok);
    CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tok, AData7_tok);

    CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tok, AData0_tok);
    CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tok, AData1_tok);
    CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tok, AData2_tok);
    CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tok, AData3_tok);
    CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tok, AData4_tok);
    CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tok, AData5_tok);
    CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tok, AData6_tok);
    CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tok, AData7_tok);


    uint32_t output_offset = v * 256 * hidden_len + h * 128 * 256 + hh * 16 * 256 + vv * 16 * 64;

    simd<float, 128> DData0;
    simd<float, 128> DData1;
    simd<float, 128> DData2;
    simd<float, 128> DData3;
    simd<float, 128> DData4;
    simd<float, 128> DData5;
    simd<float, 128> DData6;
    simd<float, 128> DData7;

    DData0 = -1.0 * CData00;
    DData0 = pow<float, 128, float>(2.718f, DData0);
    DData0 = DData0 + 1.0;
    DData0 = 1.0/DData0;
    DData0 = DData0 * CData00;
    DData0 = DData0 * CData01;

    block_store<fp16, 128>(outputs + output_offset, DData0);
    output_offset += 128;

    DData1 = -1.0 * CData10;
    DData1 = pow<float, 128, float>(2.718f, DData1);
    DData1 = DData1 + 1.0;
    DData1 = 1.0/DData1;
    DData1 = DData1 * CData10;
    DData1 = DData1 * CData11;

    block_store<fp16, 128>(outputs + output_offset, DData1);
    output_offset += 128;

    DData2 = -1.0 * CData20;
    DData2 = pow<float, 128, float>(2.718f, DData2);
    DData2 = DData2 + 1.0;
    DData2 = 1.0/DData2;
    DData2 = DData2 * CData20;
    DData2 = DData2 * CData21;

    block_store<fp16, 128>(outputs + output_offset, DData2);
    output_offset += 128;

    DData3 = -1.0 * CData30;
    DData3 = pow<float, 128, float>(2.718f, DData3);
    DData3 = DData3 + 1.0;
    DData3 = 1.0/DData3;
    DData3 = DData3 * CData30;
    DData3 = DData3 * CData31;

    block_store<fp16, 128>(outputs + output_offset, DData3);
    output_offset += 128;

    DData4 = -1.0 * CData40;
    DData4 = pow<float, 128, float>(2.718f, DData4);
    DData4 = DData4 + 1.0;
    DData4 = 1.0/DData4;
    DData4 = DData4 * CData40;
    DData4 = DData4 * CData41;

    block_store<fp16, 128>(outputs + output_offset, DData4);
    output_offset += 128;

    DData5 = -1.0 * CData50;
    DData5 = pow<float, 128, float>(2.718f, DData5);
    DData5 = DData5 + 1.0;
    DData5 = 1.0/DData5;
    DData5 = DData5 * CData50;
    DData5 = DData5 * CData51;

    block_store<fp16, 128>(outputs + output_offset, DData5);
    output_offset += 128;

    DData6 = -1.0 * CData60;
    DData6 = pow<float, 128, float>(2.718f, DData6);
    DData6 = DData6 + 1.0;
    DData6 = 1.0/DData6;
    DData6 = DData6 * CData60;
    DData6 = DData6 * CData61;

    block_store<fp16, 128>(outputs + output_offset, DData6);
    output_offset += 128;

    DData7 = -1.0 * CData70;
    DData7 = pow<float, 128, float>(2.718f, DData7);
    DData7 = DData7 + 1.0;
    DData7 = 1.0/DData7;
    DData7 = DData7 * CData70;
    DData7 = DData7 * CData71;

    block_store<fp16, 128>(outputs + output_offset, DData7);
    output_offset += 128;
}

ESIMD_INLINE void gemm_down_q41weights_xmx16(fp16 *inputs, uint8_t *weights, fp16 *scales, fp16 *zps, float *outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, float* scratch, nd_item<2>& ndi)
{
    //slm_init(256 * 128 * sizeof(fp16) * 2);
    constexpr uint32_t slmOffsetScale = 0;
    constexpr uint32_t slmOffsetZps = 256 * 128 * sizeof(fp16);
    const uint32_t localRange = 32;

    float* debug = outputs + token_len * output_len;

    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int localLinearId = ndi.get_local_linear_id();
    int hh = localLinearId & 0x07;
    int vv = localLinearId >> 3;

    if (v * 256 + vv * 64 >= token_len)
    {
        return;
    }

    uint32_t globalOffsetInput = v * 256 * input_len;
    uint32_t globalOffsetWeight = h * 256 * input_len;
    uint32_t globalOffsetOutput = 256 * v * output_len + 256 * h;

    uint32_t localOffsetInput = globalOffsetInput + vv * 64 * 16;
    uint32_t localOffsetWeight = globalOffsetWeight/2 + hh * 32 * 16;
    uint32_t localOffsetWeightY = h *256 + hh * 32;
    uint32_t localOffsetWeightX = 0;
    

    uint32_t loopStep = 32;
    uint32_t loopNum = input_len / loopStep;

    uint32_t scaleOffset = (h * 256 + hh * 32) * input_len / 32;

    simd<fp16, 32> scales_buf;
    simd<fp16, 32> zps_buf;

    simd<fp16, 128> AData0_tik;
    simd<fp16, 128> AData1_tik;
    simd<fp16, 128> AData2_tik;
    simd<fp16, 128> AData3_tik;
    simd<fp16, 128> AData4_tik;
    simd<fp16, 128> AData5_tik;
    simd<fp16, 128> AData6_tik;
    simd<fp16, 128> AData7_tik;

    simd<uint8_t, 512> BData;
    simd<uint8_t, 256> BData0;
    simd<uint8_t, 256> BData1;

    simd<fp16, 256> BData0_tik;
    simd<fp16, 256> BData1_tik;

    simd<fp16, 128> AData0_tok;
    simd<fp16, 128> AData1_tok;
    simd<fp16, 128> AData2_tok;
    simd<fp16, 128> AData3_tok;
    simd<fp16, 128> AData4_tok;
    simd<fp16, 128> AData5_tok;
    simd<fp16, 128> AData6_tok;
    simd<fp16, 128> AData7_tok;


    simd<fp16, 256> BData0_tok;
    simd<fp16, 256> BData1_tok;

    simd<float, 128> CData00 = 0.0;
    simd<float, 128> CData10 = 0.0;
    simd<float, 128> CData20 = 0.0;
    simd<float, 128> CData30 = 0.0;
    simd<float, 128> CData40 = 0.0;
    simd<float, 128> CData50 = 0.0;
    simd<float, 128> CData60 = 0.0;
    simd<float, 128> CData70 = 0.0;

    simd<float, 128> CData01 = 0.0;
    simd<float, 128> CData11 = 0.0;
    simd<float, 128> CData21 = 0.0;
    simd<float, 128> CData31 = 0.0;
    simd<float, 128> CData41 = 0.0;
    simd<float, 128> CData51 = 0.0;
    simd<float, 128> CData61 = 0.0;
    simd<float, 128> CData71 = 0.0;

    AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
    AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
    AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
    AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
    AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
    AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
    AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
    AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

    // BData0.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 16, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);
    // BData1.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 16, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY + 16);
    BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);
    // scales_buf = block_load<fp16, 32>(scales + scaleOffset);
    // zps_buf = block_load<fp16, 32>(zps + scaleOffset);
    //scales_buf = slm_block_load<fp16, 32>(slmOffsetScale + scaleOffset);
    scales_buf = block_load<fp16, 32>(scales + scaleOffset);
    //zps_buf = slm_block_load<fp16, 32>(slmOffsetZps + scaleOffset);
    zps_buf = block_load<fp16, 32>(zps + scaleOffset);
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
    }

    localOffsetInput += 16 * 256;
    localOffsetWeightX += 4;
    scaleOffset += 32;

    BData0_tik = BData0 & 0x0f;
    BData1_tik = BData1 & 0x0f;

    simd<fp16, 32> s;
    simd<fp16, 32> z;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s  + z;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s  + z;
    }

    // if (h == 0 && v == 0 && hh == 1 && vv == 0)
    // {
    //     block_store<uint8_t, 512>((uint8_t *)debug, BData);
    // }
    for (int i = 0; i < loopNum - 1; i++)
    {
        AData0_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

        BData0_tok = BData0 >> 4;
        BData1_tok = BData1 >> 4;

        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tok.select<32, 1>(j * 32) = BData0_tok.select<32, 1>(j * 32) * s  + z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tok.select<32, 1>(j * 32) = BData1_tok.select<32, 1>(j * 32) * s  + z;
        }

        //scales_buf = slm_block_load<fp16, 32>(slmOffsetScale + scaleOffset);
        scales_buf = block_load<fp16, 32>(scales + scaleOffset);
        zps_buf = block_load<fp16, 32>(zps + scaleOffset);
        
        localOffsetInput += 16 * 256;

        CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tik, AData0_tik);
        CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tik, AData1_tik);
        CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tik, AData2_tik);
        CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tik, AData3_tik);
        CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tik, AData4_tik);
        CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tik, AData5_tik);
        CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tik, AData6_tik);
        CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tik, AData7_tik);

        CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tik, AData0_tik);
        CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tik, AData1_tik);
        CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tik, AData2_tik);
        CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tik, AData3_tik);
        CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tik, AData4_tik);
        CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tik, AData5_tik);
        CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tik, AData6_tik);
        CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tik, AData7_tik);


        AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

        // BData0.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 16, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);
        // BData1.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 16, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY + 16);

        BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);

        // if (h == 0 && v == 0 && hh == 1 && vv == 0)
        // {
        //     block_store<uint8_t, 512>((uint8_t *)debug + 512 * i + 512, BData);
        // }
        // scales_buf = block_load<fp16, 32>(scales + scaleOffset);
        // zps_buf = block_load<fp16, 32>(zps + scaleOffset);
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
        }

        localOffsetInput += 16 * 256;
        localOffsetWeightX += 4;
        scaleOffset += 32;

        BData0_tik = BData0 & 0x0f;
        BData1_tik = BData1 & 0x0f;
        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s  + z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s  + z;
        }

        CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tok, AData0_tok);
        CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tok, AData1_tok);
        CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tok, AData2_tok);
        CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tok, AData3_tok);
        CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tok, AData4_tok);
        CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tok, AData5_tok);
        CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tok, AData6_tok);
        CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tok, AData7_tok);

        CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tok, AData0_tok);
        CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tok, AData1_tok);
        CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tok, AData2_tok);
        CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tok, AData3_tok);
        CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tok, AData4_tok);
        CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tok, AData5_tok);
        CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tok, AData6_tok);
        CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tok, AData7_tok);


    }

    AData0_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
    AData1_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
    AData2_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
    AData3_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
    AData4_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
    AData5_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
    AData6_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
    AData7_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);

    BData0_tok = BData0 >> 4;
    BData1_tok = BData1 >> 4;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tok.select<32, 1>(j * 32) = BData0_tok.select<32, 1>(j * 32) * s  + z;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tok.select<32, 1>(j * 32) = BData1_tok.select<32, 1>(j * 32) * s  + z;
    }
    localOffsetInput += 16 * 256;

    CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tik, AData0_tik);
    CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tik, AData1_tik);
    CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tik, AData2_tik);
    CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tik, AData3_tik);
    CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tik, AData4_tik);
    CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tik, AData5_tik);
    CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tik, AData6_tik);
    CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tik, AData7_tik);

    CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tik, AData0_tik);
    CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tik, AData1_tik);
    CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tik, AData2_tik);
    CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tik, AData3_tik);
    CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tik, AData4_tik);
    CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tik, AData5_tik);
    CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tik, AData6_tik);
    CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tik, AData7_tik);

    CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tok, AData0_tok);
    CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tok, AData1_tok);
    CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tok, AData2_tok);
    CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tok, AData3_tok);
    CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tok, AData4_tok);
    CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tok, AData5_tok);
    CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tok, AData6_tok);
    CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tok, AData7_tok);

    CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tok, AData0_tok);
    CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tok, AData1_tok);
    CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tok, AData2_tok);
    CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tok, AData3_tok);
    CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tok, AData4_tok);
    CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tok, AData5_tok);
    CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tok, AData6_tok);
    CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tok, AData7_tok);


    uint32_t localOutputOffsetX = h * 256 + hh * 32;
    uint32_t localOutputOffsetY = v * 256 + vv * 64;

    float *write_buffer = outputs;
    uint32_t surfaceHeight = token_len;
    
    if (v * 256 + vv * 64 + 64 > token_len)
    {
        write_buffer = scratch;
        localOutputOffsetY = 0;
        surfaceHeight = 64;
    }

    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData00);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData01);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData10);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData11);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData20);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData21);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData30);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData31);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData40);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData41);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData50);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData51);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData60);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData61);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData70);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData71);


}



bool runQ41FfnSwiglu_L3(sycl::queue* q, uint8_t *up, uint8_t *down, uint8_t *input, uint8_t *output, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, uint8_t *scratch_buffer)
{
    // shuffle input
    uint8_t *shuffleTt = scratch_buffer;
    const uint32_t blockWidth = 4096;
    uint32_t threadShuffleH = (input_len + blockWidth - 1)/blockWidth;
    uint32_t alignedTokenLen = (token_len + 255)/256 * 256;
    uint32_t threadShuffleV = alignedTokenLen / 16;
    uint32_t localThreadShuffle = 32;

    sycl::range<2> GlobalRangeShuffle(localThreadShuffle * threadShuffleH, threadShuffleV);
    sycl::range<2> LocalRangeShuffle(localThreadShuffle, 1);
    sycl::nd_range<2> RangeShuffle(GlobalRangeShuffle, LocalRangeShuffle);

    try {
        sycl::event e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(RangeShuffle, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        shuffle_input((float *)input, (fp16 *)shuffleTt, token_len, input_len, ndi);
                    });
                });
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    // // up + gate + silu
    uint8_t *gateResult = scratch_buffer + alignedTokenLen * input_len * sizeof(fp16);
    uint8_t *scales_up = up + (input_len * hidden_len * 2) / 2;
    uint8_t *zps_up = scales_up + (input_len * hidden_len * 2) / 32 * sizeof(fp16);
    uint32_t threadA_up = (token_len + 255)/256;
    uint32_t threadB_up = (hidden_len * 2 + 255)/256;
    uint32_t localThreadNum_up = 32;

    sycl::range<2> GlobalRange_up(localThreadNum_up * threadB_up, threadA_up);
    sycl::range<2> LocalRange_up(localThreadNum_up, 1);
    sycl::nd_range<2> Range_up(GlobalRange_up, LocalRange_up);

    try {
        sycl::event e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(Range_up, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        swiglu_q41weights_xmx16((fp16 *)shuffleTt, (uint8_t *)up, (fp16 *)scales_up, (fp16 *)zps_up, (fp16 *)gateResult, token_len, input_len, hidden_len, ndi);
                    });
                });
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    // down
    uint8_t* outputScratch = gateResult + alignedTokenLen * hidden_len * sizeof(fp16);
    uint8_t *scales_down = down + (input_len * hidden_len) / 2;
    uint8_t *zps_down = scales_down + (input_len * hidden_len) / 32 * sizeof(fp16);
    uint32_t threadA_down = (token_len + 255)/256;
    uint32_t threadB_down = (input_len + 255)/256;
    uint32_t localThreadNum_down = 32;

    sycl::range<2> GlobalRange_down(localThreadNum_down * threadB_down, threadA_down);
    sycl::range<2> LocalRange_down(localThreadNum_down, 1);
    sycl::nd_range<2> Range_down(GlobalRange_down, LocalRange_down);

    try {
        sycl::event e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(Range_down, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        gemm_down_q41weights_xmx16((fp16 *)gateResult, (uint8_t *)down, (fp16 *)scales_down, (fp16 *)zps_down, (float *)output, token_len, hidden_len, input_len, (float *)outputScratch, ndi);
                    });
                });
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    uint32_t tailStart = token_len/64*64;
    uint32_t tailLen = token_len - tailStart;
    if (tailLen > 0)
    {
        q->memcpy((float *)output + tailStart * input_len, outputScratch, tailLen * input_len * sizeof(float));
    }

}

