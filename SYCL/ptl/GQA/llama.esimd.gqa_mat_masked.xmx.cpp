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

#define FP32_MAX (1.7e+38)
#define FP32_MIN (-1.7e+38)

using namespace std;
using namespace sycl::ext::intel::esimd;

#define GROUP_SIZE 128

extern "C" bool runGQA_mat_masked_fusion_xmx_simd16(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale, uint8_t* shuffleTt);
extern "C" bool runGQA_mat_masked_fusion_xmx_simd8(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale, uint8_t* shuffleTt);

ESIMD_INLINE void gqa_mat_kernel_hidden128_masked_simd8(uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* mask, uint8_t* out, int token_len, int kv_len, uint32_t KVHead, uint32_t QHead, float attn_scale, nd_item<2>& ndi)
{
    __ESIMD_NS::slm_init(32 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 32 * 64 * sizeof(fp16) + 32 * sizeof(float) + 32 * sizeof(float));
    constexpr uint32_t slmOffset_shuffled_QData = 0;
    constexpr uint32_t slmOffset_shuffled_KData = 32 * 128 * sizeof(fp16);
    constexpr uint32_t slmOffset_shuffled_QKData = slmOffset_shuffled_QData;
    constexpr uint32_t slmOffset_shuffled_SData = 32 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16);
    
    constexpr uint32_t slmOffset_shuffled_VData = 32 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16);
    constexpr uint32_t slmOffset_accSoftMax = 32 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 32 * 64 * sizeof(fp16);
    constexpr uint32_t slmOffset_compensates = 32 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 32 * 64 * sizeof(fp16) + 32 * sizeof(float);



    // attn_scale passed as parameter
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint32_t alignedKVLen = (kv_len + 15)/16*16;
    uint32_t n_kv_integer = (kv_len + 7) / 8;

    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int localLinearId = ndi.get_local_linear_id();

    uint32_t offsetQ = v * 32 * QHead * 128 + h * 128;
    
    int hh = localLinearId & 0x03;
    int vv = localLinearId >> 2;
    simd<fp16, 128> qDataBuf;

    if (v * 32 + localLinearId < token_len)
    {
        qDataBuf = block_load<float, 128>((float*)qState + offsetQ + localLinearId * QHead * 128);
    }

    int bidx = localLinearId >> 3;
    int iidx = localLinearId & 0x7;
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        slm_block_store<fp16, 16>(slmOffset_shuffled_QData + sizeof(fp16) * (iidx * 16 + i * 8 * 16 + bidx * 8 * 16 * 8), qDataBuf.select<16, 1>(i * 16));
    }

    barrier();
    simd<uint32_t, 16> base16Offset(baseOffsetInc16);
    simd<uint32_t, 64> base64Offset;
    base64Offset.select<16, 1>(0) = base16Offset;
    base64Offset.select<16, 1>(16) = base64Offset.select<16, 1>(0) + 16;
    base64Offset.select<16, 1>(32) = base64Offset.select<16, 1>(16) + 16;
    base64Offset.select<16, 1>(48) = base64Offset.select<16, 1>(32) + 16;

    simd<fp16, 128> qData0 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 0 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData1 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 1 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData2 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 2 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData3 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 3 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData4 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 4 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData5 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 5 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData6 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 6 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData7 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 7 * 8 * 16 * sizeof(fp16));

    simd<fp16, 128> kData0;
    simd<fp16, 128> kData1;
    simd<fp16, 128> kData2;
    simd<fp16, 128> kData3;
    simd<fp16, 128> kData4;
    simd<fp16, 128> kData5;
    simd<fp16, 128> kData6;
    simd<fp16, 128> kData7;


    simd<fp16, 128> vData0;
    simd<fp16, 128> vData1;
    simd<fp16, 128> vData2;
    simd<fp16, 128> vData3;
    simd<fp16, 128> vData4;
    simd<fp16, 128> vData5;
    simd<fp16, 128> vData6;
    simd<fp16, 128> vData7;

    simd<fp16, 128> sData0;
    simd<fp16, 128> sData1;
    simd<fp16, 128> sData2;
    simd<fp16, 128> sData3;

    int loopStep = 8 * 8;
    uint32_t startPos = kv_len - token_len;
    int loopNum = ( startPos + (v + 1) * 32 + loopStep - 1) / loopStep;

    simd<uint32_t, 8> base8Offset(baseOffsetInc8);
    simd<uint32_t, 8> scatteredOffsetK = h * KVHead / QHead * 128 + base8Offset * KVHead * 128 + bidx * 8 * KVHead * 128 + iidx * 16;
    uint32_t directOffsetV = h * KVHead/QHead * 128 + localLinearId * KVHead * 128 * 2;

    simd<float, 8*16> accResult = 0.0;
    float prevmax = FP32_MIN;

    simd<uint8_t, 8> shifter = 0;
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        shifter[j] = (1 << j);
    }
    simd<uint16_t, 64> kq_mask = 1;

    for (int l = 0; l < loopNum; l++)
    {
        //simd<fp16, 128> kDataTemp0 = block_load<fp16, 128>((fp16 *)kState + offsetK);
        //simd<fp16, 128> kDataTemp1 = block_load<fp16, 128>((fp16 *)kState + offsetK + 32 * 128);
        simd<fp16, 128> kDataTemp0 = 0.0;
        simd<fp16, 128> kDataTemp1 = 0.0;
        
        if (l * loopStep + bidx * 8 < kv_len)
        {
            kDataTemp0.template bit_cast_view<uint32_t>().select<64, 1>(0) = __ESIMD_ENS::lsc_gather<
                uint32_t,
                8,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                8,
                uint32_t
                >((uint32_t*)kState, scatteredOffsetK * sizeof(fp16));
        }
        if (l * loopStep + bidx * 8 + 32 < kv_len)
        {
            kDataTemp1.template bit_cast_view<uint32_t>().select<64, 1>(0) = __ESIMD_ENS::lsc_gather<
                uint32_t,
                8,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                8,
                uint32_t
                >((uint32_t*)kState, (scatteredOffsetK + 32 * KVHead * 128) * sizeof(fp16));
        }
        // simd<fp16, 128> vDataTemp0 = block_load<fp16, 128>((fp16 *)vState + offsetV);
        // simd<fp16, 128> vDataTemp1 = block_load<fp16, 128>((fp16 *)vState + offsetV + 32 * 128);
        simd<fp16, 128> vDataTemp0 = 0.0;
        simd<fp16, 128> vDataTemp1 = 0.0;
        if (l * loopStep + 2 * localLinearId < kv_len)
        {
            vDataTemp0 = block_load<fp16, 128>((fp16*)vState + directOffsetV);
        }
        if (l * loopStep + 2 * localLinearId + 1 < kv_len)
        {
            vDataTemp1 = block_load<fp16, 128>((fp16*)vState + directOffsetV + KVHead * 128);
        }

        uint32_t mask_offset = (v * 32 + localLinearId)*n_kv_integer + l * 8;
        simd<uint8_t, 8> mask_value = 0;
        auto mask_uint64_value = mask_value.template bit_cast_view<uint64_t>();
        mask_uint64_value[0] = *(uint64_t *)(mask + mask_offset);

        scatteredOffsetK += loopStep * KVHead * 128;
        directOffsetV += loopStep * KVHead * 128;

        //barrier();
        slm_block_store<fp16, 128>(slmOffset_shuffled_KData + localLinearId * 128 * sizeof(fp16), kDataTemp0);
        slm_block_store<fp16, 128>(slmOffset_shuffled_KData + (localLinearId + 32) * 128 * sizeof(fp16), kDataTemp1);
        barrier();
        // slm_block_store<fp16, 128>(slmOffset_shuffled_VData + (vbidx * 64 * 8 + viidx * 16 * 8) * sizeof(fp16), vDataTemp.select<128, 1>(0));
        // slm_block_store<fp16, 128>(slmOffset_shuffled_VData + (vbidx * 64 * 8 + 32 * 8 + viidx * 16 * 8) * sizeof(fp16), vDataTemp.select<128, 1>(128));
#pragma unroll
        for (int i = 0; i < 16; i++)
        {
            simd<fp16, 16> temp;
            temp.select<8, 2>(0) = vDataTemp0.select<8, 1>(i * 8);
            temp.select<8, 2>(1) = vDataTemp1.select<8, 1>(i * 8);
            slm_block_store<fp16, 16>(slmOffset_shuffled_VData + sizeof(fp16)*(localLinearId * 2 * 8 + i * 8 * 64), temp);
        }
        
        
        simd<float, 64> acc = 0.0;
        bool needCompute = l * loopStep + 8 * vv < kv_len;
        
        if (needCompute)
        {
            kData0 = slm_block_load<fp16, 128>(slmOffset_shuffled_KData + (vv * 8 * 128 + 0 * 8 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData0, qData0);
            kData1 = slm_block_load<fp16, 128>(slmOffset_shuffled_KData + (vv * 8 * 128 + 1 * 8 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData1, qData1);
            kData2 = slm_block_load<fp16, 128>(slmOffset_shuffled_KData + (vv * 8 * 128 + 2 * 8 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData2, qData2);
            kData3 = slm_block_load<fp16, 128>(slmOffset_shuffled_KData + (vv * 8 * 128 + 3 * 8 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData3, qData3);
            kData4 = slm_block_load<fp16, 128>(slmOffset_shuffled_KData + (vv * 8 * 128 + 4 * 8 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData4, qData4);
            kData5 = slm_block_load<fp16, 128>(slmOffset_shuffled_KData + (vv * 8 * 128 + 5 * 8 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData5, qData5);
            kData6 = slm_block_load<fp16, 128>(slmOffset_shuffled_KData + (vv * 8 * 128 + 6 * 8 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData6, qData6);
            kData7 = slm_block_load<fp16, 128>(slmOffset_shuffled_KData + (vv * 8 * 128 + 7 * 8 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData7, qData7);
        }
// #pragma unroll
//             for (int i = 0; i < 8; i++)
//             {
//                 slm_block_store<float, 8>(slmOffset_shuffled_QKData + sizeof(float)*((hh * 8 + i) * 64 + vv * 8), acc.select<8, 1>(i * 8));
//             }
        slm_block_store<float, 64>(slmOffset_shuffled_QKData + sizeof(float)*(localLinearId * 64), acc);
        
        barrier();

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            uint8_t value = mask_value[k];
            kq_mask.select<8, 1>(8 * k) = (value & shifter);
        }
        if (v * 32 + localLinearId >= token_len)
        {
            kq_mask = 1;
        }

        simd<float, 64> qkrow;
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            qkrow.select<8, 1>(i*8) = slm_block_load<float, 8>(slmOffset_shuffled_QKData + sizeof(float) * (i * 8 * 32 + 8 * localLinearId));
        }

        qkrow = qkrow * attn_scale;
        //simd<float, 64> qkrow = slm_block_load<float, 64>(slmOffset_shuffled_QKData + localLinearId * 64 * sizeof(float));
        simd<uint32_t, 64> k_idx = base64Offset + l * loopStep;
        uint32_t t_idx = startPos + v * 32 + localLinearId;
        qkrow.merge(FP32_MIN, k_idx >= kv_len);
        qkrow.merge(FP32_MIN, kq_mask);
        float curmax = hmax<float, float, 64>(qkrow);
        curmax = curmax>prevmax?curmax:prevmax;
        qkrow = qkrow - curmax;
        qkrow = sycl::ext::intel::esimd::pow<float, 64, float>(2.718f, qkrow);

        
        float qkrowSum = sycl::ext::intel::esimd::detail::sum<float, float, 64>(qkrow);
        float compensate = (l == 0)? 1.0 : sycl::ext::intel::esimd::exp(prevmax - curmax);
        float accSoftmax = slm_block_load<float, 1>(slmOffset_accSoftMax + localLinearId*sizeof(float));
        if (l == 0)
        {
            accSoftmax = 0.0;
        }
        accSoftmax = accSoftmax * compensate + qkrowSum;
        slm_block_store<float, 1>(slmOffset_accSoftMax + localLinearId*sizeof(float), accSoftmax);
        slm_block_store<float, 1>(slmOffset_compensates + localLinearId * sizeof(float), compensate);
        prevmax = curmax;
       
        simd<fp16, 64> qktemp = qkrow;
        int sbidx = localLinearId >> 3;
        int siidx = localLinearId & 0x7;

#pragma unroll
        for (int i = 0; i < 4; i ++)
        {
            slm_block_store<fp16, 16>(slmOffset_shuffled_SData + (sbidx * 8 * 64 + i * 8 * 16 + siidx * 16) * sizeof(fp16), qktemp.select<16, 1>(i * 16));
        }
        
        barrier();

        if (true)
        {
            // compensate
            simd<float, 8> compensates = slm_block_load<float, 8>(slmOffset_compensates + hh * 8 * sizeof(float));

            simd<float, 64> acc0 = 0.0;
            simd<float, 64> acc1 = 0.0;
            vData0 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (vv * 16 * 64 + 0 * 8 * 16) * sizeof(fp16));
            sData0 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 0 * 8 * 16 * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData0, sData0);
            accResult.select<16, 1>(0 * 16) = accResult.select<16, 1>(0 * 16) * compensates[0];
            vData4 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (vv * 16 * 64 + 4 * 8 * 16) * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData4, sData0);
            accResult.select<16, 1>(1 * 16) = accResult.select<16, 1>(1 * 16) * compensates[1];


            vData1 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (vv * 16 * 64 + 1 * 8 * 16) * sizeof(fp16));
            sData1 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 1 * 8 * 16 * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData1, sData1);
            accResult.select<16, 1>(2 * 16) = accResult.select<16, 1>(2 * 16) * compensates[2];

            vData5 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (vv * 16 * 64 + 5 * 8 * 16) * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData5, sData1);
            accResult.select<16, 1>(3 * 16) = accResult.select<16, 1>(3 * 16) * compensates[3];

            vData2 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (vv * 16 * 64 + 2 * 8 * 16) * sizeof(fp16));
            sData2 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 2 * 8 * 16 * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData2, sData2);
            accResult.select<16, 1>(4 * 16) = accResult.select<16, 1>(4 * 16) * compensates[4];
            
            vData6 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (vv * 16 * 64 + 6 * 8 * 16) * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData6, sData2);
            accResult.select<16, 1>(5 * 16) = accResult.select<16, 1>(5 * 16) * compensates[5];

            vData3 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (vv * 16 * 64 + 3 * 8 * 16) * sizeof(fp16));
            sData3 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 3 * 8 * 16 * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData3, sData3);
            accResult.select<16, 1>(6 * 16) = accResult.select<16, 1>(6 * 16) * compensates[6];

            vData7 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (vv * 16 * 64 + 7 * 8 * 16) * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData7, sData3);
            accResult.select<16, 1>(7 * 16) = accResult.select<16, 1>(7 * 16) * compensates[7];

            
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                accResult.select<8, 1>(i*16) = accResult.select<8, 1>(i*16) + acc0.select<8, 1>(i*8);
                accResult.select<8, 1>(i*16 + 8) = accResult.select<8, 1>(i*16 + 8) + acc1.select<8, 1>(i*8);
            }

        }
    }

    simd<float, 8> accsoftmaxs = slm_block_load<float, 8>(slmOffset_accSoftMax + hh * 8 * sizeof(float));

    uint32_t outputOffset = v * 32 * QHead * 128 + h * 128;
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        simd<float, 16> temp;
        if (accsoftmaxs[i] != 0)
        { 
            temp = accResult.select<16, 1>(i * 16) / accsoftmaxs[i];
        }
        else
        {
            temp = 0.0;
        }
        block_store<float, 16>((float*)out + outputOffset + (hh * 8 + i) * QHead * 128 + vv * 16, temp);
    }
    
}

ESIMD_INLINE void gqa_mat_kernel_hidden128_masked_simd16(uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* mask, uint8_t* out, int token_len, int kv_len, uint32_t KVHead, uint32_t QHead, float attn_scale, nd_item<2>& ndi)
{
    __ESIMD_NS::slm_init(64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 64 * sizeof(fp16) + 64 * sizeof(float) + 64 * sizeof(float));
    constexpr uint32_t slmOffset_shuffled_QData = 0;
    constexpr uint32_t slmOffset_shuffled_KData = 64 * 128 * sizeof(fp16);
    constexpr uint32_t slmOffset_shuffled_QKData = slmOffset_shuffled_QData;
    constexpr uint32_t slmOffset_shuffled_SData = 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16);
    
    constexpr uint32_t slmOffset_shuffled_VData = 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16);
    constexpr uint32_t slmOffset_accSoftMax = 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 64 * sizeof(fp16);
    constexpr uint32_t slmOffset_compensates = 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 128 * sizeof(fp16) + 64 * 64 * sizeof(fp16) + 64 * sizeof(float);



    // attn_scale passed as parameter
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    uint32_t alignedKVLen = (kv_len + 15)/16*16;
    uint32_t n_kv_integer = (kv_len + 7) / 8;

    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int localLinearId = ndi.get_local_linear_id();

    uint32_t offsetQ = v * 64 * QHead * 128 + h * 128;
    
    int hh = localLinearId & 0x07;
    int vv = localLinearId >> 3;
    simd<fp16, 128> qDataBuf0 = 0.0;
    simd<fp16, 128> qDataBuf1 = 0.0;

    if (v * 64 + localLinearId < token_len)
    {
        qDataBuf0 = block_load<float, 128>((float*)qState + offsetQ + localLinearId * QHead * 128);
    }
    if (v * 64 + localLinearId + 32 < token_len)
    {
        qDataBuf1 = block_load<float, 128>((float*)qState + offsetQ + (32 + localLinearId) * QHead * 128);
    }

    int bidx = localLinearId >> 3;
    int iidx = localLinearId & 0x7;
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        slm_block_store<fp16, 16>(slmOffset_shuffled_QData + sizeof(fp16) * (iidx * 16 + i * 8 * 16 + bidx * 8 * 16 * 8), qDataBuf0.select<16, 1>(i * 16));
        slm_block_store<fp16, 16>(slmOffset_shuffled_QData + sizeof(fp16) * (iidx * 16 + i * 8 * 16 + 32 * 128 + bidx * 8 * 16 * 8), qDataBuf1.select<16, 1>(i * 16));
    }

    barrier();
    simd<uint32_t, 16> base16Offset(baseOffsetInc16);
    simd<uint32_t, 64> base64Offset;
    base64Offset.select<16, 1>(0) = base16Offset;
    base64Offset.select<16, 1>(16) = base64Offset.select<16, 1>(0) + 16;
    base64Offset.select<16, 1>(32) = base64Offset.select<16, 1>(16) + 16;
    base64Offset.select<16, 1>(48) = base64Offset.select<16, 1>(32) + 16;

    simd<fp16, 128> qData0 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 0 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData1 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 1 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData2 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 2 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData3 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 3 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData4 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 4 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData5 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 5 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData6 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 6 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData7 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 128 * sizeof(fp16) + 7 * 8 * 16 * sizeof(fp16));

    simd<fp16, 256> kData0;
    simd<fp16, 256> kData1;
    simd<fp16, 256> kData2;
    simd<fp16, 256> kData3;
    simd<fp16, 256> kData4;
    simd<fp16, 256> kData5;
    simd<fp16, 256> kData6;
    simd<fp16, 256> kData7;


    simd<fp16, 256> vData0;
    simd<fp16, 256> vData1;
    simd<fp16, 256> vData2;
    simd<fp16, 256> vData3;
    simd<fp16, 256> vData4;
    simd<fp16, 256> vData5;
    simd<fp16, 256> vData6;
    simd<fp16, 256> vData7;

    simd<fp16, 128> sData0;
    simd<fp16, 128> sData1;
    simd<fp16, 128> sData2;
    simd<fp16, 128> sData3;

    int loopStep = 8 * 8;
    uint32_t startPos = kv_len - token_len;
    int loopNum = ( startPos + (v + 1) * 64 + loopStep - 1) / loopStep;

    int kbidx = localLinearId >> 4;
    int kiidx = localLinearId & 0x0f;
    simd<uint32_t, 16> scatteredOffsetK = h * KVHead / QHead * 128 + base16Offset * KVHead * 128 + kbidx * 16 * KVHead * 128 + kiidx * 8;
    uint32_t directOffsetV = h * KVHead/QHead * 128 + localLinearId * KVHead * 128 * 2;

    simd<float, 8*32> accResult = 0.0;
    float prevmax0 = FP32_MIN;
    float prevmax1 = FP32_MIN;

    simd<uint8_t, 8> shifter = 0;
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        shifter[j] = (1 << j);
    }

    simd<uint16_t, 64> kq_mask0 = 1;
    simd<uint16_t, 64> kq_mask1 = 1;

    for (int l = 0; l < loopNum; l++)
    {
        //simd<fp16, 128> kDataTemp0 = block_load<fp16, 128>((fp16 *)kState + offsetK);
        //simd<fp16, 128> kDataTemp1 = block_load<fp16, 128>((fp16 *)kState + offsetK + 32 * 128);
        simd<fp16, 128> kDataTemp0 = 0.0;
        simd<fp16, 128> kDataTemp1 = 0.0;
        
        if (l * loopStep + kbidx * 16 < kv_len)
        {
            kDataTemp0.template bit_cast_view<uint32_t>().select<64, 1>(0) = __ESIMD_ENS::lsc_gather<
                uint32_t,
                4,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)kState, scatteredOffsetK * sizeof(fp16));
        }
        if (l * loopStep + 32 + kbidx * 16 < kv_len)
        {
            kDataTemp1.template bit_cast_view<uint32_t>().select<64, 1>(0) = __ESIMD_ENS::lsc_gather<
                uint32_t,
                4,
                __ESIMD_ENS::lsc_data_size::u32,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached,
                16,
                uint32_t
                >((uint32_t*)kState, (scatteredOffsetK + 32 * KVHead * 128) * sizeof(fp16));
        }
        // simd<fp16, 128> vDataTemp0 = block_load<fp16, 128>((fp16 *)vState + offsetV);
        // simd<fp16, 128> vDataTemp1 = block_load<fp16, 128>((fp16 *)vState + offsetV + 32 * 128);
        simd<fp16, 128> vDataTemp0 = 0.0;
        simd<fp16, 128> vDataTemp1 = 0.0;
        if (l * loopStep + 2 * localLinearId < kv_len)
        {
            vDataTemp0 = block_load<fp16, 128>((fp16*)vState + directOffsetV);
        }
        if (l * loopStep + 2 * localLinearId + 1 < kv_len)
        {
            vDataTemp1 = block_load<fp16, 128>((fp16*)vState + directOffsetV + KVHead * 128);
        }

        simd<uint8_t, 8> mask_value0 = 0;
        simd<uint8_t, 8> mask_value1 = 0;
        auto mask_uint64_value0 = mask_value0.template bit_cast_view<uint64_t>();
        auto mask_uint64_value1 = mask_value1.template bit_cast_view<uint64_t>();
        uint32_t mask_offset0 = (v * 64 + localLinearId * 2)*n_kv_integer + l * 8;
        uint32_t mask_offset1 = (v * 64 + localLinearId * 2 + 1)*n_kv_integer + l * 8;
        mask_uint64_value0[0] = *(uint64_t *)(mask + mask_offset0);
        mask_uint64_value1[0] = *(uint64_t *)(mask + mask_offset1);

        // simd<uint8_t, 8> mask_value0 = block_load<uint8_t, 8>(mask + (v * 64 + localLinearId * 2)*n_kv_integer + l * 8);
        // simd<uint8_t, 8> mask_value1 = block_load<uint8_t, 8>(mask + (v * 64 + localLinearId * 2 + 1)*n_kv_integer + l * 8);

        scatteredOffsetK += loopStep * KVHead * 128;
        directOffsetV += loopStep * KVHead * 128;

        //barrier();
        slm_block_store<fp16, 128>(slmOffset_shuffled_KData + localLinearId * 128 * sizeof(fp16), kDataTemp0);
        slm_block_store<fp16, 128>(slmOffset_shuffled_KData + (localLinearId + 32) * 128 * sizeof(fp16), kDataTemp1);
        barrier();
        // slm_block_store<fp16, 128>(slmOffset_shuffled_VData + (vbidx * 64 * 8 + viidx * 16 * 8) * sizeof(fp16), vDataTemp.select<128, 1>(0));
        // slm_block_store<fp16, 128>(slmOffset_shuffled_VData + (vbidx * 64 * 8 + 32 * 8 + viidx * 16 * 8) * sizeof(fp16), vDataTemp.select<128, 1>(128));
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            simd<fp16, 32> temp;
            temp.select<16, 2>(0) = vDataTemp0.select<16, 1>(i * 16);
            temp.select<16, 2>(1) = vDataTemp1.select<16, 1>(i * 16);
            slm_block_store<fp16, 32>(slmOffset_shuffled_VData + sizeof(fp16)*(localLinearId * 2 * 16 + i * 16 * 64), temp);
        }
        
        
        simd<float, 128> acc = 0.0;
        //bool needCompute = 64*v + (hh + 1) * 8 + startPos >=  l * loopStep + 16 * vv;
        bool needCompute = l * loopStep + 16 * vv < kv_len;
        
        if (needCompute)
        {
            kData0 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 128 + 0 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData0, qData0);
            kData1 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 128 + 1 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData1, qData1);
            kData2 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 128 + 2 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData2, qData2);
            kData3 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 128 + 3 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData3, qData3);
            kData4 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 128 + 4 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData4, qData4);
            kData5 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 128 + 5 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData5, qData5);
            kData6 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 128 + 6 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData6, qData6);
            kData7 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 128 + 7 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData7, qData7);
        }
// #pragma unroll
//             for (int i = 0; i < 8; i++)
//             {
//                 slm_block_store<float, 8>(slmOffset_shuffled_QKData + sizeof(float)*((hh * 8 + i) * 64 + vv * 8), acc.select<8, 1>(i * 8));
//             }
        slm_block_store<float, 128>(slmOffset_shuffled_QKData + sizeof(float)*(localLinearId * 128), acc);
        
        barrier();

#pragma unroll
        for (int k = 0; k < 8; k++)
        {
            uint8_t value0 = mask_value0[k];
            kq_mask0.select<8, 1>(8 * k) = (value0 & shifter);
            uint8_t value1 = mask_value1[k];
            kq_mask1.select<8, 1>(8 * k) = (value1 & shifter);
        }
        if (v * 64 + localLinearId * 2 >= token_len)
        {
            kq_mask0 = 1;
        }
        if (v * 64 + localLinearId * 2 + 1 >= token_len)
        {
            kq_mask1 = 1;
        }

        simd<float, 64> qkrow0;
        simd<float, 64> qkrow1;
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            qkrow0.select<16, 1>(i*16) = slm_block_load<float, 16>(slmOffset_shuffled_QKData + sizeof(float) * (i * 16 * 64 + 16 * 2 * localLinearId));
            qkrow1.select<16, 1>(i*16) = slm_block_load<float, 16>(slmOffset_shuffled_QKData + sizeof(float) * (i * 16 * 64 + 16 * 2 * localLinearId + 16));
        }

        qkrow0 = qkrow0 * attn_scale;
        qkrow1 = qkrow1 * attn_scale;
        //simd<float, 64> qkrow = slm_block_load<float, 64>(slmOffset_shuffled_QKData + localLinearId * 64 * sizeof(float));
        simd<uint32_t, 64> k_idx = base64Offset + l * loopStep;
        uint32_t t_idx = startPos + v * 64 + localLinearId * 2;
        qkrow0.merge(FP32_MIN, k_idx >= kv_len);
        qkrow0.merge(FP32_MIN, kq_mask0);
        qkrow1.merge(FP32_MIN, k_idx >= kv_len);
        qkrow1.merge(FP32_MIN, kq_mask1);
        float curmax0 = hmax<float, float, 64>(qkrow0);
        float curmax1 = hmax<float, float, 64>(qkrow1);
        curmax0 = curmax0>prevmax0?curmax0:prevmax0;
        curmax1 = curmax1>prevmax1?curmax1:prevmax1;
        qkrow0 = qkrow0 - curmax0;
        qkrow1 = qkrow1 - curmax1;
        qkrow0 = sycl::ext::intel::esimd::pow<float, 64, float>(2.718f, qkrow0);
        qkrow1 = sycl::ext::intel::esimd::pow<float, 64, float>(2.718f, qkrow1);

        simd<float, 2> qkrowSum;
        qkrowSum[0] = sycl::ext::intel::esimd::detail::sum<float, float, 64>(qkrow0);
        qkrowSum[1] = sycl::ext::intel::esimd::detail::sum<float, float, 64>(qkrow1);
        simd<float, 2> compensate;
        compensate[0] = (l == 0)? 1.0 : sycl::ext::intel::esimd::exp(prevmax0 - curmax0);
        compensate[1] = (l == 0)? 1.0 : sycl::ext::intel::esimd::exp(prevmax1 - curmax1);
        simd<float, 2> accSoftmax = slm_block_load<float, 2>(slmOffset_accSoftMax + localLinearId*2*sizeof(float));
        if (l == 0)
        {
            accSoftmax = 0.0;
        }
        accSoftmax = accSoftmax * compensate + qkrowSum;
        slm_block_store<float, 2>(slmOffset_accSoftMax + localLinearId *2* sizeof(float), accSoftmax);
        slm_block_store<float, 2>(slmOffset_compensates + localLinearId *2* sizeof(float), compensate);
        prevmax0 = curmax0;
        prevmax1 = curmax1;
       
        simd<fp16, 64> qktemp0 = qkrow0;
        simd<fp16, 64> qktemp1 = qkrow1;
        int sbidx = localLinearId >> 2;
        int siidx = localLinearId & 0x3;

#pragma unroll
        for (int i = 0; i < 4; i ++)
        {
            slm_block_store<fp16, 16>(slmOffset_shuffled_SData + (sbidx * 8 * 64 + i * 8 * 16 + 2 * siidx * 16) * sizeof(fp16), qktemp0.select<16, 1>(i * 16));
            slm_block_store<fp16, 16>(slmOffset_shuffled_SData + (sbidx * 8 * 64 + i * 8 * 16 + 2 * siidx * 16 + 16) * sizeof(fp16), qktemp1.select<16, 1>(i * 16));
        }
        
        barrier();

        if (true)
        {
            // compensate
            simd<float, 8> compensates = slm_block_load<float, 8>(slmOffset_compensates + hh * 8 * sizeof(float));

            simd<float, 128> acc0 = 0.0;
            simd<float, 128> acc1 = 0.0;
            vData0 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 32 * 64 + 0 * 16 * 16) * sizeof(fp16));
            sData0 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 0 * 8 * 16 * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData0, sData0);
            accResult.select<32, 1>(0 * 32) = accResult.select<32, 1>(0 * 32) * compensates[0];
            vData4 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 32 * 64 + 4 * 16 * 16) * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData4, sData0);
            accResult.select<32, 1>(1 * 32) = accResult.select<32, 1>(1 * 32) * compensates[1];


            vData1 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 32 * 64 + 1 * 16 * 16) * sizeof(fp16));
            sData1 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 1 * 8 * 16 * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData1, sData1);
            accResult.select<32, 1>(2 * 32) = accResult.select<32, 1>(2 * 32) * compensates[2];

            vData5 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 32 * 64 + 5 * 16 * 16) * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData5, sData1);
            accResult.select<32, 1>(3 * 32) = accResult.select<32, 1>(3 * 32) * compensates[3];

            vData2 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 32 * 64 + 2 * 16 * 16) * sizeof(fp16));
            sData2 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 2 * 8 * 16 * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData2, sData2);
            accResult.select<32, 1>(4 * 32) = accResult.select<32, 1>(4 * 32) * compensates[4];
            
            vData6 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 32 * 64 + 6 * 16 * 16) * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData6, sData2);
            accResult.select<32, 1>(5 * 32) = accResult.select<32, 1>(5 * 32) * compensates[5];

            vData3 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 32 * 64 + 3 * 16 * 16) * sizeof(fp16));
            sData3 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 3 * 8 * 16 * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData3, sData3);
            accResult.select<32, 1>(6 * 32) = accResult.select<32, 1>(6 * 32) * compensates[6];

            vData7 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 32 * 64 + 7 * 16 * 16) * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData7, sData3);
            accResult.select<32, 1>(7 * 32) = accResult.select<32, 1>(7 * 32) * compensates[7];

            
#pragma unroll
            for (int i = 0; i < 8; i++)
            {
                accResult.select<16, 1>(i*32) = accResult.select<16, 1>(i*32) + acc0.select<16, 1>(i*16);
                accResult.select<16, 1>(i*32 + 16) = accResult.select<16, 1>(i*32 + 16) + acc1.select<16, 1>(i*16);
            }

        }

    }

    simd<float, 8> accsoftmaxs = slm_block_load<float, 8>(slmOffset_accSoftMax + hh * 8 * sizeof(float));

    uint32_t outputOffset = v * 64 * QHead * 128 + h * 128;
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        simd<float, 32> temp;
        if (accsoftmaxs[i] != 0)
        { 
            temp = accResult.select<32, 1>(i * 32) / accsoftmaxs[i];
        }
        else
        {
            temp = 0.0;
        }
        block_store<float, 32>((float*)out + outputOffset + (hh * 8 + i) * QHead * 128 + vv * 32, temp);
    }
    
}


bool runGQA_mat_masked_fusion_xmx_simd16(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale, uint8_t* shuffleTt) {
    sycl::event e;

    try {
        // GQA
        {
            const uint32_t localThread = 32;
            uint32_t vThreadnum =(token_len + 63)/64;
            sycl::range<2> GlobalRange(localThread * q_head, vThreadnum);
            sycl::range<2> LocalRange(localThread, 1);
            sycl::nd_range<2> Range(GlobalRange, LocalRange);

            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                    gqa_mat_kernel_hidden128_masked_simd16(query, kCache, vCache, mask, outputs, token_len, kv_len, kv_head, q_head, attn_scale, ndi);
                  });
                });
        }
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    return true;
}

bool runGQA_mat_masked_fusion_xmx_simd8(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale, uint8_t* shuffleTt) {
    sycl::event e;

    try {
        // GQA
        {
            const uint32_t localThread = 32;
            uint32_t vThreadnum =(token_len + 31)/32;
            sycl::range<2> GlobalRange(localThread * q_head, vThreadnum);
            sycl::range<2> LocalRange(localThread, 1);
            sycl::nd_range<2> Range(GlobalRange, LocalRange);

            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                    gqa_mat_kernel_hidden128_masked_simd8(query, kCache, vCache, mask, outputs, token_len, kv_len, kv_head, q_head, attn_scale, ndi);
                  });
                });
        }
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    return true;
}

