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

extern "C" bool runSDP_vit_fusion_xmx_simd16(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, unsigned q_precision, unsigned kv_precision, unsigned o_precision, uint8_t* shuffleTt);

template<typename inT, typename outT>
ESIMD_INLINE void gqa_mat_kernel_hidden72_simd16(uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* mask, uint8_t* out, int token_len, int kv_len, uint32_t KVHead, uint32_t QHead, nd_item<2>& ndi)
{
    __ESIMD_NS::slm_init(128 * 64 * sizeof(float) + 64 * 80 * sizeof(fp16) + 64 * 80 * sizeof(fp16) + 128 * 64 * sizeof(fp16) + 128 * sizeof(float) + 128 * sizeof(float));
    constexpr uint32_t slmOffset_shuffled_QData = 0;
    constexpr uint32_t slmOffset_shuffled_KData = 128 * 64 * sizeof(float);
    constexpr uint32_t slmOffset_shuffled_QKData = slmOffset_shuffled_QData;
    constexpr uint32_t slmOffset_shuffled_SData = 128 * 64 * sizeof(float) + 64 * 80 * sizeof(fp16) + 64 * 80 * sizeof(fp16);
    
    constexpr uint32_t slmOffset_shuffled_VData = 128 * 64 * sizeof(float) + 64 * 80 * sizeof(fp16);
    constexpr uint32_t slmOffset_accSoftMax = 128 * 64 * sizeof(float) + 64 * 80 * sizeof(fp16) + 64 * 80 * sizeof(fp16) + 128 * 64 * sizeof(fp16);
    constexpr uint32_t slmOffset_compensates = 128 * 64 * sizeof(float) + 64 * 80 * sizeof(fp16) + 64 * 80 * sizeof(fp16) + 128 * 64 * sizeof(fp16) + 128 * sizeof(float);


    constexpr float attn_scale = 0.1178511301977579207f; // 1.0f / sqrt(72.0f);
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int localLinearId = ndi.get_local_linear_id();

    uint32_t offsetQ = v * 128 * QHead * 72 + h * 72;
    
    int hh = localLinearId & 0x07;
    int vv = localLinearId >> 3;
    simd<fp16, 80> qDataBuf0 = 0.0;
    simd<fp16, 80> qDataBuf1 = 0.0;
    simd<fp16, 80> qDataBuf2 = 0.0;
    simd<fp16, 80> qDataBuf3 = 0.0;

    if (v * 128 + localLinearId < token_len)
    {
        qDataBuf0.select<64, 1>(0) = block_load<inT, 64>((inT*)qState + offsetQ + localLinearId * QHead * 72);
        qDataBuf0.select<8, 1>(64) = block_load<inT, 8>((inT*)qState + offsetQ + localLinearId * QHead * 72 + 64);
    }
    if (v * 128 + localLinearId + 32 < token_len)
    {
        qDataBuf1.select<64, 1>(0) = block_load<inT, 64>((inT*)qState + offsetQ + (localLinearId + 32) * QHead * 72);
        qDataBuf1.select<8, 1>(64) = block_load<inT, 8>((inT*)qState + offsetQ + (localLinearId + 32) * QHead * 72 + 64);
    }
    if (v * 128 + localLinearId + 64 < token_len)
    {
        qDataBuf2.select<64, 1>(0) = block_load<inT, 64>((inT*)qState + offsetQ + (localLinearId + 64) * QHead * 72);
        qDataBuf2.select<8, 1>(64) = block_load<inT, 8>((inT*)qState + offsetQ + (localLinearId + 64) * QHead * 72 + 64);
    }
    if (v * 128 + localLinearId + 96 < token_len)
    {
        qDataBuf3.select<64, 1>(0) = block_load<inT, 64>((inT*)qState + offsetQ + (localLinearId + 96) * QHead * 72);
        qDataBuf3.select<8, 1>(64) = block_load<inT, 8>((inT*)qState + offsetQ + (localLinearId + 96) * QHead * 72 + 64);
    }

    int bidx = localLinearId >> 3;
    int iidx = localLinearId & 0x7;
#pragma unroll
    for (int i = 0; i < 5; i++)
    {
        slm_block_store<fp16, 16>(slmOffset_shuffled_QData + sizeof(fp16) * (iidx * 16 + i * 8 * 16 + bidx * 8 * 16 * 5), qDataBuf0.select<16, 1>(i * 16));
        slm_block_store<fp16, 16>(slmOffset_shuffled_QData + sizeof(fp16) * (iidx * 16 + i * 8 * 16 + bidx * 8 * 16 * 5 + 32 * 80), qDataBuf1.select<16, 1>(i * 16));
        slm_block_store<fp16, 16>(slmOffset_shuffled_QData + sizeof(fp16) * (iidx * 16 + i * 8 * 16 + bidx * 8 * 16 * 5 + 64 * 80), qDataBuf2.select<16, 1>(i * 16));
        slm_block_store<fp16, 16>(slmOffset_shuffled_QData + sizeof(fp16) * (iidx * 16 + i * 8 * 16 + bidx * 8 * 16 * 5 + 96 * 80), qDataBuf3.select<16, 1>(i * 16));
    }

    barrier();
    simd<uint32_t, 16> base16Offset(baseOffsetInc16);
    simd<uint32_t, 64> base64Offset;
    base64Offset.select<16, 1>(0) = base16Offset;
    base64Offset.select<16, 1>(16) = base64Offset.select<16, 1>(0) + 16;
    base64Offset.select<16, 1>(32) = base64Offset.select<16, 1>(16) + 16;
    base64Offset.select<16, 1>(48) = base64Offset.select<16, 1>(32) + 16;

    simd<fp16, 128> qData0 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 80 * sizeof(fp16) + 0 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData1 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 80 * sizeof(fp16) + 1 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData2 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 80 * sizeof(fp16) + 2 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData3 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 80 * sizeof(fp16) + 3 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData4 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + hh * 8 * 80 * sizeof(fp16) + 4 * 8 * 16 * sizeof(fp16));

    simd<fp16, 128> qData5 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + (hh+8) * 8 * 80 * sizeof(fp16) + 0 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData6 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + (hh+8) * 8 * 80 * sizeof(fp16) + 1 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData7 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + (hh+8) * 8 * 80 * sizeof(fp16) + 2 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData8 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + (hh+8) * 8 * 80 * sizeof(fp16) + 3 * 8 * 16 * sizeof(fp16));
    simd<fp16, 128> qData9 = slm_block_load<fp16, 128>(slmOffset_shuffled_QData + (hh+8) * 8 * 80 * sizeof(fp16) + 4 * 8 * 16 * sizeof(fp16));


    simd<fp16, 256> kData0;
    simd<fp16, 256> kData1;
    simd<fp16, 256> kData2;
    simd<fp16, 256> kData3;
    simd<fp16, 256> kData4;

    simd<fp16, 256> vData0;
    simd<fp16, 256> vData1;
    simd<fp16, 256> vData2;
    simd<fp16, 256> vData3;
    simd<fp16, 128> vData4;
    simd<fp16, 128> vData5;
    simd<fp16, 128> vData6;
    simd<fp16, 128> vData7;

    simd<fp16, 128> sData0;
    simd<fp16, 128> sData1;
    simd<fp16, 128> sData2;
    simd<fp16, 128> sData3;
    simd<fp16, 128> sData4;
    simd<fp16, 128> sData5;
    simd<fp16, 128> sData6;
    simd<fp16, 128> sData7;

    int loopStep = 8 * 8;
    int loopNum = (kv_len + loopStep - 1) / loopStep;

    simd<uint32_t, 16> scatteredOffsetK1 = h * KVHead / QHead * 72 + base16Offset * KVHead * 72 + vv * 16 * KVHead * 72 + hh * 8;
    simd<uint32_t, 16> scatteredOffsetK2 = h * KVHead / QHead * 72 + base16Offset * KVHead * 72 + vv * 16 * KVHead * 72 + 64;
    uint32_t directOffsetV = h * KVHead/QHead * 72 + localLinearId * KVHead * 72 * 2;

    uint32_t maskOffset0 = (v * 128 + localLinearId) * kv_len;
    uint32_t maskOffset1 = (v * 128 + localLinearId + 32) * kv_len;
    uint32_t maskOffset2 = (v * 128 + localLinearId + 64) * kv_len;
    uint32_t maskOffset3 = (v * 128 + localLinearId + 96) * kv_len;

    simd<float, 64> maskData0 = 0.0;
    simd<float, 64> maskData1 = 0.0;
    simd<float, 64> maskData2 = 0.0;
    simd<float, 64> maskData3 = 0.0;

    simd<float, 8*16> accResult0 = 0.0;
    simd<float, 8*16> accResult1 = 0.0;
    simd<float, 8> accResultTail0 = 0.0;
    simd<float, 8> accResultTail1 = 0.0;
    simd<float, 8> accResultTail2 = 0.0;
    simd<float, 8> accResultTail3 = 0.0;
    float accSoftmax0 = 0.0;
    float accSoftmax1 = 0.0;
    float accSoftmax2 = 0.0;
    float accSoftmax3 = 0.0;
    float prevmax0 = 0.0; // if the max is less than 0, leave it alone
    float prevmax1 = 0.0;
    float prevmax2 = 0.0;
    float prevmax3 = 0.0;

    for (int l = 0; l < loopNum; l++)
    {
        //simd<fp16, 128> kDataTemp0 = block_load<fp16, 128>((fp16 *)kState + offsetK);
        //simd<fp16, 128> kDataTemp1 = block_load<fp16, 128>((fp16 *)kState + offsetK + 32 * 128);
        simd<fp16, 128> kDataTemp0 = 0.0;
        simd<fp16, 128> kDataTemp1 = 0.0;
        simd<fp16, 128> kDataTemp2 = 0.0;
        
        if (l * loopStep + vv * 16 <= kv_len - 16)
        {
            kDataTemp0.template bit_cast_view<uint32_t>().select<64, 1>(0) = __ESIMD_ENS::lsc_gather<
                    uint32_t,
                    4,
                    __ESIMD_ENS::lsc_data_size::u32,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached,
                    16,
                    uint32_t
                    >((uint32_t*)kState, scatteredOffsetK1 * sizeof(fp16));

        }
        else if (l * loopStep + vv * 16 < kv_len)
        {
            uint32_t tailing = kv_len - l * loopStep - vv * 16;
            //simd_mask<8> read_mask = base8Offset < tailing;
            simd<uint32_t, 16> tailScatteredOffsetK1 = scatteredOffsetK1;
            const uint32_t sK1 = scatteredOffsetK1[0];
            tailScatteredOffsetK1.merge(sK1, base16Offset >= tailing);
            kDataTemp0.template bit_cast_view<uint32_t>().select<64, 1>(0) = __ESIMD_ENS::lsc_gather<
                    uint32_t,
                    4,
                    __ESIMD_ENS::lsc_data_size::u32,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached,
                    16,
                    uint32_t
                    >((uint32_t*)kState, tailScatteredOffsetK1 * sizeof(fp16));
        }

        slm_block_store<fp16, 128>(slmOffset_shuffled_KData + (vv * 16 * 80 + hh * 16 * 8)*sizeof(fp16), kDataTemp0);

        if (hh == 7)
        {
            if (l * loopStep + vv * 16 <= kv_len - 16)
            {
                kDataTemp1.template bit_cast_view<uint32_t>().select<64, 1>(0) = __ESIMD_ENS::lsc_gather<
                        uint32_t,
                        4,
                        __ESIMD_ENS::lsc_data_size::u32,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached,
                        16,
                        uint32_t
                        >((uint32_t*)kState, scatteredOffsetK2 * sizeof(fp16));

            }
            else if (l * loopStep + vv * 16 < kv_len)
            {
                uint32_t tailing = kv_len - l * loopStep - vv * 16;
                //simd_mask<8> read_mask = base8Offset < tailing;
                simd<uint32_t, 16> tailScatteredOffsetK2 = scatteredOffsetK2;
                const uint32_t sK2 = scatteredOffsetK2[0];
                tailScatteredOffsetK2.merge(sK2, base16Offset >= tailing);
                kDataTemp1.template bit_cast_view<uint32_t>().select<64, 1>(0) = __ESIMD_ENS::lsc_gather<
                        uint32_t,
                        4,
                        __ESIMD_ENS::lsc_data_size::u32,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached,
                        16,
                        uint32_t
                        >((uint32_t*)kState, tailScatteredOffsetK2 * sizeof(fp16));
            }
            slm_block_store<fp16, 128>(slmOffset_shuffled_KData + (vv * 16 * 80 + 8 * 16 * 8)*sizeof(fp16), kDataTemp1);
            slm_block_store<fp16, 128>(slmOffset_shuffled_KData + (vv * 16 * 80 + 9 * 16 * 8)*sizeof(fp16), kDataTemp2);
        }
        // simd<fp16, 128> vDataTemp0 = block_load<fp16, 128>((fp16 *)vState + offsetV);
        // simd<fp16, 128> vDataTemp1 = block_load<fp16, 128>((fp16 *)vState + offsetV + 32 * 128);
        simd<fp16, 80> vDataTemp0 = 0.0;
        simd<fp16, 80> vDataTemp1 = 0.0;
        if (l * loopStep + 2*localLinearId < kv_len)
        {
            vDataTemp0.select<64, 1>(0) = block_load<fp16, 64>((fp16*)vState + directOffsetV);
            vDataTemp0.select<8, 1>(64) = block_load<fp16, 8>((fp16*)vState + directOffsetV + 64);
        }
        if (l * loopStep + 2*localLinearId + 1 < kv_len)
        {
            vDataTemp1.select<64, 1>(0) = block_load<fp16, 64>((fp16*)vState + directOffsetV + KVHead * 72);
            vDataTemp1.select<8, 1>(64) = block_load<fp16, 8>((fp16*)vState + directOffsetV +KVHead * 72 + 64);
        }

        if (mask)
        {
            maskData0 = block_load<float, 64>((float *)mask + maskOffset0);
            maskData1 = block_load<float, 64>((float *)mask + maskOffset1);
            maskData2 = block_load<float, 64>((float *)mask + maskOffset2);
            maskData3 = block_load<float, 64>((float *)mask + maskOffset3);
        }

        scatteredOffsetK1 += loopStep * KVHead * 72;
        scatteredOffsetK2 += loopStep * KVHead * 72;
        directOffsetV += loopStep * KVHead * 72;
        maskOffset0 += 64;
        maskOffset1 += 64;
        maskOffset2 += 64;
        maskOffset3 += 64;

        //barrier();
        
        barrier();
        // slm_block_store<fp16, 128>(slmOffset_shuffled_VData + (vbidx * 64 * 8 + viidx * 16 * 8) * sizeof(fp16), vDataTemp.select<128, 1>(0));
        // slm_block_store<fp16, 128>(slmOffset_shuffled_VData + (vbidx * 64 * 8 + 32 * 8 + viidx * 16 * 8) * sizeof(fp16), vDataTemp.select<128, 1>(128));
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            simd<fp16, 32> temp;
            temp.select<16, 2>(0) = vDataTemp0.select<16, 1>(i * 16);
            temp.select<16, 2>(1) = vDataTemp1.select<16, 1>(i * 16);
            slm_block_store<fp16, 32>(slmOffset_shuffled_VData + sizeof(fp16)*(localLinearId * 2 * 16 + i * 16 * 64), temp);
        }
        slm_block_store<fp16, 8>(slmOffset_shuffled_VData + sizeof(fp16)*(64 * 64 + localLinearId * 2 * 8), vDataTemp0.select<8, 1>(64));
        slm_block_store<fp16, 8>(slmOffset_shuffled_VData + sizeof(fp16)*(64 * 64 + localLinearId * 2 * 8 + 8), vDataTemp1.select<8, 1>(64));
        
        
        simd<float, 128> acc = 0.0;
        simd<float, 128> acc2 = 0.0;
        if (true)
        {
            kData0 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 80 + 0 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData0, qData0);
            acc2 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc2, kData0, qData5);
            kData1 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 80 + 1 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData1, qData1);
            acc2 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc2, kData1, qData6);
            kData2 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 80 + 2 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData2, qData2);
            acc2 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc2, kData2, qData7);
            kData3 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 80 + 3 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData3, qData3);
            acc2 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc2, kData3, qData8);
            kData4 = slm_block_load<fp16, 256>(slmOffset_shuffled_KData + (vv * 16 * 80 + 4 * 16 * 16) * sizeof(fp16));
            acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, kData4, qData4);
            acc2 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc2, kData4, qData9);
        }
// #pragma unroll
//             for (int i = 0; i < 8; i++)
//             {
//                 slm_block_store<float, 8>(slmOffset_shuffled_QKData + sizeof(float)*((hh * 8 + i) * 64 + vv * 8), acc.select<8, 1>(i * 8));
//             }
        slm_block_store<float, 128>(slmOffset_shuffled_QKData + sizeof(float)*(localLinearId * 128), acc);
        slm_block_store<float, 128>(slmOffset_shuffled_QKData + sizeof(float)*((localLinearId + 32) * 128), acc2);

        
        barrier();
        simd<float, 64> qkrow0;
        simd<float, 64> qkrow1;
        simd<float, 64> qkrow2;
        simd<float, 64> qkrow3;
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            qkrow0.select<16, 1>(i*16) = slm_block_load<float, 16>(slmOffset_shuffled_QKData + sizeof(float) * (i * 16 * 64 + 16 * localLinearId));
            qkrow1.select<16, 1>(i*16) = slm_block_load<float, 16>(slmOffset_shuffled_QKData + sizeof(float) * (i * 16 * 64 + 16 * (localLinearId + 32)));
            qkrow2.select<16, 1>(i*16) = slm_block_load<float, 16>(slmOffset_shuffled_QKData + sizeof(float) * (64 * 64 + i * 16 * 64 + 16 * localLinearId));
            qkrow3.select<16, 1>(i*16) = slm_block_load<float, 16>(slmOffset_shuffled_QKData + sizeof(float) * (64 * 64 + i * 16 * 64 + 16 * (localLinearId + 32)));
        }

        qkrow0 = qkrow0 * attn_scale;
        qkrow1 = qkrow1 * attn_scale;
        qkrow2 = qkrow2 * attn_scale;
        qkrow3 = qkrow3 * attn_scale;
        //simd<float, 64> qkrow = slm_block_load<float, 64>(slmOffset_shuffled_QKData + localLinearId * 64 * sizeof(float));
        simd<uint32_t, 64> k_idx = base64Offset + l * loopStep;
        qkrow0 = qkrow0 + maskData0;
        qkrow1 = qkrow1 + maskData1;
        qkrow2 = qkrow2 + maskData2;
        qkrow3 = qkrow3 + maskData3;
        qkrow0.merge(FP32_MIN, k_idx >= kv_len);
        qkrow1.merge(FP32_MIN, k_idx >= kv_len);
        qkrow2.merge(FP32_MIN, k_idx >= kv_len);
        qkrow3.merge(FP32_MIN, k_idx >= kv_len);
        float curmax0 = hmax<float, float, 64>(qkrow0);
        float curmax1 = hmax<float, float, 64>(qkrow1);
        float curmax2 = hmax<float, float, 64>(qkrow2);
        float curmax3 = hmax<float, float, 64>(qkrow3);
        curmax0 = curmax0>prevmax0?curmax0:prevmax0;
        curmax1 = curmax1>prevmax1?curmax1:prevmax1;
        curmax2 = curmax2>prevmax2?curmax2:prevmax2;
        curmax3 = curmax3>prevmax3?curmax3:prevmax3;
        qkrow0 = qkrow0 - curmax0;
        qkrow1 = qkrow1 - curmax1;
        qkrow2 = qkrow2 - curmax2;
        qkrow3 = qkrow3 - curmax3;
        qkrow0 = sycl::ext::intel::esimd::pow<float, 64, float>(2.718f, qkrow0);
        qkrow1 = sycl::ext::intel::esimd::pow<float, 64, float>(2.718f, qkrow1);
        qkrow2 = sycl::ext::intel::esimd::pow<float, 64, float>(2.718f, qkrow2);
        qkrow3 = sycl::ext::intel::esimd::pow<float, 64, float>(2.718f, qkrow3);

        
        float qkrowSum0 = sycl::ext::intel::esimd::detail::sum<float, float, 64>(qkrow0);
        float qkrowSum1 = sycl::ext::intel::esimd::detail::sum<float, float, 64>(qkrow1);
        float qkrowSum2 = sycl::ext::intel::esimd::detail::sum<float, float, 64>(qkrow2);
        float qkrowSum3 = sycl::ext::intel::esimd::detail::sum<float, float, 64>(qkrow3);
        float compensate0 = (l == 0)? 1.0 : sycl::ext::intel::esimd::exp(prevmax0 - curmax0);
        float compensate1 = (l == 0)? 1.0 : sycl::ext::intel::esimd::exp(prevmax1 - curmax1);
        float compensate2 = (l == 0)? 1.0 : sycl::ext::intel::esimd::exp(prevmax2 - curmax2);
        float compensate3 = (l == 0)? 1.0 : sycl::ext::intel::esimd::exp(prevmax3 - curmax3);
        accSoftmax0 = accSoftmax0 * compensate0 + qkrowSum0;
        accSoftmax1 = accSoftmax1 * compensate1 + qkrowSum1;
        accSoftmax2 = accSoftmax2 * compensate2 + qkrowSum2;
        accSoftmax3 = accSoftmax3 * compensate3 + qkrowSum3;
        slm_block_store<float, 1>(slmOffset_compensates + localLinearId * sizeof(float), compensate0);
        slm_block_store<float, 1>(slmOffset_compensates + (localLinearId + 32) * sizeof(float), compensate1);
        slm_block_store<float, 1>(slmOffset_compensates + (localLinearId + 64) * sizeof(float), compensate2);
        slm_block_store<float, 1>(slmOffset_compensates + (localLinearId + 96) * sizeof(float), compensate3);
        prevmax0 = curmax0;
        prevmax1 = curmax1;
        prevmax2 = curmax2;
        prevmax3 = curmax3;

        accResultTail0 = accResultTail0 * compensate0;
        accResultTail1 = accResultTail1 * compensate1;
        accResultTail2 = accResultTail2 * compensate2;
        accResultTail3 = accResultTail3 * compensate3;

       
        simd<fp16, 64> qktemp0 = qkrow0;
        simd<fp16, 64> qktemp1 = qkrow1;
        simd<fp16, 64> qktemp2 = qkrow2;
        simd<fp16, 64> qktemp3 = qkrow3;
        int sbidx = localLinearId >> 3;
        int siidx = localLinearId & 0x7;

#pragma unroll
        for (int i = 0; i < 4; i ++)
        {
            slm_block_store<fp16, 16>(slmOffset_shuffled_SData + (sbidx * 8 * 64 + i * 8 * 16 + siidx * 16) * sizeof(fp16), qktemp0.select<16, 1>(i * 16));
            slm_block_store<fp16, 16>(slmOffset_shuffled_SData + (32 * 64 + sbidx * 8 * 64 + i * 8 * 16 + siidx * 16) * sizeof(fp16), qktemp1.select<16, 1>(i * 16));
            slm_block_store<fp16, 16>(slmOffset_shuffled_SData + (64 * 64 + sbidx * 8 * 64 + i * 8 * 16 + siidx * 16) * sizeof(fp16), qktemp2.select<16, 1>(i * 16));
            slm_block_store<fp16, 16>(slmOffset_shuffled_SData + (96 * 64 + sbidx * 8 * 64 + i * 8 * 16 + siidx * 16) * sizeof(fp16), qktemp3.select<16, 1>(i * 16));
        }
        
        barrier();

        {
            // compensate
            simd<float, 8> compensates = slm_block_load<float, 8>(slmOffset_compensates + hh * 8 * sizeof(float));
            simd<float, 8> compensates2 = slm_block_load<float, 8>(slmOffset_compensates + (64 + hh * 8) * sizeof(float));

            simd<float, 128> acc0 = 0.0;
            simd<float, 128> acc1 = 0.0;
            vData0 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 16 * 64 + 0 * 16 * 16) * sizeof(fp16));
            sData0 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 0 * 8 * 16 * sizeof(fp16));
            vData4 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (64 * 64 + 0 * 8 * 16) * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData0, sData0);
            accResult0.select<16, 1>(0 * 16) = accResult0.select<16, 1>(0 * 16) * compensates[0];
            accResult0.select<16, 1>(1 * 16) = accResult0.select<16, 1>(1 * 16) * compensates[1];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                accResultTail0 += vData4.select<8, 1>(j * 8) * qktemp0[j];
                accResultTail1 += vData4.select<8, 1>(j * 8) * qktemp1[j];
            }
            sData4 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + 64 * 64 * sizeof(fp16) + hh * 8 * 64 * sizeof(fp16) + 0 * 8 * 16 * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData0, sData4);
            accResult1.select<16, 1>(0 * 16) = accResult1.select<16, 1>(0 * 16) * compensates2[0];
            accResult1.select<16, 1>(1 * 16) = accResult1.select<16, 1>(1 * 16) * compensates2[1];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                accResultTail2 += vData4.select<8, 1>(j * 8) * qktemp2[j];
                accResultTail3 += vData4.select<8, 1>(j * 8) * qktemp3[j];
            }


            vData1 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 16 * 64 + 1 * 16 * 16) * sizeof(fp16));
            sData1 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 1 * 8 * 16 * sizeof(fp16));
            vData5 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (64 * 64 + 1 * 8 * 16) * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData1, sData1);
            accResult0.select<16, 1>(2 * 16) = accResult0.select<16, 1>(2 * 16) * compensates[2];
            accResult0.select<16, 1>(3 * 16) = accResult0.select<16, 1>(3 * 16) * compensates[3];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                accResultTail0 += vData5.select<8, 1>(j * 8) * qktemp0[j + 16];
                accResultTail1 += vData5.select<8, 1>(j * 8) * qktemp1[j + 16];
            }
            sData5 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + 64 * 64 * sizeof(fp16) + hh * 8 * 64 * sizeof(fp16) + 1 * 8 * 16 * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData1, sData5);
            accResult1.select<16, 1>(2 * 16) = accResult1.select<16, 1>(2 * 16) * compensates2[2];
            accResult1.select<16, 1>(3 * 16) = accResult1.select<16, 1>(3 * 16) * compensates2[3];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                accResultTail2 += vData5.select<8, 1>(j * 8) * qktemp2[j + 16];
                accResultTail3 += vData5.select<8, 1>(j * 8) * qktemp3[j + 16];
            }


            vData2 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 16 * 64 + 2 * 16 * 16) * sizeof(fp16));
            sData2 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 2 * 8 * 16 * sizeof(fp16));
            vData6 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (64 * 64 + 2 * 8 * 16) * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData2, sData2);
            accResult0.select<16, 1>(4 * 16) = accResult0.select<16, 1>(4 * 16) * compensates[4];
            accResult0.select<16, 1>(5 * 16) = accResult0.select<16, 1>(5 * 16) * compensates[5];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                accResultTail0 += vData6.select<8, 1>(j * 8) * qktemp0[j + 32];
                accResultTail1 += vData6.select<8, 1>(j * 8) * qktemp1[j + 32];
            }
            sData6 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + 64 * 64 * sizeof(fp16) + hh * 8 * 64 * sizeof(fp16) + 2 * 8 * 16 * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData2, sData6);
            accResult1.select<16, 1>(4 * 16) = accResult1.select<16, 1>(4 * 16) * compensates2[4];
            accResult1.select<16, 1>(5 * 16) = accResult1.select<16, 1>(5 * 16) * compensates2[5];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                accResultTail2 += vData6.select<8, 1>(j * 8) * qktemp2[j + 32];
                accResultTail3 += vData6.select<8, 1>(j * 8) * qktemp3[j + 32];
            }

            vData3 = slm_block_load<fp16, 256>(slmOffset_shuffled_VData + (vv * 16 * 64 + 3 * 16 * 16) * sizeof(fp16));
            sData3 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + hh * 8 * 64 * sizeof(fp16) + 3 * 8 * 16 * sizeof(fp16));
            vData7 = slm_block_load<fp16, 128>(slmOffset_shuffled_VData + (64 * 64 + 3 * 8 * 16) * sizeof(fp16));
            acc0 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc0, vData3, sData3);
            accResult0.select<16, 1>(6 * 16) = accResult0.select<16, 1>(6 * 16) * compensates[6];
            accResult0.select<16, 1>(7 * 16) = accResult0.select<16, 1>(7 * 16) * compensates[7];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                accResultTail0 += vData7.select<8, 1>(j * 8) * qktemp0[j + 48];
                accResultTail1 += vData7.select<8, 1>(j * 8) * qktemp1[j + 48];
            }
            sData7 = slm_block_load<fp16, 128>(slmOffset_shuffled_SData + 64 * 64 * sizeof(fp16) + hh * 8 * 64 * sizeof(fp16) + 3 * 8 * 16 * sizeof(fp16));
            acc1 = xmx::dpas<8, 8, float, float, fp16, fp16>(acc1, vData3, sData7);
            accResult1.select<16, 1>(6 * 16) = accResult1.select<16, 1>(6 * 16) * compensates2[6];
            accResult1.select<16, 1>(7 * 16) = accResult1.select<16, 1>(7 * 16) * compensates2[7];
#pragma unroll
            for (int j = 0; j < 16; j++)
            {
                accResultTail2 += vData7.select<8, 1>(j * 8) * qktemp2[j + 48];
                accResultTail3 += vData7.select<8, 1>(j * 8) * qktemp3[j + 48];
            }

            accResult0 = accResult0 + acc0;
            accResult1 = accResult1 + acc1;
        }
    }

    slm_block_store<float, 1>(slmOffset_accSoftMax + localLinearId*sizeof(float), accSoftmax0);
    slm_block_store<float, 1>(slmOffset_accSoftMax + (32 + localLinearId)*sizeof(float), accSoftmax1);
    slm_block_store<float, 1>(slmOffset_accSoftMax + (64 + localLinearId)*sizeof(float), accSoftmax2);
    slm_block_store<float, 1>(slmOffset_accSoftMax + (96 + localLinearId)*sizeof(float), accSoftmax3);
    barrier();
    simd<float, 8> accsoftmaxs = slm_block_load<float, 8>(slmOffset_accSoftMax + hh * 8 * sizeof(float));
    simd<float, 8> accsoftmaxs2 = slm_block_load<float, 8>(slmOffset_accSoftMax + (64 + hh * 8) * sizeof(float));
    float accsoftmaxTail0 = accSoftmax0;
    float accsoftmaxTail1 = accSoftmax1;
    float accsoftmaxTail2 = accSoftmax2;
    float accsoftmaxTail3 = accSoftmax3;

    uint32_t outputOffset = v * 128 * QHead * 72 + h * 72;
    {
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            if (v * 128 + hh * 8 + i < token_len)
            {
                simd<float, 16> temp1;
                if (abs(accsoftmaxs[i]) >  1e-5)
                { 
                    temp1 = accResult0.select<16, 1>(i * 16) / accsoftmaxs[i];
                }
                else
                {
                    temp1 = 0.0;
                }
                block_store<outT, 16>((outT*)out + outputOffset + (hh * 8 + i) * QHead * 72 + vv * 16, temp1);
            }

            if (v * 128 + hh * 8 + i + 64 < token_len)
            {
                simd<float, 16> temp1;
                if (abs(accsoftmaxs2[i]) > 1e-5)
                { 
                    temp1 = accResult1.select<16, 1>(i * 16) / accsoftmaxs2[i];
                }
                else
                {
                    temp1 = 0.0;
                }
                block_store<outT, 16>((outT*)out + outputOffset + (64 + hh * 8 + i) * QHead * 72 + vv * 16, temp1);
            }
        }

        simd<float, 8> temp2;
        if (v * 128 + localLinearId < token_len)
        {
            if (abs(accsoftmaxTail0) > 1e-5)
            {
                temp2 = accResultTail0.select<8, 1>(0) / accsoftmaxTail0;
            }
            else
            {
                temp2 = 0.0;
            }
            block_store<outT, 8>((outT*)out + outputOffset + localLinearId * QHead * 72 + 64, temp2);
        }
        if (v * 128 + localLinearId + 32 < token_len)
        {
            if (abs(accsoftmaxTail1) > 1e-5)
            {
                temp2 = accResultTail1.select<8, 1>(0) / accsoftmaxTail1;
            }
            else
            {
                temp2 = 0.0;
            }
            block_store<outT, 8>((outT*)out + outputOffset + (32 + localLinearId) * QHead * 72 + 64, temp2);
        }
        if (v * 128 + localLinearId + 64 < token_len)
        {
            if (abs(accsoftmaxTail2) > 1e-5)
            {
                temp2 = accResultTail2.select<8, 1>(0) / accsoftmaxTail2;
            }
            else
            {
                temp2 = 0.0;
            }
            block_store<outT, 8>((outT*)out + outputOffset + (64 + localLinearId) * QHead * 72 + 64, temp2);
        }
        if (v * 128 + localLinearId + 96 < token_len)
        {
            if (abs(accsoftmaxTail3) >1e-5)
            {
                temp2 = accResultTail3.select<8, 1>(0) / accsoftmaxTail3;
            }
            else
            {
                temp2 = 0.0;
            }
            block_store<outT, 8>((outT*)out + outputOffset + (96 + localLinearId) * QHead * 72 + 64, temp2);
        }
    }
    
}

bool runSDP_vit_fusion_xmx_simd16(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, unsigned q_precision, unsigned kv_precision, unsigned o_precision, uint8_t* shuffleTt) {
    sycl::event e;

    // const uint32_t SimdWidth = 8;
    // const uint32_t alignedKVLen = (kv_len + 15) / 16 * 16;
    // q->wait();
    // uint8_t* temp1 = new uint8_t[token_len * q_head * 72 * sizeof(fp16)];
    // uint8_t* temp2 = new uint8_t[kv_len * kv_head * 72 * sizeof(fp16)];
    // uint8_t* temp3 = new uint8_t[token_len * q_head * 72 * sizeof(float)];

    // char filename[2048];
    // FILE *fp = nullptr;
    // sprintf_s(filename, "debug.q.%dx%d.f16.bin", token_len, q_head * 72);
    // q->memcpy(temp1, query, token_len * q_head * 72 * sizeof(fp16)).wait();
    // fopen_s(&fp, filename, "wb");
    // fwrite(temp1, 1, token_len * q_head * 72 * sizeof(fp16), fp);
    // fclose(fp);

    // sprintf_s(filename, "debug.k.%dx%d.f16.bin", kv_len, kv_head * 72);
    // q->memcpy(temp2, kCache, kv_len * kv_head * 72 * sizeof(fp16)).wait();
    // fopen_s(&fp, filename, "wb");
    // fwrite(temp2, 1, kv_len * kv_head * 72 * sizeof(fp16), fp);
    // fclose(fp);

    // sprintf_s(filename, "debug.v.%dx%d.f16.bin", kv_len, kv_head * 72);
    // q->memcpy(temp2, vCache, kv_len * kv_head * 72 * sizeof(fp16)).wait();
    // fopen_s(&fp, filename, "wb");
    // fwrite(temp2, 1,  kv_len * kv_head * 72 * sizeof(fp16), fp);
    // fclose(fp);
    try {
        // GQA
        {
            const uint32_t localThread = 32;
            uint32_t vThreadnum =(token_len + 127)/128;
            sycl::range<2> GlobalRange(localThread * q_head, vThreadnum);
            sycl::range<2> LocalRange(localThread, 1);
            sycl::nd_range<2> Range(GlobalRange, LocalRange);

            if (q_precision == 0 && o_precision == 0)
            {
                e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        gqa_mat_kernel_hidden72_simd16<float, float>(query, kCache, vCache, mask, outputs, token_len, kv_len, kv_head, q_head, ndi);
                    });
                    });
            }
            else if (q_precision == 0 && o_precision == 1)
            {
                e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        gqa_mat_kernel_hidden72_simd16<float, fp16>(query, kCache, vCache, mask, outputs, token_len, kv_len, kv_head, q_head, ndi);
                    });
                    });
            }
            else if (q_precision == 1 && o_precision == 0)
            {
                e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        gqa_mat_kernel_hidden72_simd16<fp16, float>(query, kCache, vCache, mask, outputs, token_len, kv_len, kv_head, q_head, ndi);
                    });
                    });
            }
            else if (q_precision == 1 && o_precision == 1)
            {
                e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        gqa_mat_kernel_hidden72_simd16<fp16, fp16>(query, kCache, vCache, mask, outputs, token_len, kv_len, kv_head, q_head, ndi);
                    });
                    });
            }
        }
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    // q->wait();
    // sprintf_s(filename, "debug.out.%dx%d.f16.bin", token_len, q_head * 72);
    // q->memcpy(temp3, outputs, token_len * q_head * 72 * sizeof(float)).wait();
    // fopen_s(&fp, filename, "wb");
    // fwrite(temp3, 1, token_len * q_head * 72 * sizeof(float), fp);
    // fclose(fp);

    return true;
}

