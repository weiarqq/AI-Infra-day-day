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

extern "C" bool runGQA_vec_masked_fusion(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale);


template<uint32_t LocalThread, uint32_t Step>
ESIMD_INLINE void gqa_kernel_hidden128_masked(uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* mask, uint8_t* out, int kv_len, uint32_t KVHead, uint32_t QHead, float attn_scale, nd_item<2>& ndi)
{
    __ESIMD_NS::slm_init(LocalThread * 128 * sizeof(float) + LocalThread * sizeof(float) + LocalThread * sizeof(float));
    constexpr uint32_t slmOffset_qkvResult = 0;
    constexpr uint32_t slmOffset_softmaxSum = LocalThread * 128 * sizeof(float);
    constexpr uint32_t slmOffset_maxqk = LocalThread * 128 * sizeof(float) + LocalThread * sizeof(float);

    // attn_scale passed as parameter

    simd<float, 128> qData;
    simd<float, 128> accResult = 0.0;
    simd<fp16, Step*128> kData;
    simd<fp16, Step*128> vData;
    float maxQK = FP32_MIN;
    float accSoftMax = 0.0;

    int loopStep = LocalThread*Step;
    int loopNum = (kv_len + loopStep - 1) / loopStep;

    int h = ndi.get_group(0);
    int t = ndi.get_group(1);
    int token_len = ndi.get_group_range(1);
    int localLinearId = ndi.get_local_linear_id();

    uint32_t offsetQ = h * 128 + t * QHead * 128;
    uint32_t offsetK = h * KVHead / QHead * 128;
    uint32_t offsetV = h * KVHead / QHead * 128;
    uint32_t offsetOutput = h * 128 + t * QHead * 128;

    // Load Q data
    qData.select<64, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        float,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)qState + offsetQ);
    qData.select<64, 1>(64) =
        __ESIMD_ENS::lsc_block_load<
        float,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((float*)qState + offsetQ + 64);

    for (int l = 0; l < loopNum; l++)
    {
        uint32_t mask_byte_offset = t * ((kv_len + 7)/8) + (l * loopStep + localLinearId * Step)/ 8;
        uint32_t mask_bit_offset = (l * loopStep + localLinearId * Step)% 8;
        uint8_t mask_int = mask[mask_byte_offset];

        if (l * loopStep + localLinearId * Step >= kv_len)
        {
            break;
        }
#pragma unroll
        for (int s = 0; s < Step; s++)
        {
            //if (l * loopStep + localLinearId * Step + s < kv_len - token_len + t + 1)
            if ((mask_int & (1 << (s + mask_bit_offset))) == 0)
            {
                kData.template select<128, 1>(s*128) =
                    __ESIMD_ENS::lsc_block_load<
                    fp16,
                    128,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((fp16*)kState + offsetK + (l * loopStep + localLinearId * Step + s) * KVHead * 128);
            }
        }
#pragma unroll
        for (int s = 0; s < Step; s++)
        {
            //if (l * loopStep + localLinearId * Step + s < kv_len - token_len + t + 1)
            if ((mask_int & (1 << (s + mask_bit_offset))) == 0)
            {
                vData.template select<128, 1>(s*128) =
                    __ESIMD_ENS::lsc_block_load<
                    fp16,
                    128,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached>((fp16*)vState + offsetV + (l * loopStep + localLinearId * Step + s) * KVHead * 128);
            }
        }

#pragma unroll
        for (int s = 0; s < Step ; s++)
        {
            // if (mask_int & (1 << (s + mask_bit_offset)))
            // {
            //     continue;
            // }
            // if (l * loopStep + localLinearId * Step + s >= kv_len - token_len + t + 1)
            if (mask_int & (1 << (s + mask_bit_offset)))
            {
                continue;
            }
            simd<float, 128> temp = qData.select<128, 1>(0) * kData.template select<128, 1>(s * 128);
            float qkResult = sycl::ext::intel::esimd::detail::sum<float, float, 128>(temp) * attn_scale;
            if (qkResult > maxQK)
            {
                float compensate = sycl::ext::intel::esimd::exp(maxQK - qkResult);
                accResult = accResult * compensate + vData.template select<128, 1>(s * 128);
                accSoftMax = accSoftMax * compensate + 1.0;
                maxQK = qkResult;
            }
            else
            {
                float compensate = sycl::ext::intel::esimd::exp(qkResult - maxQK);
                accResult = accResult + compensate * vData.template select<128, 1>(s * 128);
                accSoftMax = accSoftMax + compensate;
            }
        }
    }

    
    
    slm_block_store<float, 1>(slmOffset_maxqk + localLinearId * sizeof(float), maxQK);

    barrier();

    simd<float, LocalThread> maxQKs = slm_block_load<float, LocalThread>(slmOffset_maxqk);
    float globalMaxQK = hmax<float, float, LocalThread>(maxQKs);
    float compensate = sycl::ext::intel::esimd::exp(maxQK - globalMaxQK);
    accResult = accResult * compensate;
    accSoftMax = accSoftMax * compensate;
    slm_block_store<float, 128>(slmOffset_qkvResult + localLinearId * 128 * sizeof(float), accResult.select<128, 1>(0));
    slm_block_store<float, 1>(slmOffset_softmaxSum + localLinearId * sizeof(float), accSoftMax);

    barrier();


    constexpr uint32_t accBlock = LocalThread / 4;
    if (localLinearId < 4)
    {
        accResult = 0.0;
        accSoftMax = 0.0;
#pragma unroll
        for (int i = 0; i < accBlock; i ++)
        {
            accResult = accResult + slm_block_load<float, 128>(slmOffset_qkvResult + (localLinearId * accBlock + i) * 128 * sizeof(float));
            accSoftMax = accSoftMax + slm_block_load<float, 1>(slmOffset_softmaxSum + (localLinearId * accBlock + i) * sizeof(float));
        }

        slm_block_store<float, 128>(slmOffset_qkvResult + localLinearId * accBlock * 128 * sizeof(float), accResult.select<128, 1>(0));
        slm_block_store<float, 1>(slmOffset_softmaxSum + localLinearId * accBlock * sizeof(float), accSoftMax);
    }

    barrier();

    if (localLinearId == 0)
    {
        accResult = 0.0;
        accSoftMax = 0.0;
#pragma unroll
        for (int i = 0; i < 4; i ++)
        {
            accResult = accResult + slm_block_load<float, 128>(slmOffset_qkvResult + i * accBlock * 128 * sizeof(float));
            accSoftMax = accSoftMax + slm_block_load<float, 1>(slmOffset_softmaxSum + i * accBlock * sizeof(float));
        }

        if (accSoftMax > 0)
        {
            accResult = accResult / accSoftMax;
        }
        else
        {
            accResult = 0;
        }

        block_store<float, 128>((float*)out + offsetOutput, accResult.select<128, 1>(0));
    }

}


bool runGQA_vec_masked_fusion(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale)
{
    sycl::event e;
    try {
        const uint32_t localThread = 32;
        sycl::range<2> GlobalRange(localThread * q_head, token_len);
        sycl::range<2> LocalRange(localThread, 1);
        sycl::nd_range<2> Range(GlobalRange, LocalRange);

        e = q->submit([&](handler& cgh) {
            cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                gqa_kernel_hidden128_masked<localThread, 4>(query, kCache, vCache, mask, outputs, kv_len, kv_head, q_head, attn_scale, ndi);
              });
            });
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    return true;
}
