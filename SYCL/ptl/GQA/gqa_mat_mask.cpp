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
#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

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

extern "C" bool runGQA_mat_fusion(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale, uint8_t* shuffleTt);

ESIMD_INLINE void gqa_mat_kernel_hidden128(uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* out, int token_len, int kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale, nd_item<2>& ndi)
{
    float matMulQuantCoeff = attn_scale; // 1.0f / sqrt(128.0f);
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __ESIMD_NS::slm_init(16 * 128 * 2 * sizeof(fp16) + 64 * 64 * sizeof(float) + 64 * 16 * sizeof(fp16) + 64 * sizeof(float) + 64 * sizeof(float));
    constexpr uint32_t slmOffsetBaseK = 0;
    constexpr uint32_t slmOffsetBaseV = 16 * 128 * sizeof(fp16);
    constexpr uint32_t slmOffsetBaseKq = 2 * 16 * 128 * sizeof(fp16);
    constexpr uint32_t slmOffsetBaseSoftMax = 16 * 128 * 2 * sizeof(fp16) + 64 * 64 * sizeof(float);
    constexpr uint32_t slmOffsetBaseCompensates = 16 * 128 * 2 * sizeof(fp16) + 64 * 64 * sizeof(float) + 64 * 16 * sizeof(fp16);
    constexpr uint32_t slmOffsetBaseAccSoftmax = 16 * 128 * 2 * sizeof(fp16) + 64 * 64 * sizeof(float) + 64 * 16 * sizeof(fp16) + 64 * sizeof(float);
    int localLinearId = ndi.get_local_linear_id();
    int hh = localLinearId & 0x3;
    int vv = localLinearId >> 2;
    int blockidx = localLinearId >> 3;
    int inneridx = localLinearId & 0x7;
    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int startActivationIdx = kv_len - token_len;
    int kvSeqOutLoopCount = 64 * h + 64 + startActivationIdx;
    kvSeqOutLoopCount = (kvSeqOutLoopCount + 15) / 16;

    simd<fp16, 8 * 32> fp16QState;
    simd<float, 16 * 16> finalOutput = 0;
    simd<float, 16> softMaxDividor = 0;
    simd<float, 2> prevMax = FP32_MIN;
    simd<fp16, 16 * 16> tempOutput;
    simd<fp16, 16 * 16> tempInputA;
    simd<fp16, 16 * 16> tempInputB;

    simd<uint32_t, 16> simdBase16(baseOffsetInc16);

    unsigned int outputOffset = v * 128 + h * 64 * q_head * 128 + hh * 16 * q_head * 128 + vv * 16;
    unsigned int offsetQ = v * 128 + h * 64 * q_head * 128 + vv * 8 * q_head * 128 + hh * 32;
    unsigned int kvheadidx = v * kv_head / q_head;
    unsigned int offsetKBase = kvheadidx * 128 + localLinearId * 4;
    int vReadBlk = localLinearId >> 1;
    int vReadInn = localLinearId & 0x01;
    unsigned int offsetV = kvheadidx * 128 + vReadBlk * kv_head * 128 + vReadInn * 64;

    simd<uint32_t, 16> offsetKScattered = offsetKBase + simdBase16 * kv_head * 128;

#pragma unroll
    for (int k = 0; k < 8; k++) {
        finalOutput.select<32, 1>(k * 32) = __ESIMD_ENS::lsc_block_load<
            float,
            32,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((float*)qState + offsetQ);
        offsetQ += q_head * 128;
    }
    fp16QState.select<256, 1>(0) = finalOutput.select<256, 1>(0);
    finalOutput = 0.0;

    for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
        auto fp16KState = tempInputA.select<64, 1>(0);
        fp16KState.template bit_cast_view<uint32_t>().template select<32, 1>(0) = __ESIMD_ENS::lsc_gather<
            uint32_t,
            2,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t>((uint32_t*)kState, offsetKScattered * sizeof(fp16));

        auto fp16VState = tempInputB.select<64, 1>(0);
        fp16VState = __ESIMD_ENS::lsc_block_load<
            fp16,
            64,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((fp16*)vState + offsetV);

        auto shuffledFp16KState = tempInputA.select<64, 1>(64);
        shuffledFp16KState.select<16, 1>(0) = fp16KState.select<16, 2>(0);
        shuffledFp16KState.select<16, 1>(16) = fp16KState.select<16, 2>(1);
        shuffledFp16KState.select<16, 1>(32) = fp16KState.select<16, 2>(32);
        shuffledFp16KState.select<16, 1>(48) = fp16KState.select<16, 2>(33);

        slm_block_store<fp16, 64>(slmOffsetBaseK + localLinearId * 64 * sizeof(fp16), shuffledFp16KState.select<64, 1>(0));

        barrier();

        auto fp16KData = tempInputA.select<256, 1>(0);
        fp16KData.select<256, 1>(0) = slm_block_load<fp16, 256>(slmOffsetBaseK + hh * 16 * 32 * sizeof(fp16));

        auto fp32tempOutput = tempOutput.template bit_cast_view<float>().select<128, 1>(0);
        fp32tempOutput = 0.0;
#pragma unroll
        for (int t = 0; t < 8; t++) {
#pragma unroll
            for (int ll = 0; ll < 8; ll++) {
                simd<fp16, 16> temp = 0.0;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    temp += fp16KData.select<16, 1>((ll * 2 + kk) * 16) * fp16QState[t * 32 + ll * 2 + kk];
                }
                fp32tempOutput.select<16, 1>(16 * t) = fp32tempOutput.select<16, 1>(16 * t) + temp;
            }
        }

        fp16KData.select<256, 1>(0) = slm_block_load<fp16, 256>(slmOffsetBaseK + (hh * 16 * 32 + 256) * sizeof(fp16));

#pragma unroll
        for (int t = 0; t < 8; t++) {
#pragma unroll
            for (int ll = 0; ll < 8; ll++) {
                simd<fp16, 16> temp = 0.0;
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    temp += fp16KData.select<16, 1>((ll * 2 + kk) * 16) * fp16QState[t * 32 + ll * 2 + kk + 16];
                }
                fp32tempOutput.select<16, 1>(16 * t) = fp32tempOutput.select<16, 1>(16 * t) + temp;
            }
        }

#pragma unroll
        for (int t = 0; t < 8; t++) {
            slm_block_store<float, 16>(slmOffsetBaseKq + vv * 8 * 64 * sizeof(float) + t * 16 * 4 * sizeof(float) + hh * 16 * sizeof(float), fp32tempOutput.select<16, 1>(16 * t));
        }
        barrier();

        auto fp32KQResult = tempInputA.template bit_cast_view<float>().select<128, 1>(0);
        fp32KQResult = slm_block_load<float, 128>(slmOffsetBaseKq + localLinearId * 128 * sizeof(float));

#pragma unroll
        for (int t = 0; t < 2; t++) {
            fp32KQResult.select<32, 1>(64 * t) += fp32KQResult.select<32, 1>(64 * t + 32);
            fp32KQResult.select<16, 1>(64 * t) += fp32KQResult.select<16, 1>(64 * t + 16);
        }

#pragma unroll
        for (int t = 0; t < 2; t++) {
            fp32KQResult.select<16, 1>(16 * t) = fp32KQResult.select<16, 1>(64 * t) * matMulQuantCoeff;
        }

#pragma unroll
        for (int t = 0; t < 2; t++) {
            simd<uint32_t, 16> k_idx = simdBase16 + loopIdx * 16;
            uint32_t q_mask_idx = startActivationIdx + 64 * h + 2 * localLinearId + t;
            fp32KQResult.select<16, 1>(16 * t).merge(FP32_MIN, k_idx > q_mask_idx);
        }

        auto maxTemp = tempInputB.template bit_cast_view<float>().select<32, 1>(64);
        maxTemp.select<16, 2>(0) = fp32KQResult.select<16, 1>(0);
        maxTemp.select<16, 2>(1) = fp32KQResult.select<16, 1>(16);

        maxTemp.select<16, 1>(0) = max<float, 16, float>(maxTemp.select<16, 1>(0), maxTemp.select<16, 1>(16));
        maxTemp.select<8, 1>(0) = max<float, 8, float>(maxTemp.select<8, 1>(0), maxTemp.select<8, 1>(8));
        maxTemp.select<4, 1>(0) = max<float, 4, float>(maxTemp.select<4, 1>(0), maxTemp.select<4, 1>(4));
        maxTemp.select<2, 1>(0) = max<float, 2, float>(maxTemp.select<2, 1>(0), maxTemp.select<2, 1>(2));
        maxTemp.select<2, 1>(0) = max<float, 2, float>(maxTemp.select<2, 1>(0), prevMax.select<2, 1>(0));

        if (loopIdx == 0) {
            prevMax = maxTemp.select<2, 1>(0);
        }

        simd<float, 2> compensates = pow<float, 2, float>(2.718f, prevMax - maxTemp.select<2, 1>(0));
        slm_block_store<float, 2>(slmOffsetBaseCompensates + localLinearId * 2 * sizeof(float), compensates);
        prevMax = maxTemp.select<2, 1>(0);

#pragma unroll
        for (int t = 0; t < 2; t++) {
            fp32KQResult.select<16, 1>(t * 16) = fp32KQResult.select<16, 1>(t * 16) - maxTemp[t];
        }
        fp32KQResult.select<32, 1>(0) = pow<float, 32, float>(2.718f, fp32KQResult.select<32, 1>(0));
        simd<fp16, 32> fp16SoftmaxResult = fp32KQResult.select<32, 1>(0);
        slm_block_store<fp16, 32>(slmOffsetBaseSoftMax + sizeof(fp16) * localLinearId * 2 * 16, fp16SoftmaxResult);

        auto sumTemp = tempInputB.template bit_cast_view<float>().select<32, 1>(96);
        sumTemp.select<16, 2>(0) = fp32KQResult.select<16, 1>(0);
        sumTemp.select<16, 2>(1) = fp32KQResult.select<16, 1>(16);
        sumTemp.select<16, 1>(0) = sumTemp.select<16, 1>(0) + sumTemp.select<16, 1>(16);
        sumTemp.select<8, 1>(0) = sumTemp.select<8, 1>(0) + sumTemp.select<8, 1>(8);
        sumTemp.select<4, 1>(0) = sumTemp.select<4, 1>(0) + sumTemp.select<4, 1>(4);
        sumTemp.select<2, 1>(0) = sumTemp.select<2, 1>(0) + sumTemp.select<2, 1>(2);
        // simd<float, 2> sumTemp;
        // sumTemp[0] = sycl::ext::intel::esimd::detail::sum<float, float, 16>(fp32KQResult.select<16, 1>(0));
        // sumTemp[1] = sycl::ext::intel::esimd::detail::sum<float, float, 16>(fp32KQResult.select<16, 1>(16));

        simd<float, 2> accSoftmax = 0.0;
        if (loopIdx > 0) {
            accSoftmax = slm_block_load<float, 2>(slmOffsetBaseAccSoftmax + localLinearId * 2 * sizeof(float));
        }
        accSoftmax = accSoftmax * compensates;
        accSoftmax = accSoftmax + sumTemp.select<2, 1>(0);

        slm_block_store<float, 2>(slmOffsetBaseAccSoftmax + localLinearId * 2 * sizeof(float), accSoftmax);

#pragma unroll
        for (int k = 0; k < 4; k++) {
            slm_block_store<fp16, 16>(slmOffsetBaseV + (vReadBlk * 16 + (vReadInn * 4 + k) * 16 * 16) * sizeof(fp16), fp16VState.select<16, 1>(16 * k));
        }

        barrier();

        auto fp16SoftmaxData = tempOutput.select<256, 1>(0);
        fp16SoftmaxData.select<256, 1>(0) = slm_block_load<fp16, 256>(slmOffsetBaseSoftMax + hh * 256 * sizeof(fp16));

        simd<float, 16> compensate = slm_block_load<float, 16>(slmOffsetBaseCompensates + hh * 16 * sizeof(float));

#pragma unroll
        for (int t = 0; t < 16; t++) {
            finalOutput.select<16, 1>(t * 16) = finalOutput.select<16, 1>(t * 16) * compensate[t];
        }

        auto fp16VData = tempInputA.select<256, 1>(0);
        fp16VData.select<256, 1>(0) = slm_block_load<fp16, 256>(slmOffsetBaseV + vv * 256 * sizeof(fp16));

        // tempOutput = 0.0;
#pragma unroll
        for (int t = 0; t < 16; t++) {
#pragma unroll
            for (int ll = 0; ll < 1; ll++) {
                simd<fp16, 16> temp = 0.0;
#pragma unroll
                for (int kk = 0; kk < 16; kk++) {
                    temp += fp16VData.select<16, 1>((ll * 16 + kk) * 16) * fp16SoftmaxData[t * 16 + ll * 16 + kk];
                    // tempOutput.select<16, 1>(16 * t) += fp16VData.select<16, 1>(ll * 16) * fp16SoftmaxData[ll * 8 + t];
                }
                finalOutput.select<16, 1>(16 * t) = finalOutput.select<16, 1>(16 * t) + temp;
            }
        }

        // finalOutput = finalOutput + tempOutput;

        offsetKScattered += kv_head * 128 * 16;
        offsetV += kv_head * 128 * 16;
        barrier();
    }

    softMaxDividor = slm_block_load<float, 16>(slmOffsetBaseAccSoftmax + hh * 16 * sizeof(float));
    softMaxDividor = 1.0f / softMaxDividor;

#pragma unroll
    for (int t = 0; t < 16; t++) {
        if (64 * h + 16 * hh + t < token_len) {
            simd<float, 16> temp = finalOutput.select<16, 1>(16 * t);
            temp.select<16, 1>(0) = temp.select<16, 1>(0) * softMaxDividor[t];
            __ESIMD_ENS::lsc_block_store<
                float,
                16,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back>((float*)out + outputOffset + t * q_head * 128, temp.select<16, 1>(0));
        }
    }
}

ESIMD_INLINE void gqa_mat_kernel_hidden128_fullprecision(uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* out, int token_len, int kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale, nd_item<2>& ndi)
{
    float matMulQuantCoeff = attn_scale; // 1.0f / sqrt(128.0f);
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __ESIMD_NS::slm_init(16 * 128 * 2 * sizeof(fp16) + 64 * 64 * sizeof(float) + 64 * 16 * sizeof(fp16) + 64 * sizeof(float) + 64 * sizeof(float));
    constexpr uint32_t slmOffsetBaseK = 0;
    constexpr uint32_t slmOffsetBaseV = 16 * 128 * sizeof(fp16);
    constexpr uint32_t slmOffsetBaseKq = 2 * 16 * 128 * sizeof(fp16);
    constexpr uint32_t slmOffsetBaseSoftMax = 16 * 128 * 2 * sizeof(fp16) + 64 * 64 * sizeof(float);
    constexpr uint32_t slmOffsetBaseCompensates = 16 * 128 * 2 * sizeof(fp16) + 64 * 64 * sizeof(float) + 64 * 16 * sizeof(fp16);
    constexpr uint32_t slmOffsetBaseAccSoftmax = 16 * 128 * 2 * sizeof(fp16) + 64 * 64 * sizeof(float) + 64 * 16 * sizeof(fp16) + 64 * sizeof(float);
    int localLinearId = ndi.get_local_linear_id();
    int hh = localLinearId & 0x3;
    int vv = localLinearId >> 2;
    int blockidx = localLinearId >> 3;
    int inneridx = localLinearId & 0x7;
    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int startActivationIdx = kv_len - token_len;
    int kvSeqOutLoopCount = 64 * h + 64 + startActivationIdx;
    kvSeqOutLoopCount = (kvSeqOutLoopCount + 15) / 16;

    simd<fp16, 8 * 32> fp16QState;
    simd<float, 16 * 16> finalOutput = 0;
    simd<float, 16> softMaxDividor = 0;
    simd<float, 2> prevMax = FP32_MIN;
    simd<fp16, 16 * 16> tempOutput;
    simd<fp16, 16 * 16> tempInputA;
    simd<fp16, 16 * 16> tempInputB;

    simd<uint32_t, 16> simdBase16(baseOffsetInc16);

    unsigned int outputOffset = v * 128 + h * 64 * q_head * 128 + hh * 16 * q_head * 128 + vv * 16;
    unsigned int offsetQ = v * 128 + h * 64 * q_head * 128 + vv * 8 * q_head * 128 + hh * 32;
    unsigned int kvheadidx = v * kv_head / q_head;
    unsigned int offsetKBase = kvheadidx * 128 + localLinearId * 4;
    int vReadBlk = localLinearId >> 1;
    int vReadInn = localLinearId & 0x01;
    unsigned int offsetV = kvheadidx * 128 + vReadBlk * kv_head * 128 + vReadInn * 64;

    simd<uint32_t, 16> offsetKScattered = offsetKBase + simdBase16 * kv_head * 128;

#pragma unroll
    for (int k = 0; k < 8; k++) {
        finalOutput.select<32, 1>(k * 32) = __ESIMD_ENS::lsc_block_load<
            float,
            32,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((float*)qState + offsetQ);
        offsetQ += q_head * 128;
    }
    fp16QState.select<256, 1>(0) = finalOutput.select<256, 1>(0);
    finalOutput = 0.0;

    for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
        auto fp16KState = tempInputA.select<64, 1>(0);
        fp16KState.template bit_cast_view<uint32_t>().template select<32, 1>(0) = __ESIMD_ENS::lsc_gather<
            uint32_t,
            2,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached,
            16,
            uint32_t>((uint32_t*)kState, offsetKScattered * sizeof(fp16));

        auto fp16VState = tempInputB.select<64, 1>(0);
        fp16VState = __ESIMD_ENS::lsc_block_load<
            fp16,
            64,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((fp16*)vState + offsetV);

        auto shuffledFp16KState = tempInputA.select<64, 1>(64);
        shuffledFp16KState.select<16, 1>(0) = fp16KState.select<16, 2>(0);
        shuffledFp16KState.select<16, 1>(16) = fp16KState.select<16, 2>(1);
        shuffledFp16KState.select<16, 1>(32) = fp16KState.select<16, 2>(32);
        shuffledFp16KState.select<16, 1>(48) = fp16KState.select<16, 2>(33);

        slm_block_store<fp16, 64>(slmOffsetBaseK + localLinearId * 64 * sizeof(fp16), shuffledFp16KState.select<64, 1>(0));

        barrier();

        auto fp16KData = tempInputA.select<256, 1>(0);
        fp16KData.select<256, 1>(0) = slm_block_load<fp16, 256>(slmOffsetBaseK + hh * 16 * 32 * sizeof(fp16));

        auto fp32tempOutput = tempOutput.template bit_cast_view<float>().select<128, 1>(0);
        fp32tempOutput = 0.0;
#pragma unroll
        for (int t = 0; t < 8; t++) {
#pragma unroll
            for (int ll = 0; ll < 16; ll++) {
                fp32tempOutput.select<16, 1>(16 * t) += fp16KData.select<16, 1>(ll * 16) * fp16QState[t * 32 + ll];
            }
        }

        fp16KData.select<256, 1>(0) = slm_block_load<fp16, 256>(slmOffsetBaseK + (hh * 16 * 32 + 256) * sizeof(fp16));

#pragma unroll
        for (int t = 0; t < 8; t++) {
#pragma unroll
            for (int ll = 0; ll < 16; ll++) {
                fp32tempOutput.select<16, 1>(16 * t) += fp16KData.select<16, 1>(ll * 16) * fp16QState[t * 32 + ll + 16];
            }
        }

#pragma unroll
        for (int t = 0; t < 8; t++) {
            slm_block_store<float, 16>(slmOffsetBaseKq + vv * 8 * 64 * sizeof(float) + t * 16 * 4 * sizeof(float) + hh * 16 * sizeof(float), fp32tempOutput.select<16, 1>(16 * t));
        }
        barrier();

        auto fp32KQResult = tempInputA.template bit_cast_view<float>().select<128, 1>(0);
        fp32KQResult = slm_block_load<float, 128>(slmOffsetBaseKq + localLinearId * 128 * sizeof(float));

#pragma unroll
        for (int t = 0; t < 2; t++) {
            fp32KQResult.select<32, 1>(64 * t) += fp32KQResult.select<32, 1>(64 * t + 32);
            fp32KQResult.select<16, 1>(64 * t) += fp32KQResult.select<16, 1>(64 * t + 16);
        }

#pragma unroll
        for (int t = 0; t < 2; t++) {
            fp32KQResult.select<16, 1>(16 * t) = fp32KQResult.select<16, 1>(64 * t) * matMulQuantCoeff;
        }

#pragma unroll
        for (int t = 0; t < 2; t++) {
            simd<uint32_t, 16> k_idx = simdBase16 + loopIdx * 16;
            uint32_t q_mask_idx = startActivationIdx + 64 * h + 2 * localLinearId + t;
            fp32KQResult.select<16, 1>(16 * t).merge(FP32_MIN, k_idx > q_mask_idx);
        }

        auto maxTemp = tempInputB.template bit_cast_view<float>().select<32, 1>(64);
        maxTemp.select<16, 2>(0) = fp32KQResult.select<16, 1>(0);
        maxTemp.select<16, 2>(1) = fp32KQResult.select<16, 1>(16);

        maxTemp.select<16, 1>(0) = max<float, 16, float>(maxTemp.select<16, 1>(0), maxTemp.select<16, 1>(16));
        maxTemp.select<8, 1>(0) = max<float, 8, float>(maxTemp.select<8, 1>(0), maxTemp.select<8, 1>(8));
        maxTemp.select<4, 1>(0) = max<float, 4, float>(maxTemp.select<4, 1>(0), maxTemp.select<4, 1>(4));
        maxTemp.select<2, 1>(0) = max<float, 2, float>(maxTemp.select<2, 1>(0), maxTemp.select<2, 1>(2));
        maxTemp.select<2, 1>(0) = max<float, 2, float>(maxTemp.select<2, 1>(0), prevMax.select<2, 1>(0));

        if (loopIdx == 0) {
            prevMax = maxTemp.select<2, 1>(0);
        }

        simd<float, 2> compensates = pow<float, 2, float>(2.718f, prevMax - maxTemp.select<2, 1>(0));
        slm_block_store<float, 2>(slmOffsetBaseCompensates + localLinearId * 2 * sizeof(float), compensates);
        prevMax = maxTemp.select<2, 1>(0);

#pragma unroll
        for (int t = 0; t < 2; t++) {
            fp32KQResult.select<16, 1>(t * 16) = fp32KQResult.select<16, 1>(t * 16) - maxTemp[t];
        }
        fp32KQResult.select<32, 1>(0) = pow<float, 32, float>(2.718f, fp32KQResult.select<32, 1>(0));
        simd<fp16, 32> fp16SoftmaxResult = fp32KQResult.select<32, 1>(0);
        slm_block_store<fp16, 32>(slmOffsetBaseSoftMax + sizeof(fp16) * localLinearId * 2 * 16, fp16SoftmaxResult);

        auto sumTemp = tempInputB.template bit_cast_view<float>().select<32, 1>(96);
        sumTemp.select<16, 2>(0) = fp32KQResult.select<16, 1>(0);
        sumTemp.select<16, 2>(1) = fp32KQResult.select<16, 1>(16);
        sumTemp.select<16, 1>(0) = sumTemp.select<16, 1>(0) + sumTemp.select<16, 1>(16);
        sumTemp.select<8, 1>(0) = sumTemp.select<8, 1>(0) + sumTemp.select<8, 1>(8);
        sumTemp.select<4, 1>(0) = sumTemp.select<4, 1>(0) + sumTemp.select<4, 1>(4);
        sumTemp.select<2, 1>(0) = sumTemp.select<2, 1>(0) + sumTemp.select<2, 1>(2);
        // simd<float, 2> sumTemp;
        // sumTemp[0] = sycl::ext::intel::esimd::detail::sum<float, float, 16>(fp32KQResult.select<16, 1>(0));
        // sumTemp[1] = sycl::ext::intel::esimd::detail::sum<float, float, 16>(fp32KQResult.select<16, 1>(16));

        simd<float, 2> accSoftmax = 0.0;
        if (loopIdx > 0) {
            accSoftmax = slm_block_load<float, 2>(slmOffsetBaseAccSoftmax + localLinearId * 2 * sizeof(float));
        }
        accSoftmax = accSoftmax * compensates;
        accSoftmax = accSoftmax + sumTemp.select<2, 1>(0);

        slm_block_store<float, 2>(slmOffsetBaseAccSoftmax + localLinearId * 2 * sizeof(float), accSoftmax);

#pragma unroll
        for (int k = 0; k < 4; k++) {
            slm_block_store<fp16, 16>(slmOffsetBaseV + (vReadBlk * 16 + (vReadInn * 4 + k) * 16 * 16) * sizeof(fp16), fp16VState.select<16, 1>(16 * k));
        }

        barrier();

        auto fp16SoftmaxData = tempOutput.select<256, 1>(0);
        fp16SoftmaxData.select<256, 1>(0) = slm_block_load<fp16, 256>(slmOffsetBaseSoftMax + hh * 256 * sizeof(fp16));

        simd<float, 16> compensate = slm_block_load<float, 16>(slmOffsetBaseCompensates + hh * 16 * sizeof(float));

#pragma unroll
        for (int t = 0; t < 16; t++) {
            finalOutput.select<16, 1>(t * 16) = finalOutput.select<16, 1>(t * 16) * compensate[t];
        }

        auto fp16VData = tempInputA.select<256, 1>(0);
        fp16VData.select<256, 1>(0) = slm_block_load<fp16, 256>(slmOffsetBaseV + vv * 256 * sizeof(fp16));

        // tempOutput = 0.0;
#pragma unroll
        for (int t = 0; t < 16; t++) {
#pragma unroll
            for (int ll = 0; ll < 16; ll++) {
                finalOutput.select<16, 1>(16 * t) += fp16VData.select<16, 1>(ll * 16) * fp16SoftmaxData[t * 16 + ll];
            }
        }

        // finalOutput = finalOutput + tempOutput;

        offsetKScattered += kv_head * 128 * 16;
        offsetV += kv_head * 128 * 16;
        barrier();
    }

    softMaxDividor = slm_block_load<float, 16>(slmOffsetBaseAccSoftmax + hh * 16 * sizeof(float));
    softMaxDividor = 1.0f / softMaxDividor;

#pragma unroll
    for (int t = 0; t < 16; t++) {
        if (64 * h + 16 * hh + t < token_len) {
            simd<float, 16> temp = finalOutput.select<16, 1>(16 * t);
            temp.select<16, 1>(0) = temp.select<16, 1>(0) * softMaxDividor[t];
            __ESIMD_ENS::lsc_block_store<
                float,
                16,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back>((float*)out + outputOffset + t * q_head * 128, temp.select<16, 1>(0));
        }
    }
}

bool runGQA_mat_fusion(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale, uint8_t* shuffleTt)
{
    int mhaGroupH = (token_len + 63) / 64;
    // int groupV = numbOfHead;
    int groupV = q_head;
    int localH = 32;
    int localV = 1;
    sycl::range<2> GlobalRangeMha(localH * mhaGroupH, localV * groupV); // num_head x kv_len, batch size
    sycl::range<2> LocalRangeMha(localH, localV); // kv_len, x
    sycl::nd_range<2> RangeMha(GlobalRangeMha, LocalRangeMha);
    sycl::event e;

    char* envstr = getenv("LLAMA_ESIMD_SDP_FULL_PRECISION");
    bool is_full_precision = false;
    if (envstr && atoi(envstr) == 1) {
        is_full_precision = true;
    }

    try {
        // GQA
        if (is_full_precision) {
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(RangeMha, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    gqa_mat_kernel_hidden128_fullprecision(query, kCache, vCache, outputs, token_len, kv_len, kv_head, q_head, attn_scale, ndi);
                });
            });
        } else {
            e = q->submit([&](handler& cgh) {
                cgh.parallel_for(RangeMha, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    gqa_mat_kernel_hidden128(query, kCache, vCache, outputs, token_len, kv_len, kv_head, q_head, attn_scale, ndi);
                });
            });
        }
    } catch (sycl::exception const& e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    return true;
}
