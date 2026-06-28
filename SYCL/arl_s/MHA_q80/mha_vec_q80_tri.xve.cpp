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

typedef sycl::half fp16;
using namespace std;
using namespace sycl;
using namespace sycl::ext::intel::esimd;

#define FP32_MIN (-3.4e+38F)

template <typename IT, typename OT, int HEAD_DIM>
void RunVecMhaQ80Tri_xve_impl(void* stream, const IT* query, uint8_t* kcache_data, uint8_t* vcache_data, OT* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale)
{
    static_assert(HEAD_DIM == 64 || HEAD_DIM == 128 || HEAD_DIM == 256, "HEAD_DIM must be 64/128/256");
    constexpr uint32_t BLOCKS = HEAD_DIM / 32;

    const uint32_t localThread = 32;
    sycl::range<2> GlobalRange(localThread * q_head, token_len);
    sycl::range<2> LocalRange(localThread, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);
    sycl::queue* q = (sycl::queue*)stream;

    uint32_t cacheline_size = kv_head * HEAD_DIM / 32 * 34;
    uint32_t valid_len = kv_len;

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            slm_init(localThread * HEAD_DIM * sizeof(float) + localThread * sizeof(float) + localThread * sizeof(float));
            const uint32_t slmOffset_qkvResult = 0;
            const uint32_t slmOffset_softmaxSum = localThread * HEAD_DIM * sizeof(float);
            const uint32_t slmOffset_maxqk = localThread * HEAD_DIM * sizeof(float) + localThread * sizeof(float);

            int h = ndi.get_group(0);
            int t = ndi.get_group(1);
            int localLinearId = ndi.get_local_linear_id();
            int hk = h * kv_head / q_head;

            int loopStep = 8 * localThread;
            int loopNum = (valid_len + loopStep - 1) / loopStep;

            float maxQK = FP32_MIN;
            float accSoftMax = 0.0;
            simd<float, HEAD_DIM> accResult = 0.0;

            simd<fp16, HEAD_DIM> qData = block_load<IT, HEAD_DIM>(query + h * HEAD_DIM + t * q_head * HEAD_DIM);

            for (int l = 0; l < loopNum; l++) {
#pragma unroll
                for (int j = 0; j < 8; j++) {
                    uint32_t token_idx = l * loopStep + localLinearId * 8 + j;
                    if (token_idx >= valid_len) {
                        break;
                    }
                    if (token_idx >= valid_len - token_len + t + 1) {
                        break;
                    }

                    uint32_t token_offset = token_idx * cacheline_size;
                    simd<int8_t, HEAD_DIM> kRaw = block_load<int8_t, HEAD_DIM>((int8_t*)kcache_data + token_offset + hk * HEAD_DIM);
                    simd<int8_t, HEAD_DIM> vRaw = block_load<int8_t, HEAD_DIM>((int8_t*)vcache_data + token_offset + hk * HEAD_DIM);

                    simd<float, HEAD_DIM> kData;
                    simd<float, HEAD_DIM> vData;
#pragma unroll
                    for (int b = 0; b < BLOCKS; b++) {
                        uint32_t scale_off = token_offset + kv_head * HEAD_DIM + (hk * BLOCKS + b) * sizeof(fp16);
                        float kscale = (float)(*(fp16*)((uint8_t*)kcache_data + scale_off));
                        float vscale = (float)(*(fp16*)((uint8_t*)vcache_data + scale_off));
                        kData.template select<32, 1>(b * 32) = kscale * kRaw.template select<32, 1>(b * 32);
                        vData.template select<32, 1>(b * 32) = vscale * vRaw.template select<32, 1>(b * 32);
                    }

                    simd<float, HEAD_DIM> temp = qData * kData;
                    float qkResult = sycl::ext::intel::esimd::detail::sum<float, float, HEAD_DIM>(temp) * attn_scale;
                    if (qkResult > maxQK) {
                        float compensate = sycl::ext::intel::esimd::exp(maxQK - qkResult);
                        accResult = accResult * compensate + vData;
                        accSoftMax = accSoftMax * compensate + 1.0;
                        maxQK = qkResult;
                    } else {
                        float compensate = sycl::ext::intel::esimd::exp(qkResult - maxQK);
                        accResult = accResult + compensate * vData;
                        accSoftMax = accSoftMax + compensate;
                    }
                }
            }

            slm_block_store<float, 1>(slmOffset_maxqk + localLinearId * sizeof(float), maxQK);

            barrier();

            simd<float, localThread> maxQKs = slm_block_load<float, localThread>(slmOffset_maxqk);
            float globalMaxQK = hmax<float, float, localThread>(maxQKs);
            float compensate = sycl::ext::intel::esimd::exp(maxQK - globalMaxQK);
            accResult = accResult * compensate;
            accSoftMax = accSoftMax * compensate;
            slm_block_store<float, HEAD_DIM>(slmOffset_qkvResult + localLinearId * HEAD_DIM * sizeof(float), accResult);
            slm_block_store<float, 1>(slmOffset_softmaxSum + localLinearId * sizeof(float), accSoftMax);

            barrier();

            constexpr uint32_t accBlock = localThread / 4;
            if (localLinearId < 4) {
                accResult = 0.0;
                accSoftMax = 0.0;
#pragma unroll
                for (int i = 0; i < accBlock; i++) {
                    accResult = accResult + slm_block_load<float, HEAD_DIM>(slmOffset_qkvResult + (localLinearId * accBlock + i) * HEAD_DIM * sizeof(float));
                    accSoftMax = accSoftMax + slm_block_load<float, 1>(slmOffset_softmaxSum + (localLinearId * accBlock + i) * sizeof(float));
                }

                slm_block_store<float, HEAD_DIM>(slmOffset_qkvResult + localLinearId * accBlock * HEAD_DIM * sizeof(float), accResult);
                slm_block_store<float, 1>(slmOffset_softmaxSum + localLinearId * accBlock * sizeof(float), accSoftMax);
            }

            barrier();

            if (localLinearId == 0) {
                accResult = 0.0;
                accSoftMax = 0.0;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    accResult = accResult + slm_block_load<float, HEAD_DIM>(slmOffset_qkvResult + i * accBlock * HEAD_DIM * sizeof(float));
                    accSoftMax = accSoftMax + slm_block_load<float, 1>(slmOffset_softmaxSum + i * accBlock * sizeof(float));
                }

                if (accSoftMax > 0) {
                    accResult = accResult / accSoftMax;
                } else {
                    accResult = 0.0;
                }
                block_store<OT, HEAD_DIM>((OT*)outputs + h * HEAD_DIM + t * q_head * HEAD_DIM, accResult);
            }
        });
    });
}

extern "C" void __declspec(dllexport) RunMhaQ80Tri_xve(void* stream, const void* query, uint8_t* kCache, uint8_t* vCache, void* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, uint32_t head_dim, float attn_scale, int input_precision, int output_precision)
{
    if (head_dim != 128) {
        return;
    }

    if (input_precision == 0 && output_precision == 0) {
        RunVecMhaQ80Tri_xve_impl<float, float, 128>(stream, (float*)query, kCache, vCache, (float*)outputs, token_len, kv_len, kv_head, q_head, attn_scale);
    } else if (input_precision == 0 && output_precision == 1) {
        RunVecMhaQ80Tri_xve_impl<float, fp16, 128>(stream, (float*)query, kCache, vCache, (fp16*)outputs, token_len, kv_len, kv_head, q_head, attn_scale);
    } else if (input_precision == 1 && output_precision == 0) {
        RunVecMhaQ80Tri_xve_impl<fp16, float, 128>(stream, (fp16*)query, kCache, vCache, (float*)outputs, token_len, kv_len, kv_head, q_head, attn_scale);
    } else if (input_precision == 1 && output_precision == 1) {
        RunVecMhaQ80Tri_xve_impl<fp16, fp16, 128>(stream, (fp16*)query, kCache, vCache, (fp16*)outputs, token_len, kv_len, kv_head, q_head, attn_scale);
    }
}
