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

// MT: mask element type (float for fp32 mask, fp16 for fp16 mask)
template <typename OT>
void RunMatMhaQ80Arb_xve_hd128_impl(void* stream, const float* query, uint8_t* kcache_data, uint8_t* vcache_data, uint8_t* mask_data, OT* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, float attn_scale)
{
    const uint32_t localThread = 32;
    uint32_t vThreadnum = (token_len + 31) / 32;
    sycl::range<2> GlobalRange(localThread * q_head, vThreadnum);
    sycl::range<2> LocalRange(localThread, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);
    sycl::queue* q = (sycl::queue*)stream;

    uint32_t cacheline_size = kv_head * 128 / 32 * 34;

    q->submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            slm_init(32 * 128 * sizeof(fp16) + 32 * 128 * sizeof(fp16) + 32 * 32 * sizeof(float));
            constexpr uint32_t slmOffset_QData = 0;
            constexpr uint32_t slmOffset_VData = 32 * 128 * sizeof(fp16);
            constexpr uint32_t slmOffset_SData = 32 * 128 * sizeof(fp16) + 32 * 128 * sizeof(fp16);

            uint32_t v = ndi.get_group(1);
            uint32_t h = ndi.get_group(0);
            uint32_t hk = h * kv_head / q_head;
            uint32_t localId = ndi.get_local_linear_id();

            const uint32_t loopStep = 32;
            int loopNum = (kv_len + loopStep - 1) / loopStep;

            simd<uint32_t, 32> tokenIdxes({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 });
            tokenIdxes += v * 32;
            simd<uint32_t, 32> QOffsets = tokenIdxes * q_head * 128 + h * 128 + localId * 4;
            QOffsets *= sizeof(float);

            uint32_t KVOffset = localId * cacheline_size + hk * 128;

            simd<uint32_t, 32> SlmGatherOffsets({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 });
            SlmGatherOffsets *= 32 * sizeof(float);
            SlmGatherOffsets += localId * sizeof(float) + slmOffset_SData;

            simd<uint32_t, 32> k_idx({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31 });
            simd<uint32_t, 32> mask_mask({ 0x00000001, 0x00000002, 0x00000004, 0x00000008, 0x00000010, 0x00000020, 0x00000040, 0x00000080,
                0x00000100, 0x00000200, 0x00000400, 0x00000800, 0x00001000, 0x00002000, 0x00004000, 0x00008000,
                0x00010000, 0x00020000, 0x00040000, 0x00080000, 0x00100000, 0x00200000, 0x00400000, 0x00800000,
                0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000 });

            float prevmax = FP32_MIN;
            float accSoftmax = 0.0;
            simd<float, 128> accResult = 0.0;

            simd_mask<32> m = tokenIdxes < token_len;
            simd<fp16, 128> qData = gather<float, 128, 4>(query, QOffsets, m);
            qData.select<32, 1>(0).merge(0.0, m == 0);
            qData.select<32, 1>(32).merge(0.0, m == 0);
            qData.select<32, 1>(64).merge(0.0, m == 0);
            qData.select<32, 1>(96).merge(0.0, m == 0);
            slm_block_store<fp16, 128>(slmOffset_QData + localId * 128 * sizeof(fp16), qData);
            barrier();

            for (int l = 0; l < loopNum; l++) {
                simd<float, 32> qkResult = 0.0;
                simd<fp16, 128> vData = 0.0;
                if (l * loopStep + localId < kv_len) {
                    // Load raw int8 K and V data
                    simd<int8_t, 128> kRaw = block_load<int8_t, 128>((int8_t*)kcache_data + KVOffset);
                    simd<int8_t, 128> vRaw = block_load<int8_t, 128>((int8_t*)vcache_data + KVOffset);

                    // Dequantize: 4 blocks of 32 int8 values, each with an fp16 scale
                    uint32_t tokenStart = KVOffset - hk * 128;
                    simd<fp16, 128> kData;
#pragma unroll
                    for (int b = 0; b < 4; b++) {
                        uint32_t scaleOff = tokenStart + kv_head * 128 + (hk * 4 + b) * sizeof(fp16);
                        float kscale = (float)(*(fp16*)(kcache_data + scaleOff));
                        float vscale = (float)(*(fp16*)(vcache_data + scaleOff));
                        kData.select<32, 1>(b * 32) = kscale * kRaw.select<32, 1>(b * 32);
                        vData.select<32, 1>(b * 32) = vscale * vRaw.select<32, 1>(b * 32);
                    }
                    KVOffset += loopStep * cacheline_size;

#pragma unroll
                    for (int k = 0; k < 32; k++) {
                        qData = slm_block_load<fp16, 128>(slmOffset_QData + k * 128 * sizeof(fp16));
                        fp16 kvalue = kData[k * 4];
                        qkResult += qData.select<32, 1>(0) * kvalue;
                        kvalue = kData[k * 4 + 1];
                        qkResult += qData.select<32, 1>(32) * kvalue;
                        kvalue = kData[k * 4 + 2];
                        qkResult += qData.select<32, 1>(64) * kvalue;
                        kvalue = kData[k * 4 + 3];
                        qkResult += qData.select<32, 1>(96) * kvalue;
                    }
                }
                slm_block_store<float, 32>(slmOffset_SData + localId * 32 * sizeof(float), qkResult);
                barrier();
                slm_block_store<fp16, 128>(slmOffset_VData + localId * 128 * sizeof(fp16), vData);
                simd<float, 32> qkrow = slm_gather<float, 32>(SlmGatherOffsets);
                qkrow = qkrow * attn_scale;

                // Load arbitrary mask values and add to QK
                uint32_t q_row = v * 32 + localId;
                uint32_t k_start = l * loopStep;

                simd<float, 32> mask_val = 0.0;
                if (q_row < token_len) {
                    uint32_t mask_int = *(uint32_t*)(mask_data + q_row * (kv_len + 7) / 8 + k_start / 8);
                    mask_val.merge(FP32_MIN, (mask_int & mask_mask) > 0);
                    // mask_val = block_load<MT, 32>(mask_data + q_row * kv_len + k_start);
                }

                // Mask out-of-range KV positions
                mask_val.merge(FP32_MIN, k_idx >= kv_len);

                qkrow = qkrow + mask_val;

                float curmax = hmax<float, float, 32>(qkrow);
                curmax = curmax > prevmax ? curmax : prevmax;
                qkrow = qkrow - curmax;
                qkrow = sycl::ext::intel::esimd::pow<float, 32, float>(2.718281828f, qkrow);
                k_idx += loopStep;

                float qkrowSum = sycl::ext::intel::esimd::detail::sum<float, float, 32>(qkrow);
                float compensate = (l == 0) ? 1.0 : sycl::ext::intel::esimd::exp(prevmax - curmax);
                accSoftmax = accSoftmax * compensate + qkrowSum;
                prevmax = curmax;

                accResult = accResult * compensate;

                barrier();
#pragma unroll
                for (int j = 0; j < 32; j++) {
                    simd<fp16, 128> vData = slm_block_load<fp16, 128>(slmOffset_VData + j * 128 * sizeof(fp16));
                    float s = qkrow[j];
                    accResult += vData * s;
                }
            }

            simd<float, 128> outData;
            if (accSoftmax != 0) {
                outData = accResult / accSoftmax;
            } else {
                outData = 0.0;
            }
            uint32_t outputOffset = (v * 32 + localId) * q_head * 128 + h * 128;
            if (v * 32 + localId < token_len) {
                block_store<OT, 128>(outputs + outputOffset, outData);
            }
        });
    });
}

extern "C" void __declspec(dllexport) RunMhaQ80Arb_xve_hd128(void* stream, const void* query, uint8_t* kCache, uint8_t* vCache, uint8_t* mask, void* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, uint32_t head_dim, float attn_scale, int input_precision, int output_precision)
{
    assert(head_dim == 128);
    assert(input_precision == 0); // only support float input for now

    if (output_precision == 0) {
        RunMatMhaQ80Arb_xve_hd128_impl<float>(stream, (float*)query, kCache, vCache, mask, (float*)outputs, token_len, kv_len, kv_head, q_head, attn_scale);
    } else if (output_precision == 1) {
        RunMatMhaQ80Arb_xve_hd128_impl<fp16>(stream, (float*)query, kCache, vCache, mask, (fp16*)outputs, token_len, kv_len, kv_head, q_head, attn_scale);
    }
}
