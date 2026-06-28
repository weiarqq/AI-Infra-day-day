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

extern "C" bool runGQA_vec_fusion_transv(queue& q, uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* qkvOut, unsigned batch_size, unsigned kv_len, unsigned vCacheStride, unsigned numbOfHead  /*32: MHA, 8: GQA*/);
extern "C" bool runGQA_vec_fusion(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, uint32_t q_precision, uint32_t o_precision, float attn_scale);


ESIMD_INLINE void qkvFusionGqa_transv(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* out,
  int kvSeqLen,
  int vCacheStride,
  nd_item<2>& ndi) {
  constexpr float matMulQuantCoeff = 0.08838834764831844f; // 1.0f / sqrt(128.0f);
  __ESIMD_NS::slm_init(32 * 128 * sizeof(fp16) + 32 * 4 * sizeof(float));
  int localLinearId = ndi.get_local_linear_id();
  int hh = localLinearId & 0x3;
  int vv = localLinearId >> 2;
  int h = ndi.get_group(0);
  int v = ndi.get_group(1);
  int kvSeqOutLoopCount = (kvSeqLen + 0x1f) >> 5;
  simd<float, 512> qqFp32;
  simd<float, 16> kvCacheOut = 0;
  simd<float, 16> softMaxSumTemp = 0;
  simd<fp16, 16 * 16> vvCache;
  auto kk = vvCache.select<128, 1>(0);
  auto vvState = vvCache.select<128, 1>(128);
  simd<float, 128> fp32Kv;
  simd<float, 32> softMaxCache;
  simd<float, 4> softMax;
  simd<uint32_t, 8> softMaxSlmOffsets;
  float softMaxPadding = 0;
  
  unsigned int outputOffset = (v * 32 + h * 4) * 128 + vv * 16 + hh * 128;
  unsigned int outputVOffsetSlm = localLinearId * 128 * sizeof(fp16);
  unsigned int outputSoftmaxOffsetSlm = 32 * 128 * sizeof(fp16) + localLinearId * sizeof(float);
  unsigned int offsetQ = 4 * h * 128 * sizeof(float);
  unsigned int offsetK = (h * 128 + localLinearId * 128 * 8) * sizeof(fp16);
  unsigned int offsetVBase = (h * vCacheStride * 128 + vv * 16 * vCacheStride + hh * 8) * sizeof(fp16);
  simd<uint32_t, 16> offsetV;
  for (int k = 0; k < 16; k++) {
    offsetV[k] = k;
  }
  softMaxSlmOffsets.select<4, 2>(0) = offsetV.select<4, 1>(0);
  softMaxSlmOffsets.select<4, 2>(1) = softMaxSlmOffsets.select<4, 2>(0);
  softMaxSlmOffsets = softMaxSlmOffsets * 32 * sizeof(float) + outputSoftmaxOffsetSlm;
  softMaxSlmOffsets.select<4, 2>(1) = softMaxSlmOffsets.select<4, 2>(1) + sizeof(fp16);
  offsetV = offsetV * vCacheStride * sizeof(fp16) + offsetVBase;
  int kvSeqOffset = localLinearId;

#pragma unroll
  for (int kk = 0; kk < 8; kk++) {
    qqFp32.template bit_cast_view<unsigned char>().template select<256, 1>(256 * kk) =
      __ESIMD_ENS::lsc_block_load<
      unsigned char,
      256,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((unsigned char*)qState + offsetQ + 256 * kk);
  }

  for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
    vvState.template bit_cast_view<uint32_t>().template select<64, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      uint32_t,
      4,
      __ESIMD_ENS::lsc_data_size::u32,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((uint32_t*)vState, offsetV);

    if (kvSeqOffset < kvSeqLen) {
      kk.template bit_cast_view<unsigned char>().template select<256, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        unsigned char,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((unsigned char*)kState + offsetK);

      fp32Kv = kk;
#pragma unroll
      for (int l = 0; l < 4; l++) {
        simd<float, 16> output = 0;
#pragma unroll
        for (int ll = 0; ll < 8; ll++) {
          output.select<16, 1>(0) += qqFp32.select<16, 1>(128 * l + 16 * ll) * fp32Kv.select<16, 1>(16 * ll);
        }

        output.select<8, 1>(0) = output.select<8, 1>(0) + output.select<8, 1>(0 + 8);
        output.select<4, 1>(0) = output.select<4, 1>(0) + output.select<4, 1>(0 + 4);
        output.select<2, 1>(0) = output.select<2, 1>(0) + output.select<2, 1>(0 + 2);
        softMax[l] = output[0] + output[1];
      }

      softMax = softMax * matMulQuantCoeff;
      softMax = pow<float, 4, float>(2.718f, softMax);
    } else {
      softMax = 0;
    }

    slm_scatter<uint16_t, 8>(softMaxSlmOffsets, softMax.template bit_cast_view<uint16_t>());
#pragma unroll
    for (int ll = 0; ll < 4; ll++) {
      simd<fp16, 32> shuffleTemp;
      shuffleTemp = vvState.select<32, 1>(32 * ll);
      vvState.select<16, 1>(32 * ll) = shuffleTemp.select<16, 2>(0);
      vvState.select<16, 1>(32 * ll + 16) = shuffleTemp.select<16, 2>(1);
    }
#pragma unroll
    for (int ll = 0; ll < 2; ll++) {
      slm_block_store<fp16, 64>(outputVOffsetSlm + ll * 64 * sizeof(fp16), vvState.select<64, 1>(64 * ll));
    }

    barrier();
    {
      int slmVLoadOffset = vv * 16 * 32 * sizeof(fp16);
      int slmSoftMaxLoadOffset = 32 * 128 * sizeof(fp16) + hh * 32 * sizeof(float);
      softMaxCache.select<32, 1>(0) = slm_block_load<float, 32>(slmSoftMaxLoadOffset);
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        vvCache.select<128, 1>(128 * cc) = slm_block_load<fp16, 128>(slmVLoadOffset + 128 * cc * sizeof(fp16));
      }

#pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        fp32Kv = vvCache.select<128, 1>(128 * nn);
#pragma unroll
        for (int nnn = 0; nnn < 8; nnn++) {
          if (32 * loopIdx + nn * 8 + nnn < kvSeqLen) {
            kvCacheOut.select<16, 1>(0) += fp32Kv.select<16, 1>(16 * nnn) * softMaxCache[nn * 8 + nnn];
          }
        }
      }

#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        vvCache.select<128, 1>(128 * cc) = slm_block_load<fp16, 128>(slmVLoadOffset + 256 * sizeof(fp16) + 128 * cc * sizeof(fp16));
      }

#pragma unroll
      for (int nn = 0; nn < 2; nn++) {
        fp32Kv = vvCache.select<128, 1>(128 * nn);
#pragma unroll
        for (int nnn = 0; nnn < 8; nnn++) {
          if (32 * loopIdx + nn * 8 + nnn + 16 < kvSeqLen) {
            kvCacheOut.select<16, 1>(0) += fp32Kv.select<16, 1>(16 * nnn) * softMaxCache[nn * 8 + nnn + 16];
          }
        }
      }

#pragma unroll
      for (int mm = 0; mm < 2; mm++) {
        softMaxSumTemp += softMaxCache.select<16, 1>(16 * mm);
      }
    }

    offsetK += 32 * 128 * 8 * sizeof(fp16);
    offsetV += 32 * sizeof(fp16);
    kvSeqOffset += 32;
    barrier();
  }

  {
    softMaxSumTemp.select<8, 1>(0) = softMaxSumTemp.select<8, 1>(0) + softMaxSumTemp.select<8, 1>(8);
    softMaxSumTemp.select<4, 1>(0) = softMaxSumTemp.select<4, 1>(0) + softMaxSumTemp.select<4, 1>(4);
    softMaxSumTemp.select<2, 1>(0) = softMaxSumTemp.select<2, 1>(0) + softMaxSumTemp.select<2, 1>(2);
    softMaxSumTemp[0] = softMaxSumTemp[0] + softMaxSumTemp[1];
    softMaxSumTemp[0] = 1.0f / softMaxSumTemp[0];
    kvCacheOut.select<16, 1>(0) = kvCacheOut.select<16, 1>(0) * softMaxSumTemp[0];
    __ESIMD_ENS::lsc_block_store<
      float,
      16,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((float*)out + outputOffset, kvCacheOut);
  }
}

bool runGQA_vec_fusion_transv(queue& q, uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* qkvOut, unsigned batch_size, unsigned kv_len, unsigned vCacheStride, unsigned numbOfHead  /*32: MHA, 8: GQA*/) {
  int groupH = numbOfHead;
  int localH = 32;

  sycl::range<2> GlobalRange(32 * groupH, batch_size);
  sycl::range<2> LocalRange(32, 1);
  sycl::nd_range<2> Range(GlobalRange, LocalRange);
  sycl::event e;
  try {
    switch (numbOfHead) {
    case 8:
      e = q.submit([&](handler& cgh) {
        cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            qkvFusionGqa_transv(qState, kState, vState, qkvOut, kv_len, vCacheStride, ndi);
          });
        });
      break;
    default:
      break;
    }
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool success = true;
  return success;
}

template<uint32_t LocalThread, uint32_t Step, typename IT, typename OT>
ESIMD_INLINE void gqa_kernel_hidden128(uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* out, int kv_len, uint32_t KVHead, uint32_t QHead, float attn_scale, nd_item<2>& ndi)
{
    __ESIMD_NS::slm_init(LocalThread * 128 * sizeof(float) + LocalThread * sizeof(float) + LocalThread * sizeof(float));
    constexpr uint32_t slmOffset_qkvResult = 0;
    constexpr uint32_t slmOffset_softmaxSum = LocalThread * 128 * sizeof(float);
    constexpr uint32_t slmOffset_maxqk = LocalThread * 128 * sizeof(float) + LocalThread * sizeof(float);

    // attn_scale passed as parameter

    simd<fp16, 128> qData;
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
        IT,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((IT*)qState + offsetQ);
    qData.select<64, 1>(64) =
        __ESIMD_ENS::lsc_block_load<
        IT,
        64,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((IT*)qState + offsetQ + 64);

    for (int l = 0; l < loopNum; l++)
    {
#pragma unroll
        for (int s = 0; s < Step; s++)
        {
            if (l * loopStep + localLinearId * Step + s < kv_len - token_len + t + 1)
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
            if (l * loopStep + localLinearId * Step + s < kv_len - token_len + t + 1)
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
        for (int s = 0; s < Step; s++)
        {
            if (l * loopStep + localLinearId * Step + s >= kv_len - token_len + t + 1)
            {
                break;
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

        block_store<OT, 128>((OT*)out + offsetOutput, accResult.select<128, 1>(0));
    }

}


template<uint32_t batchnum>
ESIMD_INLINE void qkvFusionGqa_KV8_Q32_batchN(
  uint8_t* qState,
  uint8_t* kState,
  uint8_t* vState,
  uint8_t* out,
  int kvSeqLen,
  float attn_scale,
  nd_item<2>& ndi) {
  float matMulQuantCoeff = attn_scale; // 1.0f / sqrt(128.0f);
  __ESIMD_NS::slm_init(64 * 128 * sizeof(fp16) + 64 * 4 * 4 * sizeof(float) + 4 * 128 * 4 * sizeof(float));
  int localLinearId = ndi.get_local_linear_id();
  int hh = localLinearId & 0x3;
  int vv = localLinearId >> 2;
  int h = ndi.get_group(0);
  int kvSeqOutLoopCount = (kvSeqLen + 63) >> 6;

  simd<fp16, 512> tempInputA;
  simd<fp16, 512> tempInputB;
  simd<fp16, 256> tempOutput;
  simd<float, 64> finalOutput = 0;
  simd<float, 64> softMaxSumTemp = 0;
  simd<float, 4> prevMax;

  //auto softMaxCache = qqFp16.template bit_cast_view<float>().select<256, 1>(0);
  //auto kk = softMaxCache.template bit_cast_view<fp16>().select<256, 1>(0);
  //auto vvState = softMaxCache.template bit_cast_view<fp16>().select<256, 1>(256);

  unsigned int outputOffset = h * 4 * 128 + vv * 16 + hh * 128;
  constexpr unsigned int inputQBaseOffsetSlm = 64 * 128 * sizeof(fp16) + 64 * 4 * 4 * sizeof(float);

  unsigned int offsetQ = 4 * h * 128;
  unsigned int offsetK = h * 128 + localLinearId * 128 * 8;
  unsigned int offsetV = h * 128 + localLinearId * 128 * 8;
  //unsigned int prevMaxBaseOffsetSlm = 32 * 128 * sizeof(fp16) + 32 * 4 * 4 * sizeof(float) + 4 * 128 * 4 * sizeof(float);

  
  int kvSeqOffset = localLinearId;

  simd<uint32_t, 16> gatherOffsetQ = 0;

#pragma unroll
  for (int j = 0; j < batchnum; j++)
  {
    gatherOffsetQ[j] = j;
  }

  gatherOffsetQ.select<4, 1>(0) = gatherOffsetQ.select<4, 1>(0) * 32 * 128;
  gatherOffsetQ.select<4, 1>(4) = gatherOffsetQ.select<4, 1>(0) + 128;
  gatherOffsetQ.select<4, 1>(8) = gatherOffsetQ.select<4, 1>(4) + 128;
  gatherOffsetQ.select<4, 1>(12) = gatherOffsetQ.select<4, 1>(8) + 128;
  gatherOffsetQ += (offsetQ + 4 * localLinearId);
  gatherOffsetQ *= sizeof(float);

  auto qqFp32 = tempInputA.template bit_cast_view<float>().select<64, 1>(0);
  auto qqFp16 = tempInputA.select<64, 1>(128);

  qqFp32.select<64, 1>(0) =
    __ESIMD_ENS::lsc_gather<
    float,
    4,
    __ESIMD_ENS::lsc_data_size::u32,
    __ESIMD_ENS::cache_hint::cached,
    __ESIMD_ENS::cache_hint::cached,
    16,
    uint32_t
    >((float*)qState, gatherOffsetQ);

  qqFp16.select<64, 1>(0) = qqFp32.select<64, 1>(0) * 0.5; // multiply 0.5 for 2-batch accumulater

  slm_block_store<fp16, 64>(inputQBaseOffsetSlm + localLinearId * 128, qqFp16.select<64, 1>(0));

  barrier();

  prevMax = FP32_MIN;


  for (int loopIdx = 0; loopIdx < kvSeqOutLoopCount; loopIdx++) {
    auto vvState = tempInputB.select<256, 1>(0);
    vvState.select<128, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        fp16,
        128,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((fp16*)vState + offsetV);

    vvState.select<128, 1>(128) =
        __ESIMD_ENS::lsc_block_load<
        fp16,
        128,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((fp16*)vState + offsetV + 32 * 8 * 128);

    auto softMax = tempOutput.template bit_cast_view<float>().select<32, 1>(0);
    auto output = tempOutput.select<16, 1>(64);
    softMax = 0.0;
    output = 0.0;
    auto kk = tempInputB.select<256, 1>(256);
    if (kvSeqOffset < kvSeqLen) {
      kk.select<128, 1>(0) =
        __ESIMD_ENS::lsc_block_load<
        fp16,
        128,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((fp16*)kState + offsetK);
    }
    if (kvSeqOffset + 32 < kvSeqLen) {
      kk.select<128, 1>(128) =
        __ESIMD_ENS::lsc_block_load<
        fp16,
        128,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((fp16*)kState + offsetK + 32 * 128 * 8);
    }

      auto fp16QData = tempInputA.select<512, 1>(0);
      //if (fp32Kv[0] > 70000.0)
      {
#pragma unroll
        for (int bb = 0; bb < 4; bb++)
        {
          fp16QData.select<512, 1>(0) = slm_block_load<fp16, 512>(inputQBaseOffsetSlm + bb * 512 * sizeof(fp16));
          
          if (kvSeqOffset < kvSeqLen)
          {
#pragma unroll
            for (int mm = 0; mm < 16; mm ++)
            {
              output = 0.0;
#pragma unroll
              for (int ll = 0; ll < 2; ll++)
              {
                output.select<16, 1>(0) += fp16QData.select<16, 1>((mm * 2 + ll) * 16) * kk[bb * 32 + mm * 2 + ll];
              }
              softMax.select<16, 1>(0) = softMax.select<16, 1>(0) + output; 
            }
          }

          if (kvSeqOffset + 32 < kvSeqLen)
          {
#pragma unroll
            for (int mm = 0; mm < 16; mm ++)
            {
              output = 0.0;
#pragma unroll
              for (int ll = 0; ll < 2; ll++)
              {
                output.select<16, 1>(0) += fp16QData.select<16, 1>((mm * 2 + ll) * 16) * kk[128 + bb * 32 + mm * 2 + ll];
              }
              softMax.select<16, 1>(16) = softMax.select<16, 1>(16) + output; 
            }
          }
        }
      }


    softMax = softMax * matMulQuantCoeff * 2.0; // compensate the x 0.5 when reading the q data
    //softMax = pow<float, 16, float>(2.718f, softMax);

    if (kvSeqOffset >= kvSeqLen)
    {
      softMax.select<16, 1>(0) = FP32_MIN;
    }
    else if (kvSeqOffset >= kvSeqLen - batchnum + 1)
    {
      softMax.select<4, 4>(0) = FP32_MIN;
      if (kvSeqOffset >= kvSeqLen - batchnum + 2)
      {
        softMax.select<4, 4>(1) = FP32_MIN;
        if (kvSeqOffset >= kvSeqLen - batchnum + 3)
        {
          softMax.select<4, 4>(2) = FP32_MIN;
        }
      }
    }

    if (kvSeqOffset + 32 >= kvSeqLen)
    {
      softMax.select<16, 1>(16) = FP32_MIN;
    }
    else if (kvSeqOffset + 32 >= kvSeqLen - batchnum + 1)
    {
      softMax.select<4, 4>(16) = FP32_MIN;
      if (kvSeqOffset + 32 >= kvSeqLen - batchnum + 2)
      {
        softMax.select<4, 4>(17) = FP32_MIN;
        if (kvSeqOffset + 32>= kvSeqLen - batchnum + 3)
        {
          softMax.select<4, 4>(18) = FP32_MIN;
        }
      }
    }

    barrier();

#pragma unroll
    for (int l = 0; l < 4; l ++)
    {
      slm_block_store<float, 4>(128 * 64 * sizeof(fp16) + l * 64 * 4 * sizeof(float) + localLinearId * 4 * sizeof(float), softMax.select<4, 1>(l * 4));
      slm_block_store<float, 4>(128 * 64 * sizeof(fp16) + l * 64 * 4 * sizeof(float) + 32*4*sizeof(float) + localLinearId * 4 * sizeof(float), softMax.select<4, 1>(l * 4 + 16));
    }

//     auto shuffleTemp = tempOutput.select<256, 1>(0);
// #pragma unroll
//     for (int ll = 0; ll < 8; ll++) {
//       shuffleTemp.select<16, 1>(32 * ll) = vvState.select<16, 2>(ll * 32);
//       shuffleTemp.select<16, 1>(32 * ll + 16) = vvState.select<16, 2>(ll * 32 + 1);
//     }

//     slm_block_store<fp16, 256>(outputVOffsetSlm, shuffleTemp.select<256, 1>(0));

#pragma unroll
    for (int r = 0; r < 2; r ++)
    {
#pragma unroll
        for (int k = 0; k < 8; k ++)
        {
            slm_block_store<fp16, 16>(sizeof(fp16) * (r * 32 * 16 + localLinearId * 16 + k * 64 * 16), vvState.select<16, 1>(r * 128 + k * 16));
        }
    }

    barrier();
    {
      int slmVLoadOffset = vv * 16 * 64 * sizeof(fp16);
      
      auto softMaxCache = tempInputA.template bit_cast_view<float>().select<256, 1>(0);
      softMaxCache.select<256, 1>(0) = slm_block_load<float, 256>(64*128*sizeof(fp16) + hh * 4 * 64 * sizeof(float));

      auto maxTemp = tempInputB.template bit_cast_view<float>().select<128, 1>(0);
      maxTemp = max<float, 128, float>(softMaxCache.select<128, 1>(0), softMaxCache.select<128, 1>(128));
      maxTemp.select<64, 1>(0) = max<float, 64, float>(maxTemp.select<64, 1>(0), maxTemp.select<64, 1>(64));
      maxTemp.select<32, 1>(0) = max<float, 32, float>(maxTemp.select<32, 1>(0), maxTemp.select<32, 1>(32));
      maxTemp.select<16, 1>(0) = max<float, 16, float>(maxTemp.select<16, 1>(0), maxTemp.select<16, 1>(16));
      maxTemp.select<8, 1>(0) = max<float, 8, float>(maxTemp.select<8, 1>(0), maxTemp.select<8, 1>(8));
      maxTemp.select<4, 1>(0) = max<float, 4, float>(maxTemp.select<4, 1>(0), maxTemp.select<4, 1>(4));

      maxTemp.select<4, 1>(0) = max<float, 4, float>(maxTemp.select<4, 1>(0), prevMax.select<4, 1>(0));

      if (loopIdx == 0)
      {
        prevMax = maxTemp.select<4, 1>(0);
      }
      
#pragma unroll
      for (int j = 0; j < batchnum; j++)
      {
        softMaxCache.select<64, 4>(j) = softMaxCache.select<64, 4>(j) - maxTemp[j];
      }

      auto compensate = tempOutput.template bit_cast_view<float>().select<4, 1>(0);
      compensate = pow<float, 4, float>(2.718f, prevMax - maxTemp.select<4, 1>(0));

#pragma unroll
      for (int j = 0; j < batchnum; j++)
      {
        finalOutput.select<16, 1>(16*j) = finalOutput.select<16, 1>(16*j) * compensate[j];
      }

#pragma unroll
      for (int j = 0; j < batchnum; j++)
      {
        softMaxSumTemp.select<16, 4>(j) = softMaxSumTemp.select<16, 4>(j) * compensate[j];
      }
      prevMax = maxTemp.select<4, 1>(0);

      auto fp16VData = tempInputB.select<512, 1>(0);
      auto softmax_results = tempOutput.template bit_cast_view<float>().select<128, 1>();
#pragma unroll
      for (int bb = 0; bb < 2; bb ++)
      {
        softmax_results = pow<float, 128, float>(2.718f, softMaxCache.select<128, 1>(bb * 128));
        fp16VData.select<512, 1>(0) = slm_block_load<fp16, 512>(slmVLoadOffset + bb * 512 * sizeof(fp16));

        //simd<float, 64> kvCacheOut = 0.0;
#pragma unroll
        for (int j = 0; j < batchnum; j ++)
        {
#pragma unroll
          for (int ll = 0; ll < 32; ll ++)
          {
            finalOutput.select<16, 1>(16 * j) += fp16VData.select<16, 1>(ll * 16) * softmax_results[ll * 4 + j]; 
          }
        }

        //finalOutput = finalOutput + kvCacheOut;
        softMaxSumTemp.select<64, 1>(0) += softmax_results.select<64, 1>(0);
        softMaxSumTemp.select<64, 1>(0) += softmax_results.select<64, 1>(64);
      }

    }

    offsetK += 64 * 128 * 8;
    offsetV += 64 * 128 * 8;
    kvSeqOffset += 64;
    //barrier();
  }

  {
    simd<float, 64> shuffleTemp = softMaxSumTemp;

    softMaxSumTemp.select<32, 1>(0) = softMaxSumTemp.select<32, 1>(0) + softMaxSumTemp.select<32, 1>(32);
    softMaxSumTemp.select<16, 1>(0) = softMaxSumTemp.select<16, 1>(0) + softMaxSumTemp.select<16, 1>(16);
    softMaxSumTemp.select<8, 1>(0) = softMaxSumTemp.select<8, 1>(0) + softMaxSumTemp.select<8, 1>(8);
    softMaxSumTemp.select<4, 1>(0) = softMaxSumTemp.select<4, 1>(0) + softMaxSumTemp.select<4, 1>(4);
    softMaxSumTemp.select<4, 1>(0) = 1.0f / softMaxSumTemp.select<4, 1>(0);

#pragma unroll
    for (int j = 0; j < batchnum; j++)
    {
      finalOutput.select<16, 1>(16 * j) = finalOutput.select<16, 1>(16 * j) * softMaxSumTemp[j];
    }

#pragma unroll
    for (int j = 0; j < batchnum; j++)
    {
      __ESIMD_ENS::lsc_block_store<
        float,
        16,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((float*)out + outputOffset, finalOutput.select<16, 1>(j * 16));
        outputOffset += 32*128;
    }
  }
}


bool runSlimGQA_Mat(sycl::queue *q, uint8_t* qState, uint8_t* kState, uint8_t* vState, uint8_t* qkvOut, unsigned token_len, unsigned kv_len, unsigned numbOfHead_KV, unsigned numbOfHead_Q, float attn_scale)
{
  int groupH = numbOfHead_KV;
  int localH = 32;
  sycl::range<2> GlobalRange(32 * groupH, 1);
  sycl::range<2> LocalRange(32, 1);
  sycl::nd_range<2> Range(GlobalRange, LocalRange);
  sycl::event e;
  try {
    switch (numbOfHead_KV) {
    case 8:
      if (token_len == 4)
      {
        e = q->submit([&](handler& cgh) {
          cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            qkvFusionGqa_KV8_Q32_batchN<4>(qState, kState, vState, qkvOut, kv_len, attn_scale, ndi);
            });
          });
      }
      else if (token_len == 3)
      {
        e = q->submit([&](handler& cgh) {
          cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            qkvFusionGqa_KV8_Q32_batchN<3>(qState, kState, vState, qkvOut, kv_len, attn_scale, ndi);
            });
          });
      }
      else if (token_len == 2)
      {
        e = q->submit([&](handler& cgh) {
          cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            qkvFusionGqa_KV8_Q32_batchN<2>(qState, kState, vState, qkvOut, kv_len, attn_scale, ndi);
            });
          });
      }
      break;
    default:
      printf("Error: runSlimGQA_Mat doesn't support token_len %d, kv head %d, q head %d\n", token_len, numbOfHead_KV, numbOfHead_Q);
      break;
    }
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool success = true;
  return success;
}


bool runGQA_vec_fusion(sycl::queue* q, uint8_t* query, uint8_t* kCache, uint8_t* vCache, uint8_t* outputs, uint32_t token_len, uint32_t kv_len, uint32_t kv_head, uint32_t q_head, uint32_t q_precision, uint32_t o_precision, float attn_scale) {
    if (token_len > 1 && token_len <=4 && kv_head == 8 && q_head == 32) // special optimization for slim gqa of llama3
    {
       return runSlimGQA_Mat(q, query, kCache, vCache, outputs, token_len, kv_len, kv_head, q_head, attn_scale);
    }
    sycl::event e;
    try {
        const uint32_t localThread = 32;
        sycl::range<2> GlobalRange(localThread * q_head, token_len);
        sycl::range<2> LocalRange(localThread, 1);
        sycl::nd_range<2> Range(GlobalRange, LocalRange);

        if (q_precision == 0 && o_precision == 0)
        {
          e = q->submit([&](handler& cgh) {
              cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                  gqa_kernel_hidden128<localThread, 4, float, float>(query, kCache, vCache, outputs, kv_len, kv_head, q_head, attn_scale, ndi);
                });
              });
        }
        else if (q_precision == 1 && o_precision == 0)
        {
          e = q->submit([&](handler& cgh) {
              cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                  gqa_kernel_hidden128<localThread, 4, fp16, float>(query, kCache, vCache, outputs, kv_len, kv_head, q_head, attn_scale, ndi);
                });
              });
        }
        else if (q_precision == 0 && o_precision == 1)
        {
          e = q->submit([&](handler& cgh) {
              cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                  gqa_kernel_hidden128<localThread, 4, float, fp16>(query, kCache, vCache, outputs, kv_len, kv_head, q_head, attn_scale, ndi);
                });
              });
        }
        else if (q_precision == 1 && o_precision == 1)
        {
          e = q->submit([&](handler& cgh) {
              cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                  gqa_kernel_hidden128<localThread, 4, fp16, fp16>(query, kCache, vCache, outputs, kv_len, kv_head, q_head, attn_scale, ndi);
                });
              });
        }
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    return true;
}
