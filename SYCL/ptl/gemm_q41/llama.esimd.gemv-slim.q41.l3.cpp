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

#define GROUP_SIZE 128

extern "C" bool runLinearQ41_L3(queue & q, unsigned m, unsigned n, unsigned k, uint8_t * a, uint8_t * b, uint8_t * s, uint8_t* z, uint8_t * c);


template<uint32_t batchnum, uint32_t localRange>
ESIMD_INLINE void slimGemmGroup32Block256BatchNRangeN_L3(uint8_t* a, uint8_t* b, uint8_t* s, uint8_t* z, uint8_t* c, uint32_t rowSize, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 16;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
  //uint32_t localRange = ndi.get_local_range(0);
  //constexpr uint32_t alignedLocalRange = (localRange + 3)/4*4;
  constexpr uint32_t commonDim = localRange * 256;
  constexpr uint32_t quantDim = commonDim / 32;

  __ESIMD_NS::slm_init(localRange * 16 * batchnum * sizeof(float));
  int hh = ndi.get_local_id(0);
  int h = ndi.get_group(0);

  int blk_h = h >> 1;
  int inn_h = h & 0x01;
  int offsetABase = (h * pixelPerGroup * commonDim + hh * 64) >> 1;
  int offsetQuanBase = blk_h *32 * quantDim + inn_h * 16 + hh * 2 * 32;//h * quantPerGroup + hh * 2;
  int offsetB = hh * 64 * sizeof(float);
  int outputOffset = pixelPerGroup * h;
  simd<unsigned char, 256> aaa;
  simd<fp16, 16> quant;
  simd<fp16, 16> zps;
  simd<float, 256> bb;
  simd<fp16, 256 * 2> aa;
  simd<float, 32> acc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  simd<uint32_t, 8> baseInc8(baseOffsetInc8);

  offsetA = offsetA * sizeof(uint32_t) + offsetABase;

  simd<uint32_t, 4> offsetQuanX(baseOffsetInc4);
  offsetQuanX = offsetQuanX * localRange * 64 / 32 * 32;
  offsetQuanX = offsetQuanX + offsetQuanBase;
  simd<uint32_t, 16> offsetQuan;
  offsetQuan.select<4, 2>(0) = offsetQuanX;
  offsetQuan.select<4, 2>(1) = offsetQuanX + 32;
  offsetQuan.select<8, 1>(8) = offsetQuan.select<8, 1>(0) + 1;
  offsetQuan = offsetQuan * sizeof(fp16);


#pragma unroll
  for (int k = 0; k < 4; k++) {
    bb.template bit_cast_view<unsigned char>().template select<256, 1>(256 * k) =
      __ESIMD_ENS::lsc_block_load<
      uint8_t,
      256,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + offsetB);

    offsetB += localRange * 64 * sizeof(float);
  }

  for (int n = 0; n < pixelPerGroup; n += 2) {
    quant.select<16, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      fp16,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((fp16*)s, offsetQuan);

    zps.select<16, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      fp16,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((fp16*)z, offsetQuan);

#pragma unroll
    for (int b = 0; b < 8; b++)
    {
      aaa.template bit_cast_view<uint32_t>().template select<8, 1>(b * 8) =
        __ESIMD_ENS::lsc_gather<
        uint32_t,
        1,
        __ESIMD_ENS::lsc_data_size::u32,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached,
        8,
        uint32_t
        >((uint32_t*)a, offsetA);
      offsetA += localRange*64/2;
    }

    simd<float, 256> low = aaa & 0xf;
    simd<float, 256> hig = aaa >> 4;
#pragma unroll
    for (int k = 0; k < 16; k++) {
      aa.select<16, 1>(32 * k) = low.select<16, 1>(16 * k);
      aa.select<16, 1>(32 * k + 16) = hig.select<16, 1>(16 * k);
    }

#pragma unroll
    for (int k = 0; k < 16; k++) {
      aa.select<32, 1>(32 * k) = quant[k] * aa.select<32, 1>(32 * k);
      aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) + zps[k];
    }
    

    {
      simd<float, 32> bcc = 0.0;
#pragma unroll
      for (int k = 0; k < 16; k++) {
        bcc.select<16, 1>(0) += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
      }
#pragma unroll
      for (int k = 0; k < 16; k++) {
        bcc.select<16, 1>(16) += aa.select<16, 1>(16 * k + 256) * bb.select<16, 1>(16 * k);
      }
      acc.select<16, 2>(0) = bcc.select<16, 1>(0);
      acc.select<16, 2>(1) = bcc.select<16, 1>(16);
      acc.select<16, 1>(0) += acc.select<16, 1>(16);
      acc.select<8, 1>(0) += acc.select<8, 1>(8);
      acc.select<4, 1>(0) += acc.select<4, 1>(4);
      acc.select<2, 1>(0) += acc.select<2, 1>(2);
    }
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    slm_block_store<float, 2>(slmAccumulationOffset, acc.select<2, 1>(0));

    offsetQuan += 2 * sizeof(fp16);  // Changed from 128 to 32
  }
  barrier();

  if (hh < batchnum) {
    bb.select<pixelPerGroup, 1>(0) = 0.0;
#pragma unroll
    for (int k = 0; k < localRange; k++)
    {
        bb.select<pixelPerGroup, 1>(0) += slm_block_load<float, pixelPerGroup>(pixelPerGroup * k * sizeof(float) + hh * localRange * pixelPerGroup * sizeof(float));
    }

    __ESIMD_ENS::lsc_block_store<
      float,
      pixelPerGroup,
      __ESIMD_ENS::lsc_data_size::default_size,
      __ESIMD_ENS::cache_hint::write_back,
      __ESIMD_ENS::cache_hint::write_back>((float*)c + outputOffset + hh * rowSize, bb.select<pixelPerGroup, 1>(0));
  }
}

template<uint32_t batchnum>
ESIMD_INLINE void slimGemmGroup32Block256BatchN_L3(uint8_t* a, uint8_t* b, uint8_t* s, uint8_t* z, uint8_t* c, uint32_t commonDim, uint32_t rowSize, nd_item<1>& ndi) {
  constexpr uint32_t pixelPerGroup = 16;
  constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
  constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3};
  uint32_t localRange = ndi.get_local_range(0);
  uint32_t quantPerGroup = commonDim / 32 * pixelPerGroup;
  uint32_t quantDim = commonDim / 32;

  __ESIMD_NS::slm_init(128 * 16 * batchnum * sizeof(float));
  int hh = ndi.get_local_id(0);
  int h = ndi.get_group(0);
  int offsetABase = (h * pixelPerGroup * commonDim + hh * 64) >> 1;
  int blk_h = h >> 1;
  int inn_h = h & 0x01;
  int offsetQuanBase = blk_h *32 * quantDim + inn_h * 16 + hh * 2 * 32;//h * quantPerGroup + hh * 2;

  int offsetB = hh * 64 * sizeof(float);
  int outputOffset = pixelPerGroup * h;
  simd<unsigned char, 256> aaa;
  simd<fp16, 16> quant;
  simd<fp16, 16> zps;
  simd<float, 256> bb;
  simd<fp16, 256 * 2> aa;
  simd<float, 32> acc(0.0f);
  simd<uint32_t, 8> offsetA(baseOffsetInc8);
  offsetA = offsetA * sizeof(uint32_t) + offsetABase;

  simd<uint32_t, 4> offsetQuanX(baseOffsetInc4);
  offsetQuanX = offsetQuanX * localRange * 64 / 32 * 32;
  offsetQuanX.merge(0, offsetQuanX >= commonDim * 32);
  offsetQuanX = offsetQuanX + offsetQuanBase;
  simd<uint32_t, 16> offsetQuan;
  offsetQuan.select<4, 2>(0) = offsetQuanX;
  offsetQuan.select<4, 2>(1) = offsetQuanX + 32;
  offsetQuan.select<8, 1>(8) = offsetQuan.select<8, 1>(0) + 1;
  offsetQuan = offsetQuan * sizeof(fp16);

#pragma unroll
  for (int k = 0; k < 4; k++) {
    if (hh * 64 + k * localRange * 64 < commonDim)
    {
      bb.template bit_cast_view<unsigned char>().template select<256, 1>(256 * k) =
        __ESIMD_ENS::lsc_block_load<
        uint8_t,
        256,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::cached,
        __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + (hh * 64 + k * localRange * 64)*sizeof(float));
    }
    else
    {
      bb.select<64, 1>(k*64) = 0.0;
    }
  }

  for (int n = 0; n < pixelPerGroup; n += 2) {
    quant.select<16, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      fp16,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((fp16*)s, offsetQuan);

    zps.select<16, 1>(0) =
      __ESIMD_ENS::lsc_gather<
      fp16,
      1,
      __ESIMD_ENS::lsc_data_size::u16,
      __ESIMD_ENS::cache_hint::cached,
      __ESIMD_ENS::cache_hint::cached,
      16,
      uint32_t
      >((fp16*)z, offsetQuan);

#pragma unroll
    for (int r = 0; r < 2; r ++)
    {
#pragma unroll
      for (int b = 0; b < 4; b++)
      {
        aaa.template bit_cast_view<uint32_t>().template select<8, 1>(r * 32 + b * 8) =
          __ESIMD_ENS::lsc_gather<
          uint32_t,
          1,
          __ESIMD_ENS::lsc_data_size::u32,
          __ESIMD_ENS::cache_hint::cached,
          __ESIMD_ENS::cache_hint::cached,
          8,
          uint32_t
          >((uint32_t*)a, offsetA + b * localRange * 64 / 2);
      }
      offsetA += commonDim/2;
    }

    simd<float, 256> low = aaa & 0xf;
    simd<float, 256> hig = aaa >> 4;
#pragma unroll
    for (int k = 0; k < 16; k++) {
      aa.select<16, 1>(32 * k) = low.select<16, 1>(16 * k);
      aa.select<16, 1>(32 * k + 16) = hig.select<16, 1>(16 * k);
    }

    //fp32Quant = quant.select<8, 1>(0);
#pragma unroll
    for (int k = 0; k < 16; k++) {
      aa.select<32, 1>(32 * k) = quant[k] * aa.select<32, 1>(32 * k);
      aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) + zps[k];
    }

    //if (aa[0] > 70000.0)
    {
      //cc = 0.0;
      simd<float, 32> bcc = 0.0;
#pragma unroll
      for (int k = 0; k < 16; k++) {
        bcc.select<16, 1>(0) += aa.select<16, 1>(16 * k) * bb.select<16, 1>(16 * k);
      }
#pragma unroll
      for (int k = 0; k < 16; k++) {
        bcc.select<16, 1>(16) += aa.select<16, 1>(16 * k + 256) * bb.select<16, 1>(16 * k);
      }
      acc.select<16, 2>(0) = bcc.select<16, 1>(0);
      acc.select<16, 2>(1) = bcc.select<16, 1>(16);
      //acc.select<32, 1>(0) += acc.select<32, 1>(32);
      acc.select<16, 1>(0) += acc.select<16, 1>(16);
      acc.select<8, 1>(0) += acc.select<8, 1>(8);
      acc.select<4, 1>(0) += acc.select<4, 1>(4);
      acc.select<2, 1>(0) += acc.select<2, 1>(2);
    }
    uint32_t slmAccumulationOffset = (hh * pixelPerGroup + n) * sizeof(float);
    //slm_scalar_store(slmAccumulationOffset, slmAccumulationTemp);
    slm_block_store<float, 2>(slmAccumulationOffset, acc.select<2, 1>(0));


    offsetQuan += 2 * sizeof(fp16);
  }
  barrier();

  for (int j = 0; j <= (batchnum + localRange - 1)/localRange; j ++)
  {
    uint32_t t_idx = j * localRange + hh;
    if (t_idx < batchnum) {
      bb.select<pixelPerGroup, 1>(0) = 0.0;
  #pragma unroll
      for (int k = 0; k < localRange; k++)
      {
          bb.select<pixelPerGroup, 1>(0) += slm_block_load<float, pixelPerGroup>(pixelPerGroup * k * sizeof(float) + t_idx * localRange * pixelPerGroup * sizeof(float));
      }

      __ESIMD_ENS::lsc_block_store<
        float,
        pixelPerGroup,
        __ESIMD_ENS::lsc_data_size::default_size,
        __ESIMD_ENS::cache_hint::write_back,
        __ESIMD_ENS::cache_hint::write_back>((float*)c + outputOffset + t_idx * rowSize, bb.select<pixelPerGroup, 1>(0));
    }
  }
}

template<uint32_t localRange>
static inline sycl::event executeSlimRunLinearBlock256RangeN_L3(queue & q, uint8_t*a, uint8_t*b, uint8_t*s, uint8_t *z, uint8_t*c, unsigned m, unsigned n, unsigned k)
{
    sycl::event e;
    int groups = (m + 15) / 16;

    sycl::range<1> GlobalRange(groups * localRange);
    sycl::range<1> LocalRange(localRange);
    sycl::nd_range<1> RangeV2(GlobalRange, localRange);
    if (k == 1)
    {
      e = q.submit([&](handler& cgh) {
        cgh.parallel_for(RangeV2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
          slimGemmGroup32Block256BatchNRangeN_L3<1, localRange>((uint8_t*)a, (uint8_t*)b, (uint8_t*)s, (uint8_t*)z, (uint8_t*)c, m, ndi);
          });
        });
    }
    else
    {
      printf("Error! Q41 GEMV only supports k = 1 right now.\n");
    }

    return e;
}

static inline sycl::event executeSlimRunLinearBlock256_L3(queue & q, uint8_t*a, uint8_t*b, uint8_t*s, uint8_t*z, uint8_t*c, unsigned m, unsigned n, unsigned k)
{
    sycl::event e;
    int groups = (m + 15) / 16;

    uint32_t localRange = (n + 255)/ 256;

    sycl::range<1> GlobalRange(groups * localRange);
    sycl::range<1> LocalRange(localRange);
    sycl::nd_range<1> RangeV2(GlobalRange, localRange);
    if (k == 1)
    {
      e = q.submit([&](handler& cgh) {
        cgh.parallel_for(RangeV2, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
          slimGemmGroup32Block256BatchN_L3<1>((uint8_t*)a, (uint8_t*)b, (uint8_t*)s, (uint8_t*)z, (uint8_t*)c, n, m, ndi);
          });
        });
    }
    else
    {
      printf("Error! Q41 GEMV only supports k = 1 right now.\n");
    }

    return e;
}


bool runLinearQ41_L3(queue & q, unsigned m, unsigned n, unsigned k, uint8_t * a, uint8_t * b, uint8_t * s, uint8_t * z, uint8_t * c)
{
  if (n%64>0)
  {
    printf("Error! Only multiplies of 64 are supported for common dim! %d not supported\n", n);
  }
 
  sycl::event e;
  // Launches the task on the GPU.
  try {
    switch(n)
    {
      // RangeN = n / 256;
      case 1536:
        e = executeSlimRunLinearBlock256RangeN_L3<6>(q, a, b, s, z, c, m, n, k);
        break;
      case 2048:
        e = executeSlimRunLinearBlock256RangeN_L3<8>(q, a, b, s, z, c, m, n, k);
        break;
      case 2560:
        e = executeSlimRunLinearBlock256RangeN_L3<10>(q, a, b, s, z, c, m, n, k);
        break;
      case 3072:
        e = executeSlimRunLinearBlock256RangeN_L3<12>(q, a, b, s, z, c, m, n, k);
        break;
      case 3584:
        e = executeSlimRunLinearBlock256RangeN_L3<14>(q, a, b, s, z, c, m, n, k);
        break;
      case 4096:
        e = executeSlimRunLinearBlock256RangeN_L3<16>(q, a, b, s, z, c, m, n, k);
        break;
      case 6144:
        e = executeSlimRunLinearBlock256RangeN_L3<24>(q, a, b, s, z, c, m, n, k);
        break;
      case 8192:
        e = executeSlimRunLinearBlock256RangeN_L3<32>(q, a, b, s, z, c, m, n, k);
        break;
      case 8960:
        e = executeSlimRunLinearBlock256RangeN_L3<35>(q, a, b, s, z, c, m, n, k);
        break;
      case 9728:
        e = executeSlimRunLinearBlock256RangeN_L3<38>(q, a, b, s, z, c, m, n, k);
        break;
      case 11008:
        e = executeSlimRunLinearBlock256RangeN_L3<43>(q, a, b, s, z, c, m, n, k);
        break;
      case 12288:
        e = executeSlimRunLinearBlock256RangeN_L3<48>(q, a, b, s, z, c, m, n, k);
        break;
      case 14336:
        e = executeSlimRunLinearBlock256RangeN_L3<56>(q, a, b, s, z, c, m, n, k);
        break;
      case 18944:
        e = executeSlimRunLinearBlock256RangeN_L3<74>(q, a, b, s, z, c, m, n, k);
        break;
      default:
        e = executeSlimRunLinearBlock256_L3(q, a, b, s, z, c, m, n, k);
        break;
    }
  } catch (sycl::exception const& e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return false;
  }

  bool success = true;
  return success;

}