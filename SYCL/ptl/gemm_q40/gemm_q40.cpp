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

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

#include <map>

#include <windows.h>

using namespace sycl;
using fp16 = sycl::half;
#define DEVICE_MEM_ALIGNMENT 4096;

using namespace std;
using namespace sycl::ext::intel::esimd;

#define GROUP_SIZE 128

extern "C" bool runGemmESimd_Q40Weights_L1(queue& q, unsigned m, unsigned n, unsigned k, uint8_t* a, uint8_t* b, uint8_t* s, uint8_t* c, uint8_t* shuffleTt);

template <uint32_t startBlobA, uint32_t startBlobB>
ESIMD_INLINE void mmaXve8x8x16(simd<fp16, 64 * 16> aa, simd<fp16, 32 * 8> bb, simd<fp16, 8 * 16>& cc)
{
#pragma unroll
    for (int kk = 0; kk < 8; kk++) {
#pragma unroll
        for (int kkk = 0; kkk < 8; kkk++) {
            cc.select<16, 1>(16 * kkk) += aa.select<16, 1>(32 * 16 * startBlobA + (startBlobB * 8 + kk) * 16) * bb[startBlobB * 64 + kk * 8 + kkk];
        }
    }
}

// shuffle (k, n) fp32 input to (n // 2048, aligned(k, 8) , 2048) shape
ESIMD_INLINE void fp32ShuffleToFp16Quant(uint8_t* a, uint8_t* b, uint32_t hiddenDim, uint32_t tokenLength, nd_item<2>& ndi)
{
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    int h = ndi.get_group(0); // [0, n // 32)
    int v = ndi.get_group(1); // [0, aligned(k, 8) // 8)
    int hh = ndi.get_local_id(0);
    int alignedTokenSize = (tokenLength + 7) >> 3;
    alignedTokenSize = alignedTokenSize << 3;
    int oobHiddenDim = hiddenDim & 0x7ff;
    int hiddenDimGroup = hiddenDim >> 11;
    uint32_t baseOffset = (v * 8 * hiddenDim + hh * 32 + h * 2048) * sizeof(float);
    int offsetOut = h * alignedTokenSize * 2048 + hh * 8 * 32;
    int offsetBaseQuantOutput = alignedTokenSize * hiddenDim + h * alignedTokenSize * (2048 / 64) + hh;
    uint32_t loadStepping;
    uint32_t inboundThreads;
    simd<uint32_t, 8> offsetVecQuantOutput(baseOffsetInc8);

    if (h < hiddenDimGroup) {
        offsetOut += v * 8 * 2048;
        offsetBaseQuantOutput += v * 8 * (32);
        inboundThreads = 32;
    } else {
        offsetOut += v * 8 * oobHiddenDim;
        offsetBaseQuantOutput += v * 8 * (oobHiddenDim >> 6);
        inboundThreads = oobHiddenDim >> 6;
    }

    offsetVecQuantOutput = offsetVecQuantOutput * inboundThreads * sizeof(fp16) + offsetBaseQuantOutput * sizeof(fp16);

    loadStepping = (inboundThreads - 1) * 32 * sizeof(uint32_t);
    simd<uint32_t, 8> offsetIn(baseOffsetInc8);
    simd<float, 512> fp32Input;
    simd<fp16, 512> fp16Output;
    simd<float, 16> maxAbs = 1.0f / 32.0f;
    offsetIn = offsetIn * hiddenDim * sizeof(uint32_t) + baseOffset;

    if (hh < inboundThreads) {
#pragma unroll
        for (int l = 0; l < 2; l++) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
                fp32Input.template bit_cast_view<uint32_t>().template select<64, 1>(256 * l + 64 * k) = __ESIMD_ENS::lsc_gather<
                    uint32_t,
                    8,
                    __ESIMD_ENS::lsc_data_size::u32,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached,
                    8,
                    uint32_t>((uint32_t*)a, offsetIn);
                offsetIn += 8 * sizeof(uint32_t);
            }

#pragma unroll
            for (int kk = 0; kk < 16; kk++) {
                simd<float, 16> maxAbsTemp = abs<float, 16>(fp32Input.select<16, 1>(256 * l + 16 * kk));
                maxAbs = max<float, 16>(maxAbs, maxAbsTemp);
            }

            offsetIn += loadStepping;
        }

        simd<float, 8> maxLog2;
        simd<float, 16> quant;
        simd<fp16, 8> outputQuant;
        maxAbs.select<8, 1>(0) = max<float, 8>(maxAbs.select<8, 1>(0), maxAbs.select<8, 1>(8));
        maxLog2 = log2<float, 8, saturation_off_tag>(maxAbs.select<8, 1>(0)); // log<float, 8, float>(maxAbs.select<8, 1>(0));
        // maxLog2 = max<float, 8>(maxLog2, 0.0f);
        maxLog2 = rndu<float, 8>(maxLog2);
        maxLog2 = 11.0f - maxLog2;
        maxLog2 = min<float, 8>(maxLog2, 5.0f);
        maxLog2 = max<float, 8>(maxLog2, -5.0f);
        quant.select<8, 1>(0) = pow<float, 8, float>(2.0f, maxLog2);
        quant.select<8, 1>(8) = quant.select<8, 1>(0);
        maxLog2 = -1.0f * maxLog2;
        maxLog2 = pow<float, 8, float>(2.0f, maxLog2);
        outputQuant = maxLog2.select<8, 1>(0);

        __ESIMD_ENS::lsc_scatter<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::write_back,
            __ESIMD_ENS::cache_hint::write_back,
            8,
            uint32_t>((fp16*)b, offsetVecQuantOutput, outputQuant.select<8, 1>(0));

#pragma unroll
        for (int k = 0; k < 32; k++) {
            fp16Output.select<16, 1>(16 * k) = fp32Input.select<16, 1>(16 * k) * quant;
        }

#pragma unroll
        for (int l = 0; l < 2; l++) {
#pragma unroll
            for (int k = 0; k < 2; k++) {
                __ESIMD_ENS::lsc_block_store<
                    fp16,
                    128,
                    __ESIMD_ENS::lsc_data_size::default_size,
                    __ESIMD_ENS::cache_hint::write_back,
                    __ESIMD_ENS::cache_hint::write_back>((fp16*)b + offsetOut + 128 * k + 256 * inboundThreads * l, fp16Output.select<128, 1>(256 * l + 128 * k));
            }
        }
    }
}

ESIMD_INLINE void gemmReduce2048WeightsQ40Group128InputShffuledFp16Quantized(uint8_t* a, uint8_t* b, uint8_t* s, uint8_t* c, uint8_t* quantB, int hiddenDim, int tokenSize, int reduceIdx, int lastReduce, nd_item<2>& ndi)
{
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
    __ESIMD_NS::slm_init(32 * 8 * 16 * 2 * sizeof(fp16));
    int hh = ndi.get_local_linear_id(); // [0, 64)
    int h = ndi.get_group(0); // [0, (row + 15) / 16)
    int v = reduceIdx; // [0, (row + 15) / 16)
    int outputRow = ndi.get_group_range(0) * 16;
    int hiddenDimInt4Size = hiddenDim >> 1;
    int hiddenDimDequantSize = hiddenDim >> 7;
    uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
    uint32_t baseOffsetA = globalOffset >> 1;
    uint32_t baseOffsetQuant = v * 16 * sizeof(fp16) + h * (hiddenDim >> 2) + (hh >> 2) * sizeof(fp16);
    uint32_t baseOffsetB = hh * 256 * sizeof(fp16);
    uint32_t baseOffsetQuantB = hh * 32 * sizeof(fp16);
    uint32_t offsetC = h * 16 + hh * outputRow;
    simd<fp16, 8 * 32> bb;
    simd<fp16, 64 * 16> aa;
    simd<fp16, 8 * 16> cc;
    simd<uint32_t, 16> offset(baseOffsetInc16);
    simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
    uint32_t loopCount = (tokenSize + 7) >> 3;

    offsetQuant = offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;
#pragma unroll
    for (int k = 0; k < 2; k++) {
        cc.select<16, 1>(16 * k) = __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)s, offsetQuant);

        offsetQuant += 8 * sizeof(fp16);
    }

    offset = offset * hiddenDimInt4Size + baseOffsetA;

#pragma unroll
    for (int k = 0; k < 2; k++) {
        bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * k) = __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            simd<uint8_t, 16> bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 0);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 0) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 0) * 16 + 16 * 16) = bitShiftTemp >> 4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 1);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 1) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 1) * 16 + 16 * 16) = bitShiftTemp >> 0x4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 2);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 2) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 2) * 16 + 16 * 16) = bitShiftTemp >> 4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 3);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 3) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 3) * 16 + 16 * 16) = bitShiftTemp >> 4;
        }
        offset += 128 * sizeof(uint32_t);
    }

    aa = aa - (fp16)8.0f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(0);
    }

#pragma unroll
    for (int k = 32; k < 64; k++) {
        aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
    }

    for (int nn = 0; nn < loopCount; nn++) {
        uint32_t slmPingPong = nn & 0x1;
        cc = 0;
        auto quantBFp16 = offsetQuant.bit_cast_view<fp16>();
        if (hh < 8) {
            quantBFp16.template bit_cast_view<uint8_t>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
                uint8_t,
                64,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((uint8_t*)quantB + baseOffsetQuantB);

            baseOffsetQuantB += 8 * 32 * sizeof(fp16);
        }
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(0) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 256;
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(256) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 512 * 32 - 256;

        mmaXve8x8x16<0, 0>(aa, bb, cc);
        mmaXve8x8x16<0, 1>(aa, bb, cc);
        mmaXve8x8x16<0, 2>(aa, bb, cc);
        mmaXve8x8x16<0, 3>(aa, bb, cc);

        bb.template bit_cast_view<uint8_t>().template select<256, 1>(0) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 256;
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(256) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 512 * 32 - 256;

        mmaXve8x8x16<1, 0>(aa, bb, cc);
        mmaXve8x8x16<1, 1>(aa, bb, cc);
        mmaXve8x8x16<1, 2>(aa, bb, cc);
        mmaXve8x8x16<1, 3>(aa, bb, cc);

#pragma unroll
        for (int k = 0; k < 8; k++) {
            slm_block_store<fp16, 16>((hh * 16 + k * 16 * 32 + slmPingPong * 16 * 32 * 8) * sizeof(fp16), cc.select<16, 1>(16 * k));
        }

        barrier();

        if (hh < 8) {
            if (8 * nn + hh < tokenSize) {
                if (v != 0) {
                    cc.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
                        fp16,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached>((fp16*)c + offsetC * 2);
                } else {
                    cc.select<16, 1>(0) = 0;
                }
                uint32_t slmOffset = hh * 16 * 32 * sizeof(fp16) + slmPingPong * 16 * 32 * 8 * sizeof(fp16);
                // #pragma unroll
                //         for (int k = 0; k < 2; k++) {
                //           bb.template bit_cast_view<fp16>().template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(slmOffset + k * 128 * sizeof(fp16));
                //         }
                // #pragma unroll
                //         for (int k = 0; k < 16; k++) {
                //           cc.select<16, 1>(0) += quantBFp16[k] * bb.template bit_cast_view<fp16>().template select<16, 1>(16 * k);
                //         }
                // #pragma unroll
                //         for (int k = 0; k < 2; k++) {
                //           bb.template bit_cast_view<fp16>().template select<128, 1>(128 * k) = slm_block_load<fp16, 128>(slmOffset + 256 * sizeof(fp16) + k * 128 * sizeof(fp16));
                //         }
                // #pragma unroll
                //         for (int k = 0; k < 16; k++) {
                //           cc.select<16, 1>(0) += quantBFp16[16 + k] * bb.template bit_cast_view<fp16>().template select<16, 1>(16 * k);
                //         }
                // aa.select<32 * 16, 1>(0) = slm_block_load<fp16, 32 * 16>(slmOffset);

#pragma unroll
                for (int k = 0; k < 32; k++) {
                    simd<fp16, 16> temp = slm_block_load<fp16, 16>(slmOffset + k * 16 * sizeof(fp16));
                    cc.select<16, 1>(0) = cc.select<16, 1>(0) + quantBFp16[k] * temp.select<16, 1>(0);
                }

                if (!lastReduce) {
                    __ESIMD_ENS::lsc_block_store<
                        fp16,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::write_back,
                        __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC * 2, cc.select<16, 1>(0));
                } else {
                    simd<float, 16> outputTemp = cc.select<16, 1>(0);
                    __ESIMD_ENS::lsc_block_store<
                        float,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::write_back,
                        __ESIMD_ENS::cache_hint::write_back>((float*)c + offsetC, outputTemp);
                }

                offsetC += 8 * outputRow;
            }
        }
    }
}

template <uint32_t localRange>
ESIMD_INLINE void gemm_reduce_Q40_F16Q_RangeN(uint8_t* a, uint8_t* b, uint8_t* s, uint8_t* c, uint8_t* quantB, int reduceDim, int hiddenDim, int tokenSize, int reduceIdx, int lastReduce, nd_item<2>& ndi)
{
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
    __ESIMD_NS::slm_init(32 * 8 * 16 * 2 * sizeof(fp16));
    int hh = ndi.get_local_linear_id(); // [0, 64)
    int h = ndi.get_group(0); // [0, (row + 15) / 16)
    int v = reduceIdx; // [0, (row + 15) / 16)
    int outputRow = ndi.get_group_range(0) * 16;
    int hiddenDimInt4Size = hiddenDim >> 1;
    int hiddenDimDequantSize = hiddenDim >> 7;
    uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
    uint32_t baseOffsetA = globalOffset >> 1;
    uint32_t baseOffsetQuant = v * 16 * sizeof(fp16) + h * (hiddenDim >> 2) + (hh >> 2) * sizeof(fp16);
    uint32_t baseOffsetB = hh * 256 * sizeof(fp16);
    uint32_t baseOffsetQuantB = hh * reduceDim / 64 * sizeof(fp16);
    uint32_t offsetC = h * 16 + hh * outputRow;
    simd<fp16, 8 * 32> bb;
    simd<fp16, 64 * 16> aa;
    simd<fp16, 8 * 16> cc;
    simd<uint32_t, 16> offset(baseOffsetInc16);
    simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
    uint32_t loopCount = (tokenSize + 7) >> 3;

    offsetQuant = offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;
#pragma unroll
    for (int k = 0; k < 2; k++) {
        cc.select<16, 1>(16 * k) = __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)s, offsetQuant);

        offsetQuant += (localRange >> 2) * sizeof(fp16); // localRange * 32 / 128 * sizeof(fp16)
    }

    offset = offset * hiddenDimInt4Size + baseOffsetA;

#pragma unroll
    for (int k = 0; k < 2; k++) {
        bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * k) = __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            simd<uint8_t, 16> bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 0);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 0) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 0) * 16 + 16 * 16) = bitShiftTemp >> 4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 1);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 1) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 1) * 16 + 16 * 16) = bitShiftTemp >> 0x4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 2);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 2) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 2) * 16 + 16 * 16) = bitShiftTemp >> 4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 3);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 3) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 3) * 16 + 16 * 16) = bitShiftTemp >> 4;
        }
        offset += (localRange << 4); // localRange * 32 / 2
    }

    aa = aa - (fp16)8.0f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(0);
    }

#pragma unroll
    for (int k = 32; k < 64; k++) {
        aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
    }

    for (int nn = 0; nn < loopCount; nn++) {
        uint32_t slmPingPong = nn & 0x1;
        cc = 0;
        auto quantBFp16 = offsetQuant.bit_cast_view<fp16>();
        if (hh < 8) {
            quantBFp16.template bit_cast_view<uint8_t>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
                uint8_t,
                64,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((uint8_t*)quantB + baseOffsetQuantB);

            baseOffsetQuantB += 8 * reduceDim / 64 * sizeof(fp16);
        }
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(0) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 256;
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(256) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 512 * localRange - 256;

        mmaXve8x8x16<0, 0>(aa, bb, cc);
        mmaXve8x8x16<0, 1>(aa, bb, cc);
        mmaXve8x8x16<0, 2>(aa, bb, cc);
        mmaXve8x8x16<0, 3>(aa, bb, cc);

        bb.template bit_cast_view<uint8_t>().template select<256, 1>(0) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 256;
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(256) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 512 * localRange - 256;

        mmaXve8x8x16<1, 0>(aa, bb, cc);
        mmaXve8x8x16<1, 1>(aa, bb, cc);
        mmaXve8x8x16<1, 2>(aa, bb, cc);
        mmaXve8x8x16<1, 3>(aa, bb, cc);

#pragma unroll
        for (int k = 0; k < 8; k++) {
            slm_block_store<fp16, 16>((hh * 16 + k * 16 * 32 + slmPingPong * 16 * 32 * 8) * sizeof(fp16), cc.select<16, 1>(16 * k));
        }

        barrier();

        if (hh < 8) {
            if (8 * nn + hh < tokenSize) {
                if (v != 0) {
                    cc.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
                        fp16,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached>((fp16*)c + offsetC * 2);
                } else {
                    cc.select<16, 1>(0) = 0;
                }
                uint32_t slmOffset = hh * 16 * 32 * sizeof(fp16) + slmPingPong * 16 * 32 * 8 * sizeof(fp16);
                // aa.select<localRange * 16, 1>(0) = slm_block_load<fp16, localRange * 16>(slmOffset);
                simd<fp16, localRange * 16> temp = slm_block_load<fp16, localRange * 16>(slmOffset);

#pragma unroll
                for (int k = 0; k < localRange; k++) {
                    cc.select<16, 1>(0) = cc.select<16, 1>(0) + quantBFp16[k] * temp.template select<16, 1>(16 * k);
                }
                if (!lastReduce) {
                    __ESIMD_ENS::lsc_block_store<
                        fp16,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::write_back,
                        __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC * 2, cc.select<16, 1>(0));
                } else {
                    simd<float, 16> outputTemp = cc.select<16, 1>(0);
                    __ESIMD_ENS::lsc_block_store<
                        float,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::write_back,
                        __ESIMD_ENS::cache_hint::write_back>((float*)c + offsetC, outputTemp);
                }

                offsetC += 8 * outputRow;
            }
        }
    }
}

ESIMD_INLINE void gemm_reduce_Q40_F16Q(uint8_t* a, uint8_t* b, uint8_t* s, uint8_t* c, uint8_t* quantB, int reduceDim, int hiddenDim, int tokenSize, int reduceIdx, int lastReduce, nd_item<2>& ndi)
{
    constexpr uint32_t baseOffsetInc16[16] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    constexpr uint32_t baseOffsetInc4[4] = { 0, 1, 2, 3 };
    __ESIMD_NS::slm_init(32 * 8 * 16 * 2 * sizeof(fp16));
    int hh = ndi.get_local_linear_id(); // [0, 64)
    int h = ndi.get_group(0); // [0, (row + 15) / 16)
    int v = reduceIdx; // [0, (row + 15) / 16)
    int outputRow = ndi.get_group_range(0) * 16;
    uint32_t localRange = ndi.get_local_range(0);
    int hiddenDimInt4Size = hiddenDim >> 1;
    int hiddenDimDequantSize = hiddenDim >> 7;
    uint32_t globalOffset = v * 2048 + h * hiddenDim * 16 + hh * 32;
    uint32_t baseOffsetA = globalOffset >> 1;
    uint32_t baseOffsetQuant = v * 16 * sizeof(fp16) + h * (hiddenDim >> 2) + (hh >> 2) * sizeof(fp16);
    uint32_t baseOffsetB = hh * 256 * sizeof(fp16);
    uint32_t baseOffsetQuantB = hh * reduceDim / 64 * sizeof(fp16);
    uint32_t offsetC = h * 16 + hh * outputRow;
    simd<fp16, 8 * 32> bb;
    simd<fp16, 64 * 16> aa;
    simd<fp16, 8 * 16> cc;
    simd<uint32_t, 16> offset(baseOffsetInc16);
    simd<uint32_t, 16> offsetQuant(baseOffsetInc16);
    uint32_t loopCount = (tokenSize + 7) >> 3;

    offsetQuant = offsetQuant * hiddenDimDequantSize * sizeof(fp16) + baseOffsetQuant;
#pragma unroll
    for (int k = 0; k < 2; k++) {
        cc.select<16, 1>(16 * k) = __ESIMD_ENS::lsc_gather<
            fp16,
            1,
            __ESIMD_ENS::lsc_data_size::u16,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((fp16*)s, offsetQuant);

        offsetQuant += (localRange >> 2) * sizeof(fp16); // localRange * 32 / 128 * sizeof(fp16)
    }

    offset = offset * hiddenDimInt4Size + baseOffsetA;

#pragma unroll
    for (int k = 0; k < 2; k++) {
        bb.template bit_cast_view<uint32_t>().template select<64, 1>(64 * k) = __ESIMD_ENS::lsc_gather<
            uint32_t,
            4,
            __ESIMD_ENS::lsc_data_size::u32,
            __ESIMD_ENS::cache_hint::uncached,
            __ESIMD_ENS::cache_hint::uncached,
            16,
            uint32_t>((uint32_t*)a, offset);

#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            simd<uint8_t, 16> bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 0);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 0) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 0) * 16 + 16 * 16) = bitShiftTemp >> 4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 1);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 1) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 1) * 16 + 16 * 16) = bitShiftTemp >> 0x4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 2);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 2) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 2) * 16 + 16 * 16) = bitShiftTemp >> 4;
            bitShiftTemp = bb.template bit_cast_view<uint8_t>().select<16, 4>(256 * k + 64 * kk + 3);
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 3) * 16) = bitShiftTemp & 0xf;
            aa.select<16, 1>(32 * 16 * k + (4 * kk + 3) * 16 + 16 * 16) = bitShiftTemp >> 4;
        }
        offset += (localRange << 4); // localRange * 32 / 2
    }

    aa = aa - (fp16)8.0f;
#pragma unroll
    for (int k = 0; k < 32; k++) {
        aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(0);
    }

#pragma unroll
    for (int k = 32; k < 64; k++) {
        aa.select<16, 1>(16 * k) = aa.select<16, 1>(16 * k) * cc.select<16, 1>(16);
    }

    for (int nn = 0; nn < loopCount; nn++) {
        uint32_t slmPingPong = nn & 0x1;
        cc = 0;
        auto quantBFp16 = offsetQuant.bit_cast_view<fp16>();
        if (hh < 8) {
            quantBFp16.template bit_cast_view<uint8_t>().template select<64, 1>(0) = __ESIMD_ENS::lsc_block_load<
                uint8_t,
                64,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::cached,
                __ESIMD_ENS::cache_hint::cached>((uint8_t*)quantB + baseOffsetQuantB);

            baseOffsetQuantB += 8 * reduceDim / 64 * sizeof(fp16);
        }
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(0) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 256;
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(256) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 512 * localRange - 256;

        mmaXve8x8x16<0, 0>(aa, bb, cc);
        mmaXve8x8x16<0, 1>(aa, bb, cc);
        mmaXve8x8x16<0, 2>(aa, bb, cc);
        mmaXve8x8x16<0, 3>(aa, bb, cc);

        bb.template bit_cast_view<uint8_t>().template select<256, 1>(0) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 256;
        bb.template bit_cast_view<uint8_t>().template select<256, 1>(256) = __ESIMD_ENS::lsc_block_load<
            uint8_t,
            256,
            __ESIMD_ENS::lsc_data_size::default_size,
            __ESIMD_ENS::cache_hint::cached,
            __ESIMD_ENS::cache_hint::cached>((uint8_t*)b + baseOffsetB);
        baseOffsetB += 512 * localRange - 256;

        mmaXve8x8x16<1, 0>(aa, bb, cc);
        mmaXve8x8x16<1, 1>(aa, bb, cc);
        mmaXve8x8x16<1, 2>(aa, bb, cc);
        mmaXve8x8x16<1, 3>(aa, bb, cc);

#pragma unroll
        for (int k = 0; k < 8; k++) {
            slm_block_store<fp16, 16>((hh * 16 + k * 16 * 32 + slmPingPong * 16 * 32 * 8) * sizeof(fp16), cc.select<16, 1>(16 * k));
        }

        barrier();

        if (hh < 8) {
            if (8 * nn + hh < tokenSize) {
                if (v != 0) {
                    cc.select<16, 1>(0) = __ESIMD_ENS::lsc_block_load<
                        fp16,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached>((fp16*)c + offsetC * 2);
                } else {
                    cc.select<16, 1>(0) = 0;
                }
                uint32_t slmOffset = hh * 16 * 32 * sizeof(fp16) + slmPingPong * 16 * 32 * 8 * sizeof(fp16);
                simd<fp16, 512> temp = slm_block_load<fp16, 512>(slmOffset);
                for (int k = 0; k < localRange; k++) {
                    cc.select<16, 1>(0) = cc.select<16, 1>(0) + quantBFp16[k] * temp.select<16, 1>(k * 16);
                }
                if (!lastReduce) {
                    __ESIMD_ENS::lsc_block_store<
                        fp16,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::write_back,
                        __ESIMD_ENS::cache_hint::write_back>((fp16*)c + offsetC * 2, cc.select<16, 1>(0));
                } else {
                    simd<float, 16> outputTemp = cc.select<16, 1>(0);
                    __ESIMD_ENS::lsc_block_store<
                        float,
                        16,
                        __ESIMD_ENS::lsc_data_size::default_size,
                        __ESIMD_ENS::cache_hint::write_back,
                        __ESIMD_ENS::cache_hint::write_back>((float*)c + offsetC, outputTemp);
                }

                offsetC += 8 * outputRow;
            }
        }
    }
}

bool runGemmESimd_Q40Weights_L1(queue& q, unsigned m, unsigned n, unsigned k, uint8_t* a, uint8_t* b, uint8_t* s, uint8_t* c, uint8_t* shuffleTt)
{
    if (n % 64 != 0) {
        std::cout << "None supported common dimension = " << n << std::endl;
        return false;
    }
    size_t alignedTokenSize = (k + 7) / 8;
    alignedTokenSize = alignedTokenSize * 8;

    uint32_t num_2048 = n / 2048;
    uint32_t tailing = n % 2048;

    int groupShuffleQuantH = (n + 2047) / 2048;
    int groupShuffleQuantV = alignedTokenSize / 8;
    int localShuffleQuantH = 32;
    int localShuffleQuantV = 1;
    sycl::range<2> GlobalRangeShuffleQuant(groupShuffleQuantH * localShuffleQuantH, groupShuffleQuantV * localShuffleQuantV);
    sycl::range<2> LocalRangeShuffleQuant(localShuffleQuantH, localShuffleQuantV);
    sycl::nd_range<2> RangeShuffleQuant(GlobalRangeShuffleQuant, LocalRangeShuffleQuant);
    int groupReduce2048H = (m + 15) / 16;
    int groupReduce2048V = 1;
    int localReduce2048H = 2048 / 64;
    int localReduce2048V = 1;
    sycl::range<2> GlobalRangeReduce2048(groupReduce2048H * localReduce2048H, groupReduce2048V * localReduce2048V);
    sycl::range<2> LocalRangeReduce2048(localReduce2048H, localReduce2048V);
    sycl::nd_range<2> RangeReduce2048(GlobalRangeReduce2048, LocalRangeReduce2048);
    int groupReduceTailH = (m + 15) / 16;
    int groupReduceTailV = 1;
    int localReduceTailH = tailing / 64;
    int localReduceTailV = 1;
    sycl::range<2> GlobalRangeReduceTail(groupReduceTailH * localReduceTailH, groupReduceTailV * localReduceTailV);
    sycl::range<2> LocalRangeReduceTail(localReduceTailH, localReduceTailV);
    sycl::nd_range<2> RangeReduceTail(GlobalRangeReduceTail, LocalRangeReduceTail);

    // Launches the task on the GPU.
    try {
        sycl::event e;
        int lastReduce = 0;
        e = q.submit([&](handler& cgh) {
            cgh.parallel_for(RangeShuffleQuant, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                fp32ShuffleToFp16Quant((uint8_t*)b, (uint8_t*)shuffleTt, n, k, ndi);
            });
        });
        uint8_t* quantB = shuffleTt + n * alignedTokenSize * sizeof(fp16);
        for (int r = 0; r < num_2048; r++) {
            int lastReduce = (tailing == 0 && r == num_2048 - 1);
            e = q.submit([&](handler& cgh) {
                cgh.parallel_for(RangeReduce2048, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    gemm_reduce_Q40_F16Q_RangeN<32>((uint8_t*)a,
                        (uint8_t*)shuffleTt + r * alignedTokenSize * 2048 * sizeof(fp16),
                        (uint8_t*)s,
                        (uint8_t*)c,
                        (uint8_t*)quantB + r * alignedTokenSize * (2048 / 64) * sizeof(fp16),
                        2048,
                        n,
                        k,
                        r,
                        lastReduce,
                        ndi);
                });
            });
        }
        if (tailing > 0) {
            e = q.submit([&](handler& cgh) {
                cgh.parallel_for(RangeReduceTail, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL {
                    gemm_reduce_Q40_F16Q((uint8_t*)a,
                        (uint8_t*)shuffleTt + num_2048 * alignedTokenSize * 2048 * sizeof(fp16),
                        (uint8_t*)s,
                        (uint8_t*)c,
                        (uint8_t*)quantB + num_2048 * alignedTokenSize * (2048 / 64) * sizeof(fp16),
                        tailing,
                        n,
                        k,
                        num_2048,
                        1,
                        ndi);
                });
            });
        }
    } catch (sycl::exception const& e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    bool success = true;
    return success;
}
