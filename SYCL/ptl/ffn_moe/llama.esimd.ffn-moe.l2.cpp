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

extern "C" bool __declspec(dllexport) runFfnMoeFusionVec_L2(sycl::queue* q, uint8_t* inputs, uint8_t* router, uint8_t* up, uint8_t* gate, uint8_t* down, uint8_t *outputs, uint32_t total_experts, uint32_t selected_experts, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, uint8_t* shuffleTt);

template<typename IT, uint32_t localRange>
static void runRouter(sycl::queue& q, IT *inputs, float *weights, float *outputs, uint32_t input_len, uint32_t total_experts)
{
    const uint32_t outputPerGroup = 16;
    int globalGroup = (total_experts + outputPerGroup - 1) / outputPerGroup;
    int localThread = localRange;
    sycl::range<1> GlobalRange(globalGroup * localThread);
    sycl::range<1> LocalRange(localThread);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * outputPerGroup * localRange);

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);
            
            simd<IT, 128> in;
            simd<float, 128> wei;

            simd<float, outputPerGroup> out;

            uint32_t input_offset = hh * 64;
            in.template select<64, 1>(0) = block_load<IT, 64>(inputs + input_offset);
            input_offset += localRange * 64;
            in.template select<64, 1>(64) = block_load<IT, 64>(inputs + input_offset);

            uint32_t weights_offset = h * outputPerGroup * input_len + hh * 64;
        #pragma unroll
            for (int j = 0; j < outputPerGroup; j++)
            {
                wei.select<64, 1>(0) = block_load<float, 64>(weights + weights_offset);
                weights_offset += localRange * 64;
                wei.select<64, 1>(64) = block_load<float, 64>(weights + weights_offset);
                weights_offset += localRange * 64;

                simd<float, 32> temp = 0;
        #pragma unroll
                for (int k = 0; k < 4; k++)
                {
                    temp += in.template select<32, 1>(32 * k) * wei.select<32, 1>(32 * k);
                }

                out[j] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(temp);
            }
            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup, out);
            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> acc = 0.0;
        #pragma unroll
                for (int j = 0; j < localRange; j++)
                {
                    acc += slm_block_load<float, outputPerGroup>(sizeof(float) * j * outputPerGroup);
                }

                block_store<float, outputPerGroup>(outputs + h * outputPerGroup, acc);
            }

        });
    });
}

#define FP32_MIN (-1.7e+38)
static void topKAndWeights(sycl::queue& q, float* inputs, uint32_t* indexes, float* weights, uint32_t token_len, uint32_t total_experts, uint32_t selected_experts)
{
    int globalGroup = token_len;
    int localThread = 1;
    sycl::range<1> GlobalRange(globalGroup * localThread);
    sycl::range<1> LocalRange(localThread);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);
    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            int h = ndi.get_group(0);
            simd<float, 128> data = block_load<float, 128>(inputs + h * 128);
            simd<float, 128> buffer = data;
            simd<uint8_t, 128> p = 0;
            int count = 0;
            simd_mask<128> m;

            while (count < 8)
            {
                float max = hmax<float, float, 128>(buffer);
                m = (buffer > max - 0.000001);
                buffer.merge(FP32_MIN, m);
                p.merge(1, m);

                count = sycl::ext::intel::esimd::detail::sum<uint8_t, uint8_t, 128>(p);
            }

            int count2 = 0;
            if (count > 8)
            {
                p.merge(2, m);
                count2 = sycl::ext::intel::esimd::detail::sum<int, uint16_t, 128>(m);
                count2 = count2 + 8 - count;
            }

            float max = hmax<float, float, 128>(data);
            simd<float, 128> idata2 = data - max;
            simd<float, 128> edata = sycl::ext::intel::esimd::pow<float, 128, float>(2.718f, idata2);
            float sum = sycl::ext::intel::esimd::detail::sum<float, float, 128>(edata);
            simd<float, 128> sdata = edata / sum;
            
            simd<float, 8> output_weight;
            simd<uint32_t, 8> output_indexes;

            int idx = 0;
            sum = 0.0;
            int idx2 = 0;
            for (int i = 0; i < 128 && idx < 8; i++)
            {
                if (p[i] == 1)
                {
                    output_indexes[idx] = i;
                    output_weight[idx] = sdata[i];
                    sum += sdata[i];
                    ++idx;
                }
                if (p[i] == 2 && idx2 < count2)
                {
                    output_indexes[idx] = i;
                    output_weight[idx] = sdata[i];
                    sum += sdata[i];
                    ++idx;
                    ++idx2;
                }
            }

            output_weight = output_weight / sum;

            block_store<float, 8>(weights + h * 8, output_weight);
            block_store<uint32_t, 8>(indexes + h * 8, output_indexes);

        });
    });

}

template<typename IT, uint32_t localRange>
static void runUpGateSilu(sycl::queue &q, IT* inputs, uint8_t *up, uint8_t *gate, uint32_t *indexes, fp16* outputs, uint32_t selected_experts, uint32_t input_len, uint32_t output_len)
{
    uint32_t expert_size = input_len * output_len / 128 * 66;
    const uint32_t outputPerGroup = 16;
    int globalGroup = (output_len + outputPerGroup - 1) / outputPerGroup;
    int localThread = localRange;
    sycl::range<2> GlobalRange(globalGroup * localThread, selected_experts);
    sycl::range<2> LocalRange(localThread, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);


    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * outputPerGroup * localRange * 2);

            constexpr uint32_t slmOffsetUpResult = 0;
            constexpr uint32_t slmOffsetGateResult = sizeof(float) * outputPerGroup * localRange;


            constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
            constexpr uint32_t loopStep = 4;

            int h = ndi.get_group(0);
            int v = ndi.get_group(1);
            int hh = ndi.get_local_id(0);
            
            simd<IT, 128> in;
            simd<uint8_t, loopStep * 64> aaa;
            simd<fp16, loopStep * 2> scales;
            simd<fp16, loopStep * 128> wei;
            simd<fp16, outputPerGroup> out_up;
            simd<fp16, outputPerGroup> out_gate;

            uint32_t input_offset = hh * 64;
            in.template select<64, 1>(0) = block_load<IT, 64>(inputs + input_offset);
            input_offset += localRange * 64;
            in.template select<64, 1>(64) = block_load<IT, 64>(inputs + input_offset);

            // ==================================== up ==================================
            {
                uint8_t *current_up = up + indexes[v] * expert_size;
                fp16 *current_up_s = (fp16 *)(current_up + input_len * output_len / 2);
                uint32_t weights_offset = (h * outputPerGroup * input_len + hh * 64)/2;
                simd<uint32_t, 8> weights_offsets(baseOffsetInc8);
                weights_offsets = weights_offsets * 4;
                weights_offsets = weights_offsets + weights_offset;

                simd<uint32_t, 8> scales_offsets;
                scales_offsets[0] = h * outputPerGroup;
                scales_offsets[1] = scales_offsets[0] + input_len * output_len / 2 / 128;
            #pragma unroll
                for (int j = 1; j < loopStep; j++)
                {
                    scales_offsets[2 * j] = scales_offsets[0] + j;
                    scales_offsets[2 * j + 1] = scales_offsets[1] + j;
                }
                scales_offsets = scales_offsets + (hh / 2) * output_len;
                scales_offsets = scales_offsets * 2;
            #pragma unroll
                for (int j = 0; j < outputPerGroup; j += loopStep)
                {
                    scales = __ESIMD_ENS::lsc_gather<
                        fp16,
                        1,
                        __ESIMD_ENS::lsc_data_size::u16,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached,
                        8,
                        uint32_t
                        >(current_up_s, scales_offsets);
                    scales_offsets += (loopStep * 2);
            #pragma unroll
                    for (int k = 0; k < loopStep; k++)
                    {
                        aaa.template bit_cast_view<uint32_t>().template select<8, 1>(k * 16) =
                            __ESIMD_ENS::lsc_gather<
                            uint32_t,
                            1,
                            __ESIMD_ENS::lsc_data_size::u32,
                            __ESIMD_ENS::cache_hint::cached,
                            __ESIMD_ENS::cache_hint::cached,
                            8,
                            uint32_t
                            >((uint32_t*)current_up, weights_offsets);
                        weights_offsets += input_len/4;
                        aaa.template bit_cast_view<uint32_t>().template select<8, 1>(k * 16 + 8) =
                            __ESIMD_ENS::lsc_gather<
                            uint32_t,
                            1,
                            __ESIMD_ENS::lsc_data_size::u32,
                            __ESIMD_ENS::cache_hint::cached,
                            __ESIMD_ENS::cache_hint::cached,
                            8,
                            uint32_t
                            >((uint32_t*)current_up, weights_offsets);
                        weights_offsets += input_len/4;
                    }

                    simd<int8_t, loopStep * 64> lo = aaa.template bit_cast_view<int8_t>() << 4;
                    lo = lo >> 4;
                    simd<int8_t, loopStep * 64> hi = aaa.template bit_cast_view<int8_t>() >> 4;
                    wei.select<loopStep * 64, 2>(0) = lo;
                    wei.select<loopStep * 64, 2>(1) = hi;

            #pragma unroll
                    for (int k2 = 0; k2 < loopStep * 2; k2++)
                    {
                        wei.select<64, 1>(k2 * 64) = wei.select<64, 1>(k2 * 64) * scales[k2];
                    }


                    simd<float, loopStep * 32> temp = 0;
            #pragma unroll
                    for (int k = 0; k < loopStep; k++)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            temp.select<32, 1>(k * 32) += in.template select<32, 1>(32 * i) * wei.select<32, 1>(k * 128 + 32 * i);
                        }
                    }

            #pragma unroll
                    for (int k = 0; k < loopStep; k++)
                    {
                        out_up[j + k] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(temp.select<32, 1>(32 * k));
                    }
                }
            }

            // ==================================gate=======================================
            {
                uint8_t *current_gate = gate + indexes[v] * expert_size;
                fp16 *current_gate_s = (fp16 *)(current_gate + input_len * output_len / 2);
                uint32_t weights_offset = (h * outputPerGroup * input_len + hh * 64)/2;
                simd<uint32_t, 8> weights_offsets(baseOffsetInc8);
                weights_offsets = weights_offsets * 4;
                weights_offsets = weights_offsets + weights_offset;

                simd<uint32_t, 8> scales_offsets;
                scales_offsets[0] = h * outputPerGroup;
                scales_offsets[1] = scales_offsets[0] + input_len * output_len / 2 / 128;
            #pragma unroll
                for (int j = 1; j < loopStep; j++)
                {
                    scales_offsets[2 * j] = scales_offsets[0] + j;
                    scales_offsets[2 * j + 1] = scales_offsets[1] + j;
                }
                scales_offsets = scales_offsets + (hh / 2) * output_len;
                scales_offsets = scales_offsets * 2;
            #pragma unroll
                for (int j = 0; j < outputPerGroup; j += loopStep)
                {
                    scales = __ESIMD_ENS::lsc_gather<
                        fp16,
                        1,
                        __ESIMD_ENS::lsc_data_size::u16,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached,
                        8,
                        uint32_t
                        >(current_gate_s, scales_offsets);
                    scales_offsets += (loopStep * 2);
            #pragma unroll
                    for (int k = 0; k < loopStep; k++)
                    {
                        aaa.template bit_cast_view<uint32_t>().template select<8, 1>(k * 16) =
                            __ESIMD_ENS::lsc_gather<
                            uint32_t,
                            1,
                            __ESIMD_ENS::lsc_data_size::u32,
                            __ESIMD_ENS::cache_hint::cached,
                            __ESIMD_ENS::cache_hint::cached,
                            8,
                            uint32_t
                            >((uint32_t*)current_gate, weights_offsets);
                        weights_offsets += input_len/4;
                        aaa.template bit_cast_view<uint32_t>().template select<8, 1>(k * 16 + 8) =
                            __ESIMD_ENS::lsc_gather<
                            uint32_t,
                            1,
                            __ESIMD_ENS::lsc_data_size::u32,
                            __ESIMD_ENS::cache_hint::cached,
                            __ESIMD_ENS::cache_hint::cached,
                            8,
                            uint32_t
                            >((uint32_t*)current_gate, weights_offsets);
                        weights_offsets += input_len/4;
                    }

                    simd<int8_t, loopStep * 64> lo = aaa.template bit_cast_view<int8_t>() << 4;
                    lo = lo >> 4;
                    simd<int8_t, loopStep * 64> hi = aaa.template bit_cast_view<int8_t>() >> 4;
                    wei.select<loopStep * 64, 2>(0) = lo;
                    wei.select<loopStep * 64, 2>(1) = hi;

            #pragma unroll
                    for (int k2 = 0; k2 < loopStep * 2; k2++)
                    {
                        wei.select<64, 1>(k2 * 64) = wei.select<64, 1>(k2 * 64) * scales[k2];
                    }


                    simd<float, loopStep * 32> temp = 0;
            #pragma unroll
                    for (int k = 0; k < loopStep; k++)
                    {
                        for (int i = 0; i < 4; i++)
                        {
                            temp.select<32, 1>(k * 32) += in.template select<32, 1>(32 * i) * wei.select<32, 1>(k * 128 + 32 * i);
                        }
                    }

            #pragma unroll
                    for (int k = 0; k < loopStep; k++)
                    {
                        out_gate[j + k] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(temp.select<32, 1>(32 * k));
                    }
                }
            }

            slm_block_store<float, outputPerGroup>(slmOffsetUpResult + sizeof(float) * hh * outputPerGroup, out_up);
            slm_block_store<float, outputPerGroup>(slmOffsetGateResult + sizeof(float) * hh * outputPerGroup, out_gate);
            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> acc_up = 0.0;
        #pragma unroll
                for (int j = 0; j < localRange; j++)
                {
                    acc_up += slm_block_load<float, outputPerGroup>(slmOffsetUpResult + sizeof(float) * j * outputPerGroup);
                }
                slm_block_store<float, outputPerGroup>(slmOffsetUpResult, acc_up);
            }
            if (hh == 1)
            {
                simd<float, outputPerGroup> acc_gate = 0.0;
        #pragma unroll
                for (int j = 0; j < localRange; j++)
                {
                    acc_gate += slm_block_load<float, outputPerGroup>(slmOffsetGateResult + sizeof(float) * j * outputPerGroup);
                }
                slm_block_store<float, outputPerGroup>(slmOffsetGateResult, acc_gate);
                
            }

            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> acc_up = slm_block_load<float, outputPerGroup>(slmOffsetUpResult);
                simd<float, outputPerGroup> acc_gate = slm_block_load<float, outputPerGroup>(slmOffsetGateResult);

                simd<float, outputPerGroup> temp = -1.0 * acc_gate;
                temp = pow<float, outputPerGroup, float>(2.718f, temp);
                temp = temp + 1.0;
                temp = 1.0/temp;
                temp = temp * acc_gate;
                temp = temp * acc_up;
                block_store<fp16, outputPerGroup>(outputs + v * output_len + h * outputPerGroup, temp);
            }

        });
    });

}


template<typename OT, uint32_t localRange>
static void runDownMerge(sycl::queue &q, fp16* inputs, uint8_t *down, uint32_t *indexes, float *weights, OT* outputs, uint32_t selected_experts, uint32_t input_len, uint32_t output_len)
{
    uint32_t expert_size = input_len * output_len / 128 * 66;
    const uint32_t outputPerGroup = 8;
    int globalGroup = (output_len + outputPerGroup - 1) / outputPerGroup;
    int localThread = localRange;
    sycl::range<1> GlobalRange(globalGroup * localThread);
    sycl::range<1> LocalRange(localThread);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);


    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            slm_init(sizeof(float) * outputPerGroup * localRange);

            constexpr uint32_t baseOffsetInc8[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            uint32_t input_offset = hh * 64;

            simd<float, outputPerGroup> mergedOutput = 0.0;

            simd<uint32_t, 8> scales_offsets(baseOffsetInc8);
            scales_offsets = scales_offsets + (hh / 2) * output_len;
            scales_offsets = scales_offsets + h * outputPerGroup;
            scales_offsets = scales_offsets * 2;

            uint32_t weights_offset = (h * outputPerGroup * input_len + hh * 64)/2;
            simd<uint32_t, 8> weights_offsets(baseOffsetInc8);
            weights_offsets = weights_offsets * 4;
            weights_offsets = weights_offsets + weights_offset;

            for (int exp = 0; exp < selected_experts; exp ++)
            {
                simd<fp16, 64> in;
                simd<uint8_t, outputPerGroup * 32> aaa;
                simd<fp16, 8> scales;
                simd<fp16, outputPerGroup * 64> wei;
                simd<fp16, outputPerGroup> out;

                simd<uint32_t, 8> cur_weights_offsets = weights_offsets;

                in = block_load<fp16, 64>(inputs + input_offset);
                input_offset += input_len;

                uint8_t *current_down = down + indexes[exp] * expert_size;
                fp16 *current_down_s = (fp16 *)(current_down + input_len * output_len / 2);

                scales = __ESIMD_ENS::lsc_gather<
                    fp16,
                    1,
                    __ESIMD_ENS::lsc_data_size::u16,
                    __ESIMD_ENS::cache_hint::cached,
                    __ESIMD_ENS::cache_hint::cached,
                    8,
                    uint32_t
                    >(current_down_s, scales_offsets);

        #pragma unroll
                for (int k = 0; k < outputPerGroup; k++)
                {
                    aaa.template bit_cast_view<uint32_t>().template select<8, 1>(k * 8) =
                        __ESIMD_ENS::lsc_gather<
                        uint32_t,
                        1,
                        __ESIMD_ENS::lsc_data_size::u32,
                        __ESIMD_ENS::cache_hint::cached,
                        __ESIMD_ENS::cache_hint::cached,
                        8,
                        uint32_t
                        >((uint32_t*)current_down, cur_weights_offsets);
                    cur_weights_offsets += input_len/2;
                }

                simd<int8_t, outputPerGroup * 32> lo = aaa.template bit_cast_view<int8_t>() << 4;
                lo = lo >> 4;
                simd<int8_t, outputPerGroup * 32> hi = aaa.template bit_cast_view<int8_t>() >> 4;
                wei.select<outputPerGroup * 32, 2>(0) = lo;
                wei.select<outputPerGroup * 32, 2>(1) = hi;

        #pragma unroll
                for (int k = 0; k < outputPerGroup; k++)
                {
                    wei.select<64, 1>(k * 64) = wei.select<64, 1>(k * 64) * scales[k];
                }


                simd<float, outputPerGroup * 32> temp = 0;
        #pragma unroll
                for (int k = 0; k < outputPerGroup; k++)
                {
                    for (int i = 0; i < 2; i++)
                    {
                        temp.select<32, 1>(k * 32) += in.template select<32, 1>(32 * i) * wei.select<32, 1>(k * 64 + 32 * i);
                    }
                }

        #pragma unroll
                for (int k = 0; k < outputPerGroup; k++)
                {
                    out[k] = sycl::ext::intel::esimd::detail::sum<float, float, 32>(temp.select<32, 1>(32 * k));
                }

                float w = weights[exp];
                mergedOutput += out * w;
            }

            slm_block_store<float, outputPerGroup>(sizeof(float) * hh * outputPerGroup, mergedOutput);
            barrier();

            if (hh == 0)
            {
                simd<float, outputPerGroup> acc = 0.0;
        #pragma unroll
                for (int j = 0; j < localRange; j++)
                {
                    acc += slm_block_load<float, outputPerGroup>(sizeof(float) * j * outputPerGroup);
                }
                block_store<OT, outputPerGroup>(outputs + h * outputPerGroup, acc);
            }
        });
    });

}


void dump(sycl::queue *q, void *data, uint32_t size, const char* filename)
{
    q->wait();
    uint8_t *temp = new uint8_t[size];
    q->memcpy(temp, data, size).wait();
    FILE *fp = nullptr;
    fopen_s(&fp, filename, "wb");
    fwrite(temp, 1, size, fp);
    fclose(fp);
    delete[] temp;
}

bool runFfnMoeFusionVec_L2(sycl::queue* q, uint8_t* inputs, uint8_t* router, uint8_t* up, uint8_t* gate, uint8_t* down, uint8_t *outputs, uint32_t total_experts, uint32_t selected_experts, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, uint8_t* shuffleTt)
{
    if (token_len != 1)
    {
        printf("Error! runFfnMoeFusionVec only supports GEMV\n");
        return false;
    }

    //=====================================router===========================================
    float *router_output = (float *)shuffleTt; // 128
    if (input_precision == 1)
    {
        runRouter<fp16, 16>(*q, (fp16 *)inputs, (float *)router, router_output, input_len, total_experts);
    }
    else
    {
        runRouter<float, 16>(*q, (float *)inputs, (float *)router, router_output, input_len, total_experts);
    }

    //================================softmax, topk, norm===================================
    uint32_t *indexes = (uint32_t *)(router_output + token_len * total_experts); //
    float *weights = (float *)(indexes + token_len * selected_experts);
    topKAndWeights(*q, router_output, indexes, weights, token_len, total_experts, selected_experts);

    //==================================ffn up, gate, silu==================================
    fp16* silu_result = (fp16 *)(weights + token_len * selected_experts);
    if (input_precision == 1)
    {
        runUpGateSilu<fp16, 16>(*q, (fp16 *)inputs, up, gate, indexes, silu_result, selected_experts, input_len, hidden_len);
    }
    else
    {
        runUpGateSilu<float, 16>(*q, (float *)inputs, up, gate, indexes, silu_result, selected_experts, input_len, hidden_len);
    }

    if (output_precision == 1)
    {
        runDownMerge<fp16, 12>(*q, silu_result, down, indexes, weights, (fp16*)outputs, selected_experts, hidden_len, input_len);
    }
    else
    {
        runDownMerge<float, 12>(*q, silu_result, down, indexes, weights, (float*)outputs, selected_experts, hidden_len, input_len);
    }

    //dump(q, outputs, input_len * sizeof(float), "outputs.float.bin");
    
    
    // router

}