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

#include <chrono>

#include <map>

#include <windows.h>

using namespace sycl;
using fp16 = sycl::half;
#define DEVICE_MEM_ALIGNMENT 4096;


using namespace std;
using namespace sycl::ext::intel::esimd;

#define GROUP_SIZE 128

extern "C" bool __declspec(dllexport) runFfnMoeFusion_dnnl(sycl::queue* q, uint8_t* inputs, uint8_t* router, uint8_t* up, uint8_t* gate, uint8_t* down, uint8_t *outputs, uint32_t total_experts, uint32_t selected_experts, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, uint8_t* shuffleTt);
extern "C" size_t __declspec(dllexport) getScratchBufferSize_ffnmoe(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden);

class MatMulPremitiveMgr_FFN
{
public:
    static MatMulPremitiveMgr_FFN& Instance()
    {
        static MatMulPremitiveMgr_FFN ins;
        return ins;
    }

    MatMulPremitiveMgr_FFN(MatMulPremitiveMgr_FFN const&) = delete;
    void operator=(MatMulPremitiveMgr_FFN const&) = delete;

    dnnl::engine& Engine(sycl::queue* q)
    {
        if (m_recorded_queue && *q == *m_recorded_queue)
        {
            //printf("HFDebug: reuse previous engine\n");
            return m_engine;
        }
        
        //printf("HFDebug: create a new engine\n");
        m_engine = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
        m_recorded_queue = q;

        m_prims_up.clear();
        m_prims_down.clear();

        return m_engine;
    }

    void Initialize(uint32_t max_token_len, uint32_t input_len, uint32_t hidden_len, dnnl::engine &eng)
    {
        if (m_initialized && m_recorded_input_len == input_len && m_recorded_hidden_len == hidden_len && m_recorded_max_token_len >= max_token_len)
        {
            return;
        }

        m_prims_up.clear();
        m_prims_down.clear();

        for (int token_len = 1; token_len <= max_token_len; token_len ++)
        {
            dnnl::memory::desc input_md({ token_len, input_len }, dnnl::memory::data_type::f16, { input_len, 1 });
            dnnl::memory::desc up_result_md({ token_len, hidden_len }, dnnl::memory::data_type::f16, { hidden_len, 1 });
            dnnl::memory::desc up_weights_md ({ input_len, hidden_len }, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
            dnnl::memory::desc up_scale_md ({ input_len / 128, hidden_len }, dnnl::memory::data_type::f16, { hidden_len, 1 });
            dnnl::memory::desc down_weights_md ({ hidden_len, input_len }, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
            dnnl::memory::desc down_scale_md ({ hidden_len / 128, input_len }, dnnl::memory::data_type::f16, { input_len, 1 });

            dnnl::primitive_attr attr;
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), { 128, 1 }, dnnl::memory::data_type::f16);
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

            auto matmul_up_pd = dnnl::matmul::primitive_desc(eng, input_md, up_weights_md, up_result_md, attr);
            auto matmul_down_pd = dnnl::matmul::primitive_desc(eng, up_result_md, down_weights_md, input_md, attr);

            m_prims_up[token_len] = dnnl::matmul(matmul_up_pd);
            m_prims_down[token_len] = dnnl::matmul(matmul_down_pd);
        }

        m_initialized = true;
        m_recorded_input_len = input_len;
        m_recorded_hidden_len = hidden_len;
        m_recorded_max_token_len = max_token_len;
    }

    dnnl::matmul& GetUp(uint32_t token_len)
    {
        return m_prims_up[token_len];
    }

    dnnl::matmul& GetDown(uint32_t token_len)
    {
        return m_prims_down[token_len];
    }

private:
    MatMulPremitiveMgr_FFN() {}

    bool m_initialized = false;
    std::map<uint64_t, dnnl::matmul> m_prims_up;
    std::map<uint64_t, dnnl::matmul> m_prims_down;
    sycl::queue* m_recorded_queue = nullptr;
    dnnl::engine m_engine;
    uint32_t m_recorded_max_token_len = 0;
    uint32_t m_recorded_input_len = 0;
    uint32_t m_recorded_hidden_len = 0;

};

size_t getScratchBufferSize_ffnmoe(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    // hard coded for Qwen3 A3B
    uint32_t total_experts = 128;
    uint32_t selected_experts = 8;
    uint32_t input_len = 2048;
    uint32_t hidden_len = 768;
    uint32_t token_len = maxBatch;
    return      token_len * input_len * sizeof(fp16) // input fp16
            +   total_experts * input_len * sizeof(fp16) // router weights fp16
            +   token_len * total_experts * sizeof(float) // router outputs
            +   token_len * selected_experts * sizeof(uint32_t) // indexes
            +   token_len * selected_experts * sizeof(float) // weights
            +   token_len * selected_experts * sizeof(uint32_t) // scattered to offsets
            +   token_len * selected_experts * input_len * sizeof(fp16)  // scattered inputs
            +   token_len * selected_experts * input_len * sizeof(fp16)  // scattered outputs
            +   token_len * hidden_len * sizeof(fp16)  // up result
            +   token_len * hidden_len * sizeof(fp16)  // gate result
            +   token_len * hidden_len * sizeof(fp16)  // silu result
            ;
}

static bool runSiluAndMultiply2(sycl::queue& q, sycl::half* gate_result, sycl::half* up_result, sycl::half* outputs, unsigned batch_num, unsigned input_len)
{
    int groups = (input_len + 127)/128;
    //int locals = (input_len + 127) / 128;

    sycl::range<1> GlobalRange(groups);
    sycl::range<1> LocalRange(1);
    sycl::nd_range<1> RangeCommon(GlobalRange, LocalRange);

    sycl::event e;
    try {
        e = q.submit([&](handler& cgh) {
            cgh.parallel_for(RangeCommon, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
                //int hh = ndi.get_local_id(0);
                int h = ndi.get_group(0);
                //int range = ndi.get_local_range(0);

                simd<sycl::half, 128> gate;
                simd<sycl::half, 128> up;
                //int offset = hh * 128 + h * input_len;
                int offset = h * 128;

                for (int lp = 0; lp < batch_num; lp ++)
                {
                    gate = block_load<fp16, 128>(gate_result + offset);
                    up = block_load<fp16, 128>(up_result + offset);
                    //simd<float, 128> temp = up;
                    simd<float, 128> temp = -1.0 * gate;
                    temp = pow<float, 128, float>(2.718f, temp);
                    temp = temp + 1.0;
                    temp = 1.0/temp;
                    temp = temp * gate;
                    temp = temp * up;
                    block_store<fp16, 128>(outputs + offset, temp);

                    offset += input_len;
                }

                });
            });
    }
    catch (sycl::exception const& e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }
}

void dump(sycl::queue*q, void *data, uint32_t size, const char* filename)
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

static void scatterInputs(sycl::queue& q, fp16 *inputs, uint32_t* scattered_offsets, fp16 *outputs, uint32_t token_len, uint32_t input_len, uint32_t selected_experts)
{
    // if (selected_experts != 8)
    // {
    //     printf("Warning! only supports 8 selected experts by now! %d is provided\n", selected_experts);
    // }
    int globalGroup = token_len;
    int localThread = (input_len + 127) / 128;
    sycl::range<1> GlobalRange(globalGroup * localThread);
    sycl::range<1> LocalRange(localThread);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<uint32_t, 8> indexes = block_load<uint32_t, 8>(scattered_offsets + h * 8);
            uint32_t input_offset = h * input_len + hh * 128;
            simd<fp16, 128> data = block_load<fp16, 128>(inputs + input_offset);
        
        #pragma unroll
            for (int j = 0; j < 8; j ++)
            {
                uint32_t output_offset = indexes[j] * input_len + hh * 128;
                block_store<fp16, 128>(outputs + output_offset, data);
            }
        });
    });

}

template <typename OT>
static void mergeOutputs(sycl::queue& q, fp16 *inputs, uint32_t* scattered_offsets, float* weights, OT *outputs, uint32_t token_len, uint32_t input_len, uint32_t selected_experts)
{
    // if (selected_experts != 8)
    // {
    //     printf("Warning! only supports 8 selected experts by now! %d is provided\n", selected_experts);
    // }
    int globalGroup = token_len;
    int localThread = (input_len + 127) / 128;
    sycl::range<1> GlobalRange(globalGroup * localThread);
    sycl::range<1> LocalRange(localThread);
    sycl::nd_range<1> Range(GlobalRange, LocalRange);

    q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(Range, [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            int h = ndi.get_group(0);
            int hh = ndi.get_local_id(0);

            simd<uint32_t, 8> indexes = block_load<uint32_t, 8>(scattered_offsets + h * 8);
            simd<float, 8> w = block_load<float, 8>(weights + h * 8);
            uint32_t output_offset = h * input_len + hh * 128;
            simd<float, 128> outputdata = 0.0;
        
        #pragma unroll
            for (int j = 0; j < 8; j ++)
            {
                uint32_t input_offset = indexes[j] * input_len + hh * 128;
                simd<fp16, 128> data = block_load<fp16, 128>(inputs + input_offset);
                float wei = w[j];
                outputdata = outputdata + data * wei;
            }
            block_store<OT, 128>(outputs + output_offset, outputdata);
        });
    });

}


static void ffn(sycl::queue&q, dnnl::engine &eng, dnnl::stream &s, fp16 *inputs, uint8_t *up, uint8_t *gate, uint8_t *down, fp16* up_result, fp16* gate_result, fp16* silu_result, fp16* outputs, uint32_t token_len, uint32_t input_len, uint32_t hidden_len)
{
    dnnl::memory::desc input_md({ token_len, input_len }, dnnl::memory::data_type::f16, { input_len, 1 });
    dnnl::memory::desc up_result_md({ token_len, hidden_len }, dnnl::memory::data_type::f16, { hidden_len, 1 });
    dnnl::memory::desc up_weights_md ({ input_len, hidden_len }, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
    dnnl::memory::desc up_scale_md ({ input_len / 128, hidden_len }, dnnl::memory::data_type::f16, { hidden_len, 1 });
    dnnl::memory::desc down_weights_md ({ hidden_len, input_len }, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
    dnnl::memory::desc down_scale_md ({ hidden_len / 128, input_len }, dnnl::memory::data_type::f16, { input_len, 1 });

    // dnnl::primitive_attr attr;
    // attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), { 128, 1 }, dnnl::memory::data_type::f16);
    // attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

    // auto matmul_up_pd = dnnl::matmul::primitive_desc(eng, input_md, up_weights_md, up_result_md, attr);
    // auto matmul_down_pd = dnnl::matmul::primitive_desc(eng, up_result_md, down_weights_md, input_md, attr);

    // auto matmul_up = dnnl::matmul(matmul_up_pd);
    // auto matmul_down = dnnl::matmul(matmul_down_pd);
    auto matmul_up = MatMulPremitiveMgr_FFN::Instance().GetUp(token_len);
    auto matmul_down = MatMulPremitiveMgr_FFN::Instance().GetDown(token_len);

    dnnl::memory input_mem = dnnl::memory(input_md, eng, inputs);

    dnnl::memory up_weights_mem = dnnl::memory(up_weights_md, eng, up);
    dnnl::memory up_scales_mem = dnnl::memory(up_scale_md, eng, up + input_len * hidden_len / 2);
    dnnl::memory up_result_mem = dnnl::memory(up_result_md, eng, up_result);

    dnnl::memory gate_weights_mem = dnnl::memory(up_weights_md, eng, gate);
    dnnl::memory gate_scales_mem = dnnl::memory(up_scale_md, eng, gate + input_len * hidden_len / 2);
    dnnl::memory gate_result_mem = dnnl::memory(up_result_md, eng, gate_result);

    dnnl::memory silu_result_mem = dnnl::memory(up_result_md, eng, silu_result);

    dnnl::memory down_weights_mem = dnnl::memory(down_weights_md, eng, down);
    dnnl::memory down_scales_mem = dnnl::memory(down_scale_md, eng, down + input_len * hidden_len / 2);
    dnnl::memory down_result_mem = dnnl::memory(input_md, eng, outputs);

    matmul_up.execute(s, { {DNNL_ARG_SRC, input_mem}, {DNNL_ARG_WEIGHTS, up_weights_mem}, {DNNL_ARG_DST, up_result_mem}, {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, up_scales_mem} });
    matmul_up.execute(s, { {DNNL_ARG_SRC, input_mem}, {DNNL_ARG_WEIGHTS, gate_weights_mem}, {DNNL_ARG_DST, gate_result_mem}, {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, gate_scales_mem} });
    runSiluAndMultiply2(q, gate_result, up_result, silu_result, token_len, hidden_len);
    matmul_down.execute(s, { {DNNL_ARG_SRC, silu_result_mem}, {DNNL_ARG_WEIGHTS, down_weights_mem}, {DNNL_ARG_DST, down_result_mem}, {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, down_scales_mem} });
}


bool runFfnMoeFusion_dnnl(sycl::queue* q, uint8_t* inputs, uint8_t* router, uint8_t* up, uint8_t* gate, uint8_t* down, uint8_t *outputs, uint32_t total_experts, uint32_t selected_experts, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, unsigned input_precision, unsigned output_precision, uint8_t* shuffleTt)
{
    //dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());;
    dnnl::engine eng = MatMulPremitiveMgr_FFN::Instance().Engine(q);
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, *q);

    MatMulPremitiveMgr_FFN::Instance().Initialize(2048, input_len, hidden_len, eng);

    // q->wait();
    // auto t_start = std::chrono::steady_clock::now();

    fp16* input_fp16 = (fp16 *)shuffleTt;
    if (input_precision == 1)
    {
        input_fp16 = (fp16 *)inputs;
    }
    fp16* router_fp16 = (fp16 *)shuffleTt + token_len * input_len;
    float* router_outputs = (float*)(router_fp16 + input_len * total_experts);

    // ===============================router======================================
    dnnl::memory::desc input_f32_md({ token_len, input_len }, dnnl::memory::data_type::f32, { input_len, 1 }); // M x K layout
    dnnl::memory::desc input_f16_md({ token_len, input_len }, dnnl::memory::data_type::f16, { input_len, 1 }); // M x K layout
    dnnl::memory::desc router_f32_md({ input_len, total_experts }, dnnl::memory::data_type::f32, { 1, input_len}); // M x K layout
    dnnl::memory::desc router_f16_md({ input_len, total_experts }, dnnl::memory::data_type::f16, { 1, input_len}); // M x K layout
    dnnl::memory::desc router_output_md({ token_len, total_experts }, dnnl::memory::data_type::f32, { total_experts, 1}); // M x K layout

    dnnl::memory input_fp16_mem = dnnl::memory(input_f16_md, eng, input_fp16);
    if (input_precision == 0)
    {
        dnnl::memory input_fp32_mem = dnnl::memory(input_f32_md, eng, inputs);
        dnnl::reorder(input_fp32_mem, input_fp16_mem).execute(s, input_fp32_mem, input_fp16_mem);
    }

    dnnl::memory router_fp32_mem = dnnl::memory(router_f32_md, eng, router);
    dnnl::memory router_fp16_mem = dnnl::memory(router_f16_md, eng, router_fp16);
    dnnl::reorder(router_fp32_mem, router_fp16_mem).execute(s, router_fp32_mem, router_fp16_mem);

    dnnl::memory router_output_mem = dnnl::memory(router_output_md, eng, router_outputs);


    dnnl::primitive_attr attr;
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    dnnl::matmul::primitive_desc router_matmul_pd = dnnl::matmul::primitive_desc(eng, input_f16_md, router_f16_md, router_output_md, attr);
    dnnl::matmul router_matmul_p(router_matmul_pd);

    router_matmul_p.execute(s, { {DNNL_ARG_SRC, input_fp16_mem}, {DNNL_ARG_WEIGHTS, router_fp16_mem}, {DNNL_ARG_DST, router_output_mem} });

    // ===========================softmax, top-k, norm=============================
    uint32_t *indexes = (uint32_t *)(router_outputs + token_len * total_experts);
    float * weights = (float *)(indexes + token_len * selected_experts);

    topKAndWeights(*q, router_outputs, indexes, weights, token_len, total_experts, selected_experts);

    // ==========================index-map=========================================
    uint32_t *indexes_data = new uint32_t[token_len * selected_experts];
    q->memcpy(indexes_data, indexes, token_len * selected_experts * sizeof(uint32_t)).wait();
    int *counts = new int[total_experts];
    memset(counts, 0, total_experts * sizeof(int));
    int *starts = new int[total_experts];
    memset(starts, 0, total_experts * sizeof(int));
    uint32_t *to_offsets_cpu = new uint32_t[token_len * selected_experts];

    uint32_t global_offset = 0;

    for (uint32_t t = 0; t < total_experts; t++)
    {
        starts[t] = global_offset;
        for (uint32_t j = 0; j < token_len; j++)
        {
            for (uint32_t k = 0; k < selected_experts; k++)
            {
                if (indexes_data[j * selected_experts + k] == t)
                {
                    to_offsets_cpu[j * selected_experts + k] = global_offset;
                    global_offset ++;
                }
            }
        }
    }
    delete[] indexes_data;

    // ==========================scatter inputs=======================================
    uint32_t *scattered_offsets = (uint32_t *)(weights + token_len * selected_experts);
    fp16* scattered_inputs = (fp16 *)(scattered_offsets + token_len * selected_experts);
    q->memcpy(scattered_offsets, to_offsets_cpu, token_len * selected_experts * sizeof(uint32_t)).wait();
    scatterInputs(*q, input_fp16, scattered_offsets, scattered_inputs, token_len, input_len, selected_experts);


    for (uint32_t t = 0; t < total_experts - 1; t++)
    {
        counts[t] = starts[t + 1] - starts[t];
    }
    counts[total_experts - 1] = token_len * selected_experts - starts[total_experts - 1];

    // ========================== FFNs ===============================================
    fp16* scattered_outputs = scattered_inputs + token_len * selected_experts * input_len;
    fp16* up_result = scattered_outputs + token_len * selected_experts * input_len;
    fp16* gate_result = up_result + token_len * hidden_len;
    fp16* silu_result = gate_result + token_len * hidden_len;

    uint32_t weights_size_per_expert = input_len * hidden_len / 128 * 66;

    for (int t = 0; t < total_experts; t++)
    {
        //printf("%d: %d\n", t, counts[t]);
        if (counts[t] == 0)
        {
            continue;
        }
        fp16 *ip = scattered_inputs + starts[t] * input_len;
        fp16 *op = scattered_outputs + starts[t] * input_len;
        uint8_t* iup = up + t * weights_size_per_expert;
        uint8_t* igate = gate + t * weights_size_per_expert;
        uint8_t* idown = down + t * weights_size_per_expert;

        ffn(*q, eng, s, ip, iup, igate, idown, up_result, gate_result, silu_result, op, counts[t], input_len, hidden_len);
    }

    // ============================Merge==============================================
    if (output_precision == 1)
    {
        mergeOutputs<fp16>(*q, scattered_outputs, scattered_offsets, weights, (fp16 *)outputs, token_len,  input_len, selected_experts);
    }
    else
    {
        mergeOutputs<float>(*q, scattered_outputs, scattered_offsets, weights, (float *)outputs, token_len,  input_len, selected_experts);
    }

    // char filename[2048];
    // sprintf_s(filename, "outputs.%dx%d.fp32.bin", token_len, input_len);
    // dump(q, outputs, token_len * input_len * sizeof(float), filename);

    // q->wait();
    // auto t_end = std::chrono::steady_clock::now();
    // double total_time = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // printf("HFDebug: FFN moe takes %f ms\n", total_time);

    return true;

}

