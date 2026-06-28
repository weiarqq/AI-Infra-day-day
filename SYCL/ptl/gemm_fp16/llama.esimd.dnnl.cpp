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


using namespace std;
using namespace sycl::ext::intel::esimd;

#define GROUP_SIZE 128

extern "C" size_t __declspec(dllexport) getScratchBufferSize_gemm(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden);
extern "C" bool __declspec(dllexport) runGemm_Q40Weights_L2(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt);
extern "C" void __declspec(dllexport) shuffle_Q40Weights_group128_L2(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len);

extern "C" bool __declspec(dllexport) runFfnFusion_dnnl(sycl::queue* q, uint8_t * up, uint8_t * gate, uint8_t * down, uint8_t * input, uint8_t * output, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, uint8_t* shuffleTt);
extern "C" size_t __declspec(dllexport) getScratchBufferSize_ffn(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden);

extern "C" bool __declspec(dllexport) runGemm_Fp16Weights_WithBias_dnnl(sycl::queue *q, uint8_t *inputs, uint8_t *weights, uint8_t *bias, uint8_t *outputs, unsigned batch, unsigned token_len, unsigned input_len, unsigned output_len, int op, unsigned input_precision, unsigned output_precision, unsigned bias_precision, uint8_t *scratch_buffer);
extern "C" size_t __declspec(dllexport) getVisionScratchBufferSize_gemm(uint32_t maxBatch, uint32_t head_num, uint32_t maxHidden);

class MatMulPremitiveMgr
{
public:
    static MatMulPremitiveMgr& Instance()
    {
        static MatMulPremitiveMgr ins;
        return ins;
    }

    MatMulPremitiveMgr(MatMulPremitiveMgr const&) = delete;
    void operator=(MatMulPremitiveMgr const&) = delete;

    dnnl::engine& Engine(sycl::queue* q)
    {
        if (q == m_recorded_queue)
        {
            //printf("HFDebug: reuse previous engine\n");
            return m_engine;
        }
        
        //printf("HFDebug: create a new engine\n");
        m_engine = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
        m_recorded_queue = q;

        m_prims.clear();

        return m_engine;
    }

    dnnl::matmul& Get(uint64_t M, uint64_t K, uint64_t N, unsigned output_precision, dnnl::engine &eng)
    {
        uint64_t h = _hash(M, K, N, output_precision);
        auto ite = m_prims.find(h);
        if (ite != m_prims.end())
        {
            //printf("HFDebug: find an existing one. M %lld K %lld N %lld\n", M, K, N);
            return ite->second;
        }

        for (int token_len = 1; token_len <= 2048; token_len ++)
        {
            uint64_t hh = _hash(token_len, K, N, output_precision);
            int64_t G = K / GROUP_SIZE;

            dnnl::memory::data_type dt = (output_precision == 1) ? dnnl::memory::data_type::f16 : dnnl::memory::data_type::f32;

            dnnl::memory::desc src_f16_desc = dnnl::memory::desc({ token_len, K }, dnnl::memory::data_type::f16, { K, 1 });
            dnnl::memory::desc weights_desc = dnnl::memory::desc({ K, N }, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
            //dnnl::memory::desc dst_f16_desc = dnnl::memory::desc({ M, N }, dnnl::memory::data_type::f16, { N, 1 });
            dnnl::memory::desc dst_desc = dnnl::memory::desc({ token_len, N }, dt, { N, 1 });
            dnnl::memory::desc scale_f16_desc = dnnl::memory::desc({ G, N }, dnnl::memory::data_type::f16, { N, 1 });

            dnnl::primitive_attr attr;
            attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), { GROUP_SIZE, 1 }, dnnl::memory::data_type::f16);
            attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);

            auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_f16_desc, weights_desc, dst_desc, attr);
            // Create the primitive.
            // TODO: this step may take long time
            m_prims[hh] = dnnl::matmul(matmul_pd);

        }

        return m_prims[h];
    }

private:
    MatMulPremitiveMgr() {}

    uint64_t _hash(uint64_t M, uint64_t K, uint64_t N, unsigned p)
    {
        return ((M * 31 + K) * 31 + N) * 31 + p;
    }

    std::map<uint64_t, dnnl::matmul> m_prims;
    sycl::queue* m_recorded_queue;
    dnnl::engine m_engine;

};

void shuffle_Q40Weights_group128_L2(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        sycl::half d;           // delta
        uint8_t qs[128 / 2]; // nibbles / quants
    } block_q4_0;
    block_q4_0* t = (block_q4_0*)input;
    char* p = (char *)output;
    sycl::half* h = (sycl::half*)(p + input_len * output_len / 2);

    for (int i = 0; i < input_len * output_len / 128; i++)
    {
        //memcpy(p, t[i].qs, QK4_0 / 2);
        for (int j = 0; j < 128 / 2; j += 16)
        {
            int8_t shuffle[32];
            for (int k = 0; k < 16; k++)
            {
                uint8_t hi = t[i].qs[j + k] >> 4;
                uint8_t lo = t[i].qs[j + k] & 0x0f;
                shuffle[k] = lo - 8;
                shuffle[k + 16] = hi - 8;
            }

            for (int k = 0; k < 16; k++)
            {
                p[j + k] = ((shuffle[2 * k + 1] & 0x0f) << 4) | (shuffle[2 * k] & 0x0f);
            }
        }

        p += (128 / 2);

        int vv = i / (input_len / 128);
        int hh = i % (input_len / 128);
        h[hh * output_len + vv] = t[i].d;
    }

}

size_t getScratchBufferSize_gemm(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    return maxBatch * maxHidden * sizeof(fp16);
}

bool runGemm_Q40Weights_L2(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt)
{
    const int64_t M = batch;
    const int64_t N = output_len;
    const int64_t K = input_len;

    int64_t G = K / GROUP_SIZE;

    //dnnl::engine eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    dnnl::engine eng = MatMulPremitiveMgr::Instance().Engine(q);
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, *q);

    dnnl::memory::data_type dt = (output_precision == 1) ? dnnl::memory::data_type::f16 : dnnl::memory::data_type::f32;
    dnnl::memory::desc src_f32_desc = dnnl::memory::desc({ M, K }, dnnl::memory::data_type::f32, { K, 1 });
    dnnl::memory::desc src_f16_desc = dnnl::memory::desc({ M, K }, dnnl::memory::data_type::f16, { K, 1 });
    dnnl::memory::desc weights_desc = dnnl::memory::desc({ K, N }, dnnl::memory::data_type::s4, dnnl::memory::format_tag::ba);
    dnnl::memory::desc dst_desc = dnnl::memory::desc({ M, N }, dt, { N, 1 });
    dnnl::memory::desc scale_f16_desc = dnnl::memory::desc({ G, N }, dnnl::memory::data_type::f16, { N, 1 });

    //dnnl::primitive_attr attr;
    //attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), { GROUP_SIZE, 1 }, dnnl::memory::data_type::f16);
    //attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), { GROUP_SIZE, 1 }, dnnl::memory::data_type::u8);
    //attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    //// Create primitive descriptor.
    //auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_f16_desc, weights_desc, dst_f32_desc, attr);
    //// Create the primitive.
    //// TODO: this step may take long time
    //auto matmul_prim = dnnl::matmul(matmul_pd);

    dnnl::matmul matmul_prim = MatMulPremitiveMgr::Instance().Get(M, K, N, output_precision, eng);

    dnnl::memory src_fp16_mem;
    dnnl::memory weights_mem = dnnl::memory(weights_desc, eng, weights);
    dnnl::memory dst_mem = dnnl::memory(dst_desc, eng, outputs);
    dnnl::memory scale_fp16_mem = dnnl::memory(scale_f16_desc, eng, scales);

    //s.wait();
    //auto start = std::chrono::steady_clock::now();

    if (input_precision == 0) //fp32
    {
        dnnl::memory src_fp32_mem = dnnl::memory(src_f32_desc, eng, inputs);
        src_fp16_mem = dnnl::memory(src_f16_desc, eng, shuffleTt);
        dnnl::reorder(src_fp32_mem, src_fp16_mem).execute(s, src_fp32_mem, src_fp16_mem);
    }
    else
    {
        src_fp16_mem = dnnl::memory(src_f16_desc, eng, inputs);
    }
    // create GEMM primitative and excute
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, src_fp16_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, scale_fp16_mem},
    };

    matmul_prim.execute(s, args);
    //s.wait();

    //auto end = std::chrono::steady_clock::now();
    //double dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    //printf("GEMM of (%d x %d) x (%d x %d) takes %f us\n", M, K, K, N, dur);


    return true;
}

size_t getScratchBufferSize_ffn(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    return maxBatch * maxHidden * sizeof(sycl::half) * 3;
}

bool runSiluAndMultiply(sycl::queue& q, sycl::half* gate_result, sycl::half* up_result, sycl::half* outputs, unsigned batch_num, unsigned input_len)
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

// void dump(sycl::queue*q, uint8_t *data, uint32_t size, const char* filename)
// {
//     q->wait();
//     uint8_t *temp = new uint8_t[size];
//     q->memcpy(temp, data, size).wait();
//     FILE *fp = nullptr;
//     fopen_s(&fp, filename, "wb");
//     fwrite(temp, 1, size, fp);
//     fclose(fp);
//     delete[] temp;
// }

bool runFfnFusion_dnnl(sycl::queue* q, uint8_t * up, uint8_t * gate, uint8_t * down, uint8_t * input, uint8_t * output, uint32_t token_len, uint32_t input_len, uint32_t hidden_len, uint8_t* shuffleTt)
{
    uint8_t *up_output = shuffleTt + token_len * hidden_len * sizeof(sycl::half);
    uint8_t *gate_output = shuffleTt + token_len * hidden_len * sizeof(sycl::half) + token_len * hidden_len * sizeof(sycl::half);

    dnnl::engine eng = MatMulPremitiveMgr::Instance().Engine(q);
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, *q);

    runGemm_Q40Weights_L2(q, input, up, up + input_len*hidden_len/2, up_output, token_len, input_len, hidden_len, 0, 1, shuffleTt);
    runGemm_Q40Weights_L2(q, shuffleTt, gate, gate + input_len*hidden_len/2, gate_output, token_len, input_len, hidden_len, 1, 1, nullptr);
    runSiluAndMultiply(*q, (sycl::half *)gate_output, (sycl::half *)up_output, (sycl::half *)gate_output,  token_len, hidden_len );
    runGemm_Q40Weights_L2(q, gate_output, down, down + input_len*hidden_len/2, output, token_len, hidden_len, input_len, 1, 0, nullptr);


    return true;

}

size_t getVisionScratchBufferSize_gemm(uint32_t maxBatch, uint32_t head_num, uint32_t maxHidden)
{
    //size_t temp1 = head_num * maxBatch * maxBatch;
    size_t temp1 = 0;
    size_t temp2 = maxBatch * maxHidden;
    size_t count = temp1 > temp2? temp1: temp2;
    return count * sizeof(sycl::half);
}


bool runGemm_Fp16Weights_WithBias_dnnl(sycl::queue *q, uint8_t *inputs, uint8_t *weights, uint8_t *bias, uint8_t *outputs, unsigned batch, unsigned token_len, unsigned input_len, unsigned output_len, int op, unsigned input_precision, unsigned output_precision, unsigned bias_precision, uint8_t *scratch_buffer)
{
    auto dt = (output_precision == 1) ? dnnl::memory::data_type::f16 : dnnl::memory::data_type::f32;
    auto bt = (bias_precision == 1) ? dnnl::memory::data_type::f16 : dnnl::memory::data_type::f32;
    dnnl::engine eng = MatMulPremitiveMgr::Instance().Engine(q);
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, *q);

    // Only support FP16 inputs and weights as requested
    dnnl::memory::desc input_f32_md({ batch, token_len, input_len }, dnnl::memory::data_type::f32, { token_len*input_len, input_len, 1 }); // M x K layout
    dnnl::memory::desc input_f16_md({ batch, token_len, input_len }, dnnl::memory::data_type::f16, { token_len*input_len, input_len, 1 }); // M x K layout
    dnnl::memory::desc weights_md({batch, input_len, output_len}, dnnl::memory::data_type::f16, {input_len * output_len, 1, input_len}); // K x N layout
    dnnl::memory::desc output_md({batch, token_len, output_len }, dt, {token_len*output_len, output_len, 1 }); // M x N layout
    dnnl::memory::desc bias_md({1, 1, output_len}, bt, {output_len, output_len, 1 }); // 1D bias

    dnnl::memory src_fp16_mem;
    if (input_precision == 0) //fp32
    {
        dnnl::memory src_fp32_mem = dnnl::memory(input_f32_md, eng, inputs);
        src_fp16_mem = dnnl::memory(input_f16_md, eng, scratch_buffer);
        dnnl::reorder(src_fp32_mem, src_fp16_mem).execute(s, src_fp32_mem, src_fp16_mem);
    }
    else
    {
        src_fp16_mem = dnnl::memory(input_f16_md, eng, inputs);
    }
    dnnl::memory weights_mem = dnnl::memory(weights_md, eng, weights);
    dnnl::memory bias_mem = dnnl::memory(bias_md, eng, bias);
    dnnl::memory dst_mem = dnnl::memory(output_md, eng, outputs);

    dnnl::post_ops postops;
    if (op == 1)
    {
        postops.append_eltwise(dnnl::algorithm::eltwise_gelu_tanh, 1.0, 0.0);
    }


    dnnl::primitive_attr attr;
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    if (op > 0)
    {
        attr.set_post_ops(postops);
    }

    // Create matmul primitive descriptor with bias support
    dnnl::matmul::primitive_desc matmul_pd;
    if (bias == nullptr)
    {
        matmul_pd = dnnl::matmul::primitive_desc(eng, input_f16_md, weights_md, output_md, attr);
    }
    else
    {
        matmul_pd = dnnl::matmul::primitive_desc(eng, input_f16_md, weights_md, bias_md, output_md, attr);
    }
    dnnl::matmul matmul_p(matmul_pd);

    // Execute matmul with bias using DNNL_ARG_BIAS
    if (bias == nullptr)
    {
        matmul_p.execute(s, { 
            {DNNL_ARG_SRC, src_fp16_mem}, 
            {DNNL_ARG_WEIGHTS, weights_mem}, 
            {DNNL_ARG_DST, dst_mem}
        });
    }
    else
    {
        matmul_p.execute(s, { 
            {DNNL_ARG_SRC, src_fp16_mem}, 
            {DNNL_ARG_WEIGHTS, weights_mem}, 
            {DNNL_ARG_BIAS, bias_mem},
            {DNNL_ARG_DST, dst_mem}
        });
    }
    return true;
}

// The entry point for the DLL.
BOOL DllMain( HMODULE hModule,
                       DWORD  ul_reason_for_call,
                       LPVOID lpReserved
                     )
{
    return TRUE;
}
