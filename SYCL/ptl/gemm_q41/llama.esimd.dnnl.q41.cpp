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

extern "C" size_t __declspec(dllexport) getScratchBufferSize_gemm_q41(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden);
extern "C" bool __declspec(dllexport) runGemm_Q41Weights_L1(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt);
extern "C" void __declspec(dllexport) shuffle_Q41Weights_group128_L1(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len);
extern "C" void __declspec(dllexport) shuffle_Q41Weights_group32_L1(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len);

void dump(sycl::queue*q, uint8_t *data, uint32_t size, const char* filename)
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

size_t getScratchBufferSize_gemm_q41(uint32_t maxBatch, uint32_t contextLen, uint32_t maxHidden)
{
    return (maxBatch * maxHidden + 3072 * maxHidden) * sizeof(fp16);
}

void shuffle_Q41Weights_group128_L1(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        union {
            struct {
                sycl::half d; // delta
                sycl::half m; // min
            };
            uint32_t dm;
        };
        uint8_t qs[128 / 2]; // nibbles / quants
    } block_q4_1;

    block_q4_1* t = (block_q4_1*)input;
    uint8_t* p = (uint8_t *)output;
    sycl::half* h = (sycl::half*)(p + input_len * output_len / 2);
    sycl::half* z = (sycl::half*)(p + input_len * output_len / 2 + input_len * output_len / 128 * sizeof(sycl::half));

    for (int i = 0; i < input_len * output_len / 128; i++)
    {
        memcpy(p, t[i].qs, 128 / 2);

        p += (128 / 2);

        h[i] = t[i].d;
        z[i] = t[i].m;
    }
}

void shuffle_Q41Weights_group32_L1(uint8_t* input, uint8_t* output, uint32_t input_len, uint32_t output_len)
{
    typedef struct {
        union {
            struct {
                sycl::half d; // delta
                sycl::half m; // min
            };
            uint32_t dm;
        };
        uint8_t qs[32 / 2]; // nibbles / quants for 32 elements
    } block_q4_1_32;

    block_q4_1_32* t = (block_q4_1_32*)input;
    uint8_t* p = (uint8_t *)output;
    sycl::half* h = (sycl::half*)(p + input_len * output_len / 2);
    sycl::half* z = (sycl::half*)(p + input_len * output_len / 2 + input_len * output_len / 32 * sizeof(sycl::half));

    for (int i = 0; i < input_len * output_len / 32; i++)
    {
        memcpy(p, t[i].qs, 32 / 2);
        p += (32 / 2);

        h[i] = t[i].d;
        z[i] = t[i].m;
    }
}

void dequantQ41ToFp16(queue &q, unsigned size, uint8_t* weights, sycl::half* scales, sycl::half* zps, sycl::half* output) {
  int groups = size / (1024 * 32);
  int matrixSize = size;
  sycl::range<1> GlobalRange(groups * 32);
  sycl::range<1> LocalRange(32);
  sycl::nd_range<1> Range(GlobalRange, LocalRange);
  sycl::event e;
  try {
    e = q.submit([&](handler& cgh) {
    cgh.parallel_for(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL{
            int hh = ndi.get_local_linear_id();
            int h = ndi.get_group(0);
            int offsetA = (h * 32 + hh) * 1024;
            int offsetQuan = (h * 32 + hh) * 32;
            int offsetZp = (h * 32 + hh) * 32;
            int outputOffset = (h * 32 + hh) * 1024;
            simd<uint8_t, 512> aaa;
            simd<fp16, 1024> aa;
            simd<fp16, 32> ss;
            simd<fp16, 32> zz;
            simd<fp16, 16> cc(0.0f);

            aaa = block_load<uint8_t, 512>(weights + offsetA/2);
            ss = block_load<fp16, 32>(scales + offsetQuan);
            zz = block_load<fp16, 32>(zps + offsetZp);

        #pragma unroll
            for (int k = 0; k < 32; k++) {
                aa.select<16, 1>(32 * k) = aaa.select<16, 1>(16 * k) & 0xf;
                aa.select<16, 1>(32 * k + 16) = aaa.select<16, 1>(16 * k) >> 4;
            }

        #pragma unroll
            for (int k = 0; k < 32; k++) {
              aa.select<32, 1>(32 * k) = ss[k] * aa.select<32, 1>(32 * k);
              aa.select<32, 1>(32 * k) = aa.select<32, 1>(32 * k) + zz[k];
            }

        #pragma unroll
            for (int k = 0; k < 32; k++) {
                __ESIMD_ENS::lsc_block_store<
                fp16,
                32,
                __ESIMD_ENS::lsc_data_size::default_size,
                __ESIMD_ENS::cache_hint::write_back,
                __ESIMD_ENS::cache_hint::write_back>(output + outputOffset + 32 * k, aa.select<32, 1>(32 * k));
            }

        });
    });
  } catch (sycl::exception const &e) {
    std::cout << "SYCL exception caught: " << e.what() << '\n';
    return;
  }
}

bool runGemm_Q41Weights_L1(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt)
{
    const int64_t M = batch;
    const int64_t N = output_len;
    const int64_t K = input_len;

    int64_t G = K / 32;

    sycl::half* f16_weights = (sycl::half *)(shuffleTt + M * K * sizeof(sycl::half));

    dnnl::engine eng = dnnl::sycl_interop::make_engine(q->get_device(), q->get_context());
    //dnnl::engine eng = MatMulPremitiveMgr_q41::Instance().Engine(q);
    dnnl::stream s = dnnl::sycl_interop::make_stream(eng, *q);

    dnnl::memory::data_type dt = (output_precision == 1) ? dnnl::memory::data_type::f16 : dnnl::memory::data_type::f32;
    dnnl::memory::desc src_f32_desc = dnnl::memory::desc({ M, K }, dnnl::memory::data_type::f32, { K, 1 });
    dnnl::memory::desc src_f16_desc = dnnl::memory::desc({ M, K }, dnnl::memory::data_type::f16, { K, 1 });
    dnnl::memory::desc weights_desc = dnnl::memory::desc({ K, N }, dnnl::memory::data_type::f16, dnnl::memory::format_tag::ba);
    dnnl::memory::desc dst_desc = dnnl::memory::desc({ M, N }, dt, { N, 1 });

    dnnl::primitive_attr attr;
    attr.set_fpmath_mode(dnnl::fpmath_mode::f16, true);
    // Create primitive descriptor.
    auto matmul_pd = dnnl::matmul::primitive_desc(eng, src_f16_desc, weights_desc, dst_desc, attr);
    // Create the primitive.
    // TODO: this step may take long time
    auto matmul_prim = dnnl::matmul(matmul_pd);

    //dnnl::matmul matmul_prim = MatMulPremitiveMgr_q41::Instance().Get(M, K, N, output_precision, eng);

    dnnl::memory src_fp16_mem;
    dnnl::memory weights_mem = dnnl::memory(weights_desc, eng, f16_weights);
    dnnl::memory dst_mem = dnnl::memory(dst_desc, eng, outputs);

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

    dequantQ41ToFp16(*q, input_len*output_len, weights, (sycl::half *)scales, (sycl::half *)zps, f16_weights);

    // create GEMM primitative and excute
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC, src_fp16_mem},
        {DNNL_ARG_WEIGHTS, weights_mem},
        {DNNL_ARG_DST, dst_mem}
    };

    matmul_prim.execute(s, args);
    //s.wait();

    //auto end = std::chrono::steady_clock::now();
    //double dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    //printf("GEMM of (%d x %d) x (%d x %d) takes %f us\n", M, K, K, N, dur);


    return true;
}

