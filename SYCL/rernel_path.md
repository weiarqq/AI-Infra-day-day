基于 `llama.esimd/src_granite_support/wrapper/src/llama.esimd.cpp` 里的 `_setup_*` 和 `GetProcAddress` 整理如下。`dnn` 对应 `llama.esimd.dnnl.dll`，`sgr` 对应 `llama.esimd.sgr.dll`，`lgr` 对应 `llama.esimd.lgr.*.dll`。

| 平台 | 算子 | sgr 引用函数 | lgr 引用函数 | dnn 引用函数 | 备注 |
|---|---|---|---|---|---|
| MTL | GEMM Q40 | `getScratchBufferSize_gemm`<br>`runGemm_Q40Weights_L1`<br>`runGemv_Q40Weights_L1`<br>`shuffle_Q40Weights_group128_L1` |  |  | fallback: `lgc` 同名 L1 |
| MTL | GQA | `runGQA_vec_generic`<br>`runGQA_mat_generic`<br>`getScratchBufferSize_gqa` |  |  |  |
| MTL | GEMM Q41 | `getScratchBufferSize_gemm`<br>`runGemm_Q41Weights_L1`<br>`runGemv_Q41Weights_L1`<br>`shuffle_Q41Weights_group32_L1` |  |  |  |
| MTL | StoreCache | `StoreCacheFp16_xve`<br>`StoreCacheQ80_xve` |  |  |  |
| MTL | MHA Q80 | `RunMhaQ80Tri_xve`<br>`RunMhaQ80Tri_xve_hd128`<br>`RunMhaQ80Arb_xve`<br>`RunMhaQ80Arb_xve_hd128` |  |  |  |
| PTL / WCL | FFN |  |  | `getScratchBufferSize_ffn`<br>`runFfnFusion_dnnl` |  |
| PTL / WCL | GEMM Q40 |  |  | `getScratchBufferSize_gemm`<br>`runGemm_Q40Weights_L2`<br>`shuffle_Q40Weights_group128_L2` | `Gemv = Gemm` |
| PTL / WCL | GQA / GQA Masked | `runGQA_vec_generic`<br>`runGQA_mat_generic_xmx_simd16`<br>`runGQA_vec_masked_generic`<br>`runGQA_mat_masked_generic_xmx_simd16`<br>`getScratchBufferSize_gqa` |  |  |  |
| PTL / WCL | GEMM FP16 |  |  | `getVisionScratchBufferSize_gemm`<br>`runGemm_Fp16Weights_WithBias_dnnl` |  |
| PTL / WCL | Norm | `runNorm` |  |  |  |
| PTL / WCL | VIT SDP |  | `runVitSDP_generic_xmx_simd16` |  |  |
| PTL / WCL | GEMM Q41 | `runGemv_Q41Weights_L3` / `L5`<br>fallback: `runGemv_Q41Weights_L1` | `getScratchBufferSize_gemm_q41`<br>`runGemm_Q41Weights_L3_simd16` / `L5_simd16`<br>`shuffle_Q41Weights_group32_L3` / `L5` | fallback: `getScratchBufferSize_gemm_q41`<br>`runGemm_Q41Weights_L1`<br>`shuffle_Q41Weights_group32_L1` | `L3/L5` 由 `compute_units` 或 `LLAMA_ESIMD_GEMM_LAYOUT` 决定 |
| PTL / WCL | Swiglu FFN Q41 | `runSwigluFFnVec_Q41Weights_L3` / `L5` | `getScratchBufferSize_swiglu_ffn`<br>`shuffle_swiglu_Q41Weights_group32_L3` / `L5`<br>`run_q41_ffn_swiglu_L3` / `L5` |  |  |
| PTL / WCL | FFN MOE | `runFfnMoeFusionVec_L2` |  | `getScratchBufferSize_ffnmoe`<br>`runFfnMoeFusion_dnnl` |  |
| PTL / WCL | StoreCache | `StoreCacheFp16_xve`<br>`StoreCacheQ80_xve` |  |  |  |
| PTL / WCL | MHA Q80 | `RunMhaQ80Tri_xve`<br>`RunMhaQ80Tri_xmx_xe2_hd128`<br>`RunMhaQ80Arb_xve`<br>`RunMhaQ80Arb_xmx_xe2_hd128` |  |  |  |
| LNL | FFN |  |  | `getScratchBufferSize_ffn`<br>`runFfnFusion_dnnl` |  |
| LNL | GEMM Q40 |  |  | `getScratchBufferSize_gemm`<br>`runGemm_Q40Weights_L2`<br>`shuffle_Q40Weights_group128_L2` | fallback: `illm` generic |
| LNL | GQA / GQA Masked | `runGQA_vec_generic`<br>`runGQA_mat_generic_xmx_simd16`<br>`runGQA_vec_masked_generic`<br>`runGQA_mat_masked_generic_xmx_simd16`<br>`getScratchBufferSize_gqa` |  |  | fallback: `illm runGQA_generic` |
| LNL | GEMM FP16 |  |  | `getVisionScratchBufferSize_gemm`<br>`runGemm_Fp16Weights_WithBias_dnnl` |  |
| LNL | Norm | `runNorm` |  |  |  |
| LNL | VIT SDP |  | `runVitSDP_generic_xmx_simd16` |  | `lgr.15` |
| LNL | GEMM Q41 | `runGemv_Q41Weights_L3`<br>fallback: `runGemv_Q41Weights_L1` | `getScratchBufferSize_gemm_q41`<br>`runGemm_Q41Weights_L3_simd16`<br>`shuffle_Q41Weights_group32_L3` | fallback: `getScratchBufferSize_gemm_q41`<br>`runGemm_Q41Weights_L1`<br>`shuffle_Q41Weights_group32_L1` |  |
| LNL | Swiglu FFN Q41 | `runSwigluFFnVec_Q41Weights_L3` | `getScratchBufferSize_swiglu_ffn`<br>`shuffle_swiglu_Q41Weights_group32_L3`<br>`run_q41_ffn_swiglu_L3` |  |  |
| LNL | FFN MOE | `runFfnMoeFusionVec_L2` |  | `getScratchBufferSize_ffnmoe`<br>`runFfnMoeFusion_dnnl` |  |
| LNL | StoreCache | `StoreCacheFp16_xve`<br>`StoreCacheQ80_xve` |  |  |  |
| LNL | MHA Q80 | `RunMhaQ80Tri_xve`<br>`RunMhaQ80Tri_xmx_xe2_hd128`<br>`RunMhaQ80Arb_xve`<br>`RunMhaQ80Arb_xmx_xe2_hd128` |  |  |  |
| ARL_H | FFN |  |  | `getScratchBufferSize_ffn`<br>`runFfnFusion_dnnl` |  |
| ARL_H | GEMM Q40 | `runGemv_Q40Weights_L2` |  | `getScratchBufferSize_gemm`<br>`runGemm_Q40Weights_L2`<br>`shuffle_Q40Weights_group128_L2` | fallback: `illm` generic；`lgc` 也可能覆盖 gemv |
| ARL_H | GQA / GQA Masked | `runGQA_vec_generic`<br>`runGQA_mat_generic_xmx_simd8`<br>`runGQA_vec_masked_generic`<br>`runGQA_mat_masked_generic_xmx_simd8`<br>`getScratchBufferSize_gqa` |  |  | fallback: `illm runGQA_generic` |
| ARL_H | GEMM FP16 |  |  | `getVisionScratchBufferSize_gemm`<br>`runGemm_Fp16Weights_WithBias_dnnl` |  |
| ARL_H | Norm | `runNorm` |  |  |  |
| ARL_H | VIT SDP |  | `runVitSDP_generic_xmx_simd8` |  | `lgr.16` |
| ARL_H | GEMM Q41 | `runGemv_Q41Weights_L4`<br>fallback: `runGemv_Q41Weights_L1` | `getScratchBufferSize_gemm_q41`<br>`runGemm_Q41Weights_L4_simd8`<br>`shuffle_Q41Weights_group32_L4` | fallback: `getScratchBufferSize_gemm_q41`<br>`runGemm_Q41Weights_L1`<br>`shuffle_Q41Weights_group32_L1` |  |
| ARL_H | Swiglu FFN Q41 | `runSwigluFFnVec_Q41Weights_L4` | `getScratchBufferSize_swiglu_ffn`<br>`shuffle_swiglu_Q41Weights_group32_L4`<br>`run_q41_ffn_swiglu_L4` |  |  |
| ARL_H | StoreCache | `StoreCacheFp16_xve`<br>`StoreCacheQ80_xve` |  |  |  |
| ARL_H | MHA Q80 | `RunMhaQ80Tri_xve`<br>`RunMhaQ80Arb_xve` | `RunMhaQ80Tri_xmx_xe1_hd128`<br>`RunMhaQ80Arb_xmx_xe1_hd128` |  |  |
| ARL_S | FFN |  |  |  | `xpu: getScratchBufferSize_ffn, runFfnFusion_xpu` |
| ARL_S | GQA-Out |  |  |  | `xpu: getScratchBufferSize_gqao, runGQAOutFusion_xpu` |
| ARL_S | QKVProj |  |  |  | `xpu: getScratchBufferSize_qkvproj, runQkvProjFusion_xpu` |
| ARL_S | GEMM Q40 | `getScratchBufferSize_gemm`<br>`runGemm_Q40Weights_L1`<br>`runGemv_Q40Weights_L1`<br>`shuffle_Q40Weights_group128_L1` |  |  | fallback: `lgc` 同名 L1 |
| ARL_S | GQA | `runGQA_vec_generic`<br>`runGQA_mat_generic`<br>`getScratchBufferSize_gqa` |  |  |  |
| ARL_S | GEMM Q41 | `getScratchBufferSize_gemm`<br>`runGemm_Q41Weights_L1`<br>`runGemv_Q41Weights_L1`<br>`shuffle_Q41Weights_group32_L1` |  |  |  |
| ARL_S | StoreCache | `StoreCacheFp16_xve`<br>`StoreCacheQ80_xve` |  |  |  |
| ARL_S | MHA Q80 | `RunMhaQ80Tri_xve`<br>`RunMhaQ80Tri_xve_hd128`<br>`RunMhaQ80Arb_xve`<br>`RunMhaQ80Arb_xve_hd128` |  |  |  |

---
