intel GPU架构



| 平台             | 典型代号        | GPU 架构倾向           | 典型定位                     | 对 SYCL/LLM 的意义                                           |
| ---------------- | --------------- | ---------------------- | ---------------------------- | ------------------------------------------------------------ |
| `PLATFORM_MTL`   | Meteor Lake     | Xe-LPG                 | Core Ultra 100 移动端        | 老一代 Arc iGPU，适合普通 SYCL/OpenCL/Vulkan，但 XMX/DPAS 路线要谨慎 |
| `PLATFORM_LNL`   | Lunar Lake      | Xe2-LPG                | Core Ultra 200V 低功耗移动端 | 能效很好，Xe2，有 XMX；但 SoC/内存封装限制明显               |
| `PLATFORM_ARL_S` | Arrow Lake-S    | Xe-LPG                 | 桌面端/低配核显              | 桌面 CPU 核显，Xe-core 少，更多是显示/轻量计算，不适合指望大算力 |
| `PLATFORM_ARL_H` | Arrow Lake-H/HX | Xe-LPG+ / 改进移动核显 | 高性能移动端                 | 比 ARL-S 更像“能跑 GPU kernel 的移动核显”，部分资料显示移动 Arrow Lake 支持 DPAS |
| `PLATFORM_PTL`   | Panther Lake    | Xe3 / Arc B 系 iGPU    | Core Ultra Series 3          | 新一代重点，最高到 12 Xe cores，GPU 性能和 XMX 都更值得针对优化 |
| `PLATFORM_WCL`   | Wildcat Lake    | Xe3 小核显             | 主流/低成本移动端            | 也是 Xe3，但规模小，更多是低功耗/入门，不应按 PTL 高配假设   |



基于 `llama.esimd.cpp`，实际调用选择规则是：

- `RunGemmQ40Weights`: `batch <= gemm.slimSupport` 走 `gemv`，否则走 `gemm`。
- `RunGemmQ41Weights`: `batch <= gemm_q41.slimSupport` 走 `gemvq41`，否则走 `gemmq41`。

**平台路径汇总**

| 平台    | Q4_0 GEMM                                                    | Q4_0 GEMV                                                    | Q4_1 GEMM / gemmq41                                          | Q4_1 GEMV / gemvq41                                          |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `MTL`   | 优先 `sgr: runGemm_Q40Weights_L1`，失败再 `lgc: runGemm_Q40Weights_L1` | 优先 `sgr: runGemv_Q40Weights_L1`，失败时 `lgc` 直接复用 GEMM 函数 | `sgr: runGemm_Q41Weights_L1`                                 | `sgr: runGemv_Q41Weights_L1`                                 |
| `LNL`   | 优先 `dnnl: runGemm_Q40Weights_L2`，失败再 `illm: runGemm_Q40Weights_generic` | `dnnl` 下复用 GEMM；`illm` 下也复用 GEMM                     | 优先 `lgr.15: runGemm_Q41Weights_L3_simd16`；失败回退 `dnnl: runGemm_Q41Weights_L1` | 优先 `sgr: runGemv_Q41Weights_L3`；失败回退 `sgr: runGemv_Q41Weights_L1` |
| `ARL_S` | 优先 `sgr: runGemm_Q40Weights_L1`，失败再 `lgc: runGemm_Q40Weights_L1` | 优先 `sgr: runGemv_Q40Weights_L1`，失败时 `lgc` 复用 GEMM    | `sgr: runGemm_Q41Weights_L1`                                 | `sgr: runGemv_Q41Weights_L1`                                 |
| `ARL_H` | 优先 `dnnl: runGemm_Q40Weights_L2`，失败再 `illm: runGemm_Q40Weights_generic` | `dnnl` 后会尝试用 `lgc: runGemv_Q40Weights_L2`，再被 `sgr: runGemv_Q40Weights_L2` 覆盖；`illm` 回退时复用 GEMM | 优先 `lgr.16: runGemm_Q41Weights_L4_simd8`；失败回退 `dnnl: runGemm_Q41Weights_L1` | 优先 `sgr: runGemv_Q41Weights_L4`；失败回退 `sgr: runGemv_Q41Weights_L1` |
| `PTL`   | `dnnl: runGemm_Q40Weights_L2`                                | 复用 `dnnl: runGemm_Q40Weights_L2`                           | 优先 `lgr.17: runGemm_Q41Weights_L3_simd16` 或 `L5_simd16`；失败回退 `dnnl: runGemm_Q41Weights_L1` | 优先 `sgr: runGemv_Q41Weights_L3` 或 `L5`；失败回退 `sgr: runGemv_Q41Weights_L1` |
| `WCL`   | `dnnl: runGemm_Q40Weights_L2`                                | 复用 `dnnl: runGemm_Q40Weights_L2`                           | 同 PTL：`lgr.17 L3/L5`，失败回退 `dnnl L1`                   | 同 PTL：`sgr L3/L5`，失败回退 `sgr L1`                       |

**Q4_1 L3/L5 选择**

只在 `PTL/WCL` 有这个动态选择：

- 默认 `q41_gemm_layout = 3`
- `compute_units == 16` 时改成 `5`
- 环境变量 `LLAMA_ESIMD_GEMM_LAYOUT=3/5` 可以强制覆盖
- layout 3 路径：`gemmq41 = lgr runGemm_Q41Weights_L3_simd16`，`gemvq41 = sgr runGemv_Q41Weights_L3`
- layout 5 路径：`gemmq41 = lgr runGemm_Q41Weights_L5_simd16`，`gemvq41 = sgr runGemv_Q41Weights_L5`

**一句话概括**

- `MTL/ARL_S` 主要走 `sgr/lgc L1`。
- `LNL` Q4_0 走 `dnnl L2`，Q4_1 优先 `lgr+sgr L3`。
- `ARL_H` Q4_0 走 `dnnl L2 + sgr/lgc GEMV`，Q4_1 优先 `lgr+sgr L4`。
- `PTL/WCL` Q4_0 走 `dnnl L2`，Q4_1 优先 `lgr+sgr L3/L5`，失败才回退 `dnnl/sgr L1`。





基于源码看，`L3/L4/L5` 不是同一套 GEMM 简单改名，而是三套 Q4_1 权重 layout + kernel 组合。`lgr` 和 `sgr` 的职责也不同：

- `lgr`: 主要是 prefill/mat GEMM，走 XMX，大 batch。
- `sgr`: 主要是 decode/slim GEMV/GEMM，小 batch。

**总览**

| Layout | `lgr` 用途                                                   | `sgr` 用途                                                   | 平台倾向                              |
| ------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------- |
| `L3`   | `runGemm_Q41Weights_L3_simd16`，XMX simd16                   | `runGemv_Q41Weights_L3`，batch=1 走 `runLinearQ41_L3`，否则 `runSlimGemmQ41_L3` | LNL、PTL/WCL 默认                     |
| `L4`   | `runGemm_Q41Weights_L4_simd8`，XMX simd8；batch<=8 还有 XMX slim path | `runGemv_Q41Weights_L4`，batch=1 走 `runLinearQ41_L4`，否则 `runSlimGemmQ41_L4_XVE` | ARL_H                                 |
| `L5`   | `runGemm_Q41Weights_L5_simd16`，XMX simd16                   | `runGemv_Q41Weights_L5`，直接 `runSlimGemmQ41_L5`            | PTL/WCL 且 `compute_units == 16` 默认 |

**lgr vs sgr**

`lgr` 中的 GEMM 是 mat/prefill 主路径：

```cpp
runGemm_Q41Weights_L3_simd16 -> runGemmXmx16_Q41Weights_L3
runGemm_Q41Weights_L5_simd16 -> runGemmXmx16_Q41Weights_L5
runGemm_Q41Weights_L4_simd8  -> runGemmXmx8_Q41Weights_L4
```

`sgr` 中对应的是 slim/decode 路径：

```cpp
runGemv_Q41Weights_L3 -> batch==1 ? runLinearQ41_L3 : runSlimGemmQ41_L3
runGemv_Q41Weights_L4 -> batch==1 ? runLinearQ41_L4 : runSlimGemmQ41_L4_XVE
runGemv_Q41Weights_L5 -> runSlimGemmQ41_L5
```

所以名字上都是 `L3/L4/L5`，但 `lgr` 是大矩阵 GEMM，`sgr` 是小 batch/GEMV/slim GEMM。

**L3 和 L4 的主要区别**

`L3` 是 `simd16`，输出方向按 32 列组织 scale/zp：

```cpp
for (int j = 0; j < output_len/32; j++)
  for (int k = 0; k < input_len/32; k++)
    for (int i = 0; i < 32; i++)
```

`L4` 是 `simd8`，输出方向按 16 列组织 scale/zp：

```cpp
for (int j = 0; j < output_len/16; j++)
  for (int k = 0; k < input_len/32; k++)
    for (int i = 0; i < 16; i++)
```

也就是说：

- `L3`: output tile = 32，更适合 simd16/Xe2 风格。
- `L4`: output tile = 16，更适合 simd8/Xe1/ARL_H 风格。

这也解释了为什么 `lgr.16.dll` 只导出 `L4_simd8`，而 `lgr.15/17` 使用 `simd16`。

**L5 和 L3 的主要区别**

`L5` 也是 `simd16`，但权重 shuffle 明显不同。

`L3` 权重主体基本是顺序复制 `qs`：

```cpp
for (int i = 0; i < input_len * output_len / 32; i++) {
    memcpy(p, t[i].qs, 32 / 2);
}
```

`L5` 则把 weight nibble 按 `output 16` block 和 `input/32` block 重新交织：

```cpp
p[j * 16 * input_len / 2 + k * 16 * 16 + kk * 16 * 2 + jj * 2] = ...
```

同时 `L5` 的 scale/zp 排列也变成：

```cpp
h[k * output_len + j] = ...
z[k * output_len + j] = ...
```

而 `L3` 是按 output block 再 input block 排：

```cpp
h[idx] = ...
z[idx] = ...
```

所以 `L5` 不是单纯的 `L3` 变体，它是更激进的 weight/scales/zps 内存布局重排，目的是让 kernel 访问更连续、更适配特定 CU 规模。

**为什么 PTL/WCL 有 L3/L5 两套**

wrapper 里：

```cpp
q41_gemm_layout = 3;
if (compute_units == 16) {
    q41_gemm_layout = 5;
}
```

说明 `L5` 是给 `compute_units == 16` 的配置调优的，大概率是 2Xe/较小 GPU 配置。`L3` 是 PTL/WCL 的默认大配置路径。

**为什么 L4 特殊**

`L4` 对应 `lgr.16.dll`，也就是 ARL_H 路径：

```cpp
runGemm_Q41Weights_L4_simd8
```

并且 `lgr` 的 L4 wrapper 里还有小 batch 分支：

```cpp
if (batch <= 8)
    runSlimGemmQ41_L4_XMX(...)
else
    runGemmXmx8_Q41Weights_L4(...)
```

这说明 ARL_H 上，即使在 `lgr` GEMM 路径里，小 batch 也需要单独的 XMX slim kernel。相比 L3/L5，L4 更强调 simd8 和小 batch 适配。

**一句话结论**

- `L3`: Xe2/simd16 的通用 Q4_1 layout，LNL/PTL/WCL 主路径。
- `L4`: ARL_H/Xe1 风格 simd8 layout，output tile 更小，带小 batch 特化。
- `L5`: PTL/WCL 16 CU 配置的 simd16 特化 layout，权重和 scale/zp 排列比 L3 更重排，偏向特定硬件规模优化。
- `lgr` 负责大 batch XMX GEMM，`sgr` 负责 batch=1 或小 batch 的 GEMV/slim 路径。

---