

## [LLM量化](https://github.com/liguodongiot/llm-action/tree/main/model-compression/quantization)


本系列将针对一些常见大模型量化方案（GPTQ、LLM.int8()、SmoothQuant、AWQ等）进行讲述。

- [大模型量化概述](https://www.zhihu.com/question/627484732/answer/3261671478)
- 量化感知训练：
  - [大模型量化感知训练技术原理：LLM-QAT](https://zhuanlan.zhihu.com/p/647589650)
  - [大模型量化感知微调技术原理：QLoRA](https://github.com/liguodongiot/llm-action/blob/main)
  - PEQA
- 训练后量化：
  - [大模型量化技术原理：GPTQ、LLM.int8()](https://zhuanlan.zhihu.com/p/680212402)
  - [大模型量化技术原理：SmoothQuant](https://www.zhihu.com/question/576376372/answer/3388402085)
  - [大模型量化技术原理：AWQ、AutoAWQ](https://zhuanlan.zhihu.com/p/681578090)
  - [大模型量化技术原理：SpQR](https://zhuanlan.zhihu.com/p/682871823)
  - [大模型量化技术原理：ZeroQuant系列](https://zhuanlan.zhihu.com/p/683813769)
  - [大模型量化技术原理：FP8](https://www.zhihu.com/question/658712811/answer/3596678896)
  - [大模型量化技术原理：FP6](https://juejin.cn/post/7412893752090853386)
  - [大模型量化技术原理：KIVI、IntactKV、KVQuant](https://juejin.cn/post/7420231738558627874)
  - [大模型量化技术原理：Atom、QuaRot](https://juejin.cn/post/7424334647570513972)
  - [大模型量化技术原理：QoQ量化及QServe推理服务系统](https://zhuanlan.zhihu.com/p/8047106486)
  - 大模型量化技术原理：QuIP、QuIP#、OmniQuant
  - [大模型量化技术原理：FP4](https://github.com/liguodongiot/llm-action/blob/main)
- [大模型量化技术原理：总结](https://zhuanlan.zhihu.com/p/11886909512)



## 量化方案对比

#### 量化感知训练

| **特性**       | **LLM-QAT (Quantization-Aware Training)**  | **QLoRA (Quantified LoRA)**                          | **PEQA (Parameter-Efficient and Quantization-aware)** |
| -------------- | ------------------------------------------ | ---------------------------------------------------- | ----------------------------------------------------- |
| **核心理念**   | 在微调过程中引入量化误差，使模型适应低精度 | 将预训练模型量化为 4-bit，再通过 LoRA 适配器补偿精度 | 仅针对量化后的比例因子（Scaling Factors）进行微调     |
| **显存占用**   | **高**（通常需要全参数或大规模梯度的更新） | **极低**（通过 4-bit 基础权重和分页优化显著降低）    | **低**（只更新极少量的量化参数）                      |
| **训练精度**   | **最高**（能极好地保持甚至提升低精度性能） | **高**（接近 16-bit 全微调精度）                     | **中等**（受限于仅能微调比例因子）                    |
| **主要技术点** | 知识蒸馏 (KD)、量化感知梯度更新            | 4-bit NormalFloat (NF4)、双重量化、分页优化器        | 权重/激活值量化、比例因子微调                         |
| **权重状态**   | 权重在训练中保持浮点并模拟量化行为         | 基础权重冻结在 4-bit，LoRA 层为 16-bit               | 基础权重冻结，仅优化量化相关的标量                    |
| **适用场景**   | 对推理延迟极其敏感、需要极致压缩的生产环境 | 个人开发者或资源有限的情况下微调大型模型             | 极端轻量级的适配，关注参数量极简化的场景              |

#### 训练后量化

| **量化方法**    | **主要类型** | **量化对象** | **典型位数** | **核心原理 / 特点**                                          | **精度保留**         | **推理速度提升**      |
| --------------- | ------------ | ------------ | ------------ | ------------------------------------------------------------ | -------------------- | --------------------- |
| **GPTQ**        | PTQ (仅权重) | Weight       | 3/4-bit      | 基于 Hessian 矩阵的二阶信息，逐层补偿量化误差。              | 高 (4-bit 接近无损)  | 显著 (Weight-only)    |
| **AWQ**         | PTQ (仅权重) | Weight       | 3/4-bit      | **激活感知**。通过观察激活分布，保护 1% 的显著权重不被量化或进行缩放。 | 极高 (通常优于 GPTQ) | 显著 (硬件适配好)     |
| **LLM.int8()**  | PTQ (混合)   | W + A        | 8-bit        | **离群值分离**。将激活中的离群列转为 FP16 处理，其余矩阵乘法用 INT8。 | 几乎无损             | 较小 (有转换开销)     |
| **SmoothQuant** | PTQ (全量)   | W + A        | 8-bit        | **数学等价变换**。通过缩放因子将激活值的量化难度转移到权重上，使 W 和 A 都能 INT8 量化。 | 高 (8-bit 无损)      | 显著 (支持高效内核)   |
| **SpQR**        | PTQ (混合)   | Weight       | 2/3/4-bit    | **稀疏量化**。识别对精度至关重要的极少数异常权重并保持高精度，其余高度压缩。 | 极高 (支持极低比特)  | 中等 (稀疏计算较复杂) |
| **ZeroQuant**   | PTQ (全量)   | W + A        | 8-bit        | **逐标记/逐通道量化**。结合高度优化的计算内核和底层硬件加速。 | 良好                 | 显著 (面向生产环境)   |



#### KV量化

| 量化方法     | 关注点    | 主要对象   | 位数 (典型) | 核心原理 / 特点                                              | 精度保留                | 适用场景           |
| ------------ | --------- | ---------- | ----------- | ------------------------------------------------------------ | ----------------------- | ------------------ |
| **KIVI**     | KV 缓存   | KV Cache   | 2/4-bit     | **非均匀量化**。发现 Key 具有通道分布，Value 具有逐标记分布，采用不同的量化维度（分组/逐通道）。 | 良好 (Llama-2 无损)     | 超长文本序列生成   |
| **IntactKV** | KV 缓存   | KV Cache   | 4/8-bit     | **完整性保护**。识别并保留 KV 中的“初始标记”和“离群点”不量化，只量化剩余平滑部分。 | 极高 (由于保护了关键列) | 追求高精度的长文本 |
| **KVQuant**  | KV 缓存   | KV Cache   | 2/3/4-bit   | **NUQ (非均匀量化)**。利用二阶信息（Hessian）进行校准，并采用感知敏感度的位分配。 | 高 (支持极低比特)       | 极致显存压缩       |
| **Atom**     | 效率/吞吐 | W + A + KV | 4-bit       | **端到端量化**。通过混合精度和细粒度量化，结合算子对齐，在保持精度的同时大幅提升吞吐量。 | 良好                    | 高吞吐量生产环境   |
| **QuaRot**   | 全模型    | W + A + KV | 4-bit       | **旋转变换**。引入 Hadamard 变换旋转权重和激活的特征空间，**消除离群值**，实现全 4-bit 推理。 | 极高 (理论优雅)         | 4-bit 全链路加速   |

---

**核心技术差异分析**

1. 解决问题的思路：量化 vs. 旋转

* **KIVI & IntactKV**：属于“发现规律并适配”型。它们研究 KV Cache 的分布特性（比如哪些列是离群值，哪些维度更重要），然后针对性地保护或采用不同的量化策略。
* **QuaRot**：属于“改造分布”型。它通过数学上的旋转（Rotary）操作，将难以量化的“尖锐”分布（离群值）拉平。这样模型就不再有离群点，从而可以简单地进行 4-bit 全量化。

2. KV Cache 压缩的深度 (KIVI vs. KVQuant)

* **KIVI**：主要通过对 Key 和 Value 采取不同的量化维度（Key 逐通道，Value 逐标记）来平衡精度，实现简单直接。
* **KVQuant**：引入了更复杂的二阶信息校准（类似 GPTQ 对权重的处理），使得 KV Cache 即使压缩到 2-bit 或 3-bit 仍能保持较好的语言模型性能。

3. 生产环境的综合性能 (Atom)

* **Atom**：不仅关注量化算法本身，更关注“硬件如何高效执行”。它优化了 4-bit 的计算内核，使得在节省显存的同时，推理吞吐量（Throughput）能有数倍的提升，是更偏向工程落地的方法。
4. IntactKV 的独特性

* **IntactKV** 强调了 LLM 在长文本中对“前几个标记（Initial Tokens）”的高度依赖。通过将这些关键信息保持在 FP16，而量化其他部分，它能以极小的空间代价换取极高的逻辑一致性。

---

## ggml量化

#### 传统线性量化 (Legacy Quantization)

```c++
// 量化
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0; //32

    assert(k % qk == 0); // 确保输入长度是 32 的倍数

    const int nb = k / qk; // 计算总块数

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v); // 记录绝对值最大的数值
                max  = v;				 // 记录该最大值原始的正负号
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}


// 反量化
static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}
```



这段代码是 `ggml` 库（常用于 Llama.cpp 等大模型推理框架）中 **Q4_0 量化算法**的参考实现。

它的核心功能是将一组 **32 位浮点数 (FP32)** 压缩为 **4 位整数 (4-bit)**，从而大幅减少模型权重的体积并加速计算。

------

 1. 核心概念：什么是 Q4_0？

Q4_0 是一种“对称块量化”方式。

- **分块 (Blocking)**：它不一次性量化整个向量，而是每 **32 个元素**（代码中的 `qk=32`）分为一个块。

- **存储结构**：每个块由一个 **FP16 缩放系数 (Scale)** 和 **16 个字节（32 个 4-bit 权重）** 组成。

- 数学公式：对于块内的每个值 $x$，量化后的 4 位整数 $q$ 满足：

  

  $$x \approx d \cdot (q - 8)$$

  

  这里 $d$ 是缩放系数，$-8$ 是为了将无符号的 0-15 映射回有符号的范围。

------

 2. 代码逻辑逐行拆解

 第一阶段：初始化与分块


```c++
static const int qk = 32;
assert(k % qk == 0); // 确保输入长度是 32 的倍数
const int nb = k / qk; // 计算总块数
```

代码将输入数组 `x` 按 32 个元素一组进行处理。`nb` 是总的块数，每一块都会生成一个 `block_q4_0` 结构体。

 第二阶段：寻找缩放系数 (Scale)

```c++
float amax = 0.0f; 
float max  = 0.0f;

for (int j = 0; j < qk; j++) {
    const float v = x[i*qk + j];
    if (amax < fabsf(v)) {
        amax = fabsf(v); // 记录绝对值的最大值
        max  = v;        // 记录该最大值原始的正负号
    }
}
```

在该块的 32 个数中找到绝对值最大的数。

第三阶段：量化与打包

```c++
const float d  = max / -8;
const float id = d ? 1.0f/d : 0.0f; // 预计算倒数，变除法为乘法以提高效率
y[i].d = GGML_FP32_TO_FP16(d);      // 缩放系数存为 FP16 节省空间
```



**关键点**：`d = max / -8`。这里为什么要除以 -8？

- 4 位整数能表示的范围是 0 到 15。
- 在 Q4_0 标准中，映射的中点是 8。
- 通过将最大值映射到边缘，可以最大程度保留数值的精度。

> 假设max为正数，则我们需要将 x(0～15) 映射到y(~, max) 并且 原始数的0 映射到8，假设函数 y=dx+b; x是映射后到值
>
> $x= 0, y=max 得 b=max;$
>
> $x = 8, y=0 得 8w+b = 0, 由于b=max,则 8w+max =0， 得 d = max/-8$
>
> $x = y/d - max/d ==> x = y/d  + 8$
>
> 再看反量化：
>
> $y = dx+b; d = max/-8, b = max, 则 b = -8*d$
>
> $y = dx -8*d ==> y = d*(x-8)$



>Q4_1 也是类似逻辑，因为有最大和最小值了则 x(0~15) 映射到y(min, max)
>
>$y = dx + b$
>
>$x = 0,y=min 得 b = min;$
>
>$x = 15,y=max 得 max-min = 15d, d=(max-min)/15$
>
>$x = (y-b)/d = (y-min)*(1/d)$
>
>再看反量化
>
>$y = dx + b$
>
>$b = min,  y=dx+min$





接下来的循环处理 4-bit 的映射：


``` c++
for (int j = 0; j < qk/2; ++j) {
    const float x0 = x[i*qk + 0    + j]*id;     // 归一化处理
    const float x1 = x[i*qk + qk/2 + j]*id;

    const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f)); // 加上偏移量 8 并四舍五入
    const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

    y[i].qs[j]  = xi0;          // 将第一个 4-bit 放入低 4 位  00001010
    y[i].qs[j] |= xi1 << 4;     // 将第二个 4-bit 放入高 4 位	 10100000 取|逻辑 10101010
}
```

- **映射过程**：`x * id` 将数值缩放到 $[-8, 8]$ 左右。加上 `8.5f` 是为了实现四舍五入并平移到 $[0, 15]$。
- **打包 (Packing)**：因为一个 `uint8_t` 有 8 位，而一个量化值只有 4 位，所以代码将两个量化值（`xi0` 和 `xi1`）合并存储在一个字节中。

------

3. 内存结构示意

| **组成部分** | **类型**      | **大小** | **说明**                                   |
| ------------ | ------------- | -------- | ------------------------------------------ |
| **d**        | `half` (FP16) | 2 Bytes  | 该块 32 个数的共同缩放比例                 |
| **qs**       | `uint8_t[16]` | 16 Bytes | 存储 32 个 4-bit 索引 (16 * 8bit = 128bit) |

总计：每个块占用 18 字节，存储 32 个权重。

压缩比：原始 FP32 需要 $32 \times 4 = 128$ 字节。压缩后仅 18 字节，压缩率约为 7.11 倍。

------

4. 总结

这段代码通过以下步骤实现了 **FP32 $\rightarrow$ Q4_0/1** 的转换：

1. **分块**：每 32 个数一组。
2. **找最大值**：计算缩放系数 `d` 并转为 FP16。
3. **线性映射**：将浮点数映射到 0-15 的整数区间。
4. **位拼装**：将两个 4-bit 整数塞进一个 8-bit 字节。

这种做法虽然会引入少量的精度损失（量化误差），但能显著降低显存占用，是本地运行大语言模型（如 Llama 3）的核心技术。

您是想了解如何将这段代码适配到特定的硬件加速（如 AVX 或 CUDA），还是想了解如何反量化回浮点数？

Q5和Q8方式一致，区别主要在存储量化后的权重，Q4可以两个参数组成一个字节，Q5和Q8则不行

Q5

```c++
const float x0 = (x[i*qk + 0    + j] - min)*id;
const float x1 = (x[i*qk + qk/2 + j] - min)*id;

const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

// get the 5-th bit and store it in qh at the right position
qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
```

`xi0 & 0x0F`：取出第一个权重的低 4 位。

`(xi1 & 0x0F) << 4`：取出第二个权重的低 4 位，并向左移 4 位。

这两个 4-bit 被拼成了一个完整的 8-bit 字节，存入 `qs` 数组中。这部分和 Q4_0 的逻辑完全一样。

`xi0 & 0x10u`：`0x10` 二进制是 `0001 0000`。这步操作是检查第 5 位是否为 1。

`>> 4`：将这个第 5 位移到最低位（变成 0 或 1）。

`<< (j + 0)`：将这个 0 或 1 移动到 `qh` 对应的位置。例如，如果是该 Block 的第 3 个元素（j=2），它就会被移到 `qh` 的第 2 位。

`j + qk/2`：处理 Block 的后半部分。如果 Block 大小为 32，前半部分的第 5 位占 `qh` 的 0-15 位，后半部分占 16-31 位。



#### K-Quants 系列 (K-Methods)（推荐）

| **量化等级** | **推荐程度**        | **描述**                                                     |
| ------------ | ------------------- | ------------------------------------------------------------ |
| **Q2_K**     | 低                  | 极端压缩，仅用于显存极小的设备。逻辑极其模糊，模型表现下降严重。 |
| **Q3_K_M/L** | 中                  | 3-bit 量化。M（Medium）和 L（Large）在不同层使用不同位数，适合低配置。 |
| **Q4_K_M**   | **极高 (最佳平衡)** | **目前的行业标准**。在关键矩阵上使用更高位数。精度非常接近 FP16，但体积缩小约 4 倍。 |
| **Q4_K_S**   | 中                  | 相比 M 版本，S（Small）更追求体积，牺牲了一点精度。          |
| **Q5_K_M**   | 高                  | 如果你的显存允许，这是比 Q4 更稳妥的选择，精度损失几乎不可察觉。 |
| **Q6_K**     | 高                  | 极其接近原始模型。虽然体积比 Q8 小，但性能几乎一致。         |



Q4_K实现

```c++
void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K]; 					// 存储每个元素的4-bit量化值（中间态）
    uint8_t Laux[32];					// 误差优化时的临时量化值
    float   weights[32];			// 加权量化的权重
    float mins[QK_K/32];			// 每个32元素子块的偏移（min）
    float scales[QK_K/32];		// 每个32元素子块的缩放因子（scale）

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; 	// 所有32子块中最大的scale（用于全局归一化）
        float max_min = 0;		// 所有32子块中最大的min（用于全局归一化）
        for (int j = 0; j < QK_K/32; ++j) {
            
          	// 实际上计算的是这一组数据的均方根（RMS）。它代表了这组数据的“平均能量强度”。
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
          
          	// 如果直接量化，所有元素的地位是平等的。但实际上，模型对大数值（Outliers/离群值）的误差极其敏感。
            // 大数值元素： fabsf(x[i]) 很大，导致其权重 w 很高。
            // 小数值元素： 权重接近 av_x（平均能量水平）。
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
          	// 步骤3.2：调用核心函数计算该子块的最优scale和min
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
             // 步骤3.3：记录所有子块的最大scale和min（用于全局归一化）
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }
        // 将 scale/min 压缩存储到 block_q4_K 的 scales 字段
				// q4_K 的核心优化点：将 8 个子块的 scale 和 min 量化为 uint8_t，并压缩存储到 8 个字节的 scales 数组中（空间优化）。
				// 计算全局归一化因子（将scale/min映射到0~63范围）
        // 0~63只占 6bit
        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
        		// 将scale/min量化为0~63的整数
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            // 压缩存储（核心空间优化）
          	// 转为6bit后，uint8为8bit，会浪费2bit, 所以切分为高位2bit,低位4bit, 
          	// 共12块，实际应该用到 2*8=16块
            // 前4块存 正常存，不切占6bit, 高位2bit存后续4个块的高位2bit
            // 后面 ls和lm低四位拼接，高2位放到前面的高2位。。。
            if (j < 4) {
            		// 前4个子块：低6位存scale，高2位后续补；先存scale和min的低4位
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
            		// 后4个子块：
        				// scales[j+4]：低4位=scale低4位，高4位=min低4位
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                // scales[j-4]：高2位（bit6-7）存scale高2位
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                // scales[j]：高2位存min高2位
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        // 存储全局归一化基数（fp16节省空间）
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);    // scale的全局基数
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);   // min的全局基数
				// 基于压缩的 scale/min 重新计算最终量化值
				//	从 scales 数组中解析出每个子块的 scale/min，重新计算 256 元素的 4-bit 量化值：
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
        		// 从scales数组解析该子块的scale/min量化值
            get_scale_min_k4(j, y[i].scales, &sc, &m);
             // 计算最终的scale和min（全局基数*解析值）
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue; // 无意义值跳过
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
             // 将float32映射到0~15的4-bit整数
            for (int ii = 0; ii < 32; ++ii) {
            		// 公式：量化值 = (原始值 + 偏移) / 缩放因子 → 四舍五入
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l)); // 限制在4-bit范围（0~15）
                L[32*j + ii] = l;
            }
        }
				// 将 4-bit 量化值打包存储到 qs 字段
				// 2个4-bit数打包为1个 uint8_t（节省空间），256个4-bit数最终存为128个 uint8_t：
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) { // 每次处理64个元素（32个uint8_t）
            for (int l = 0; l < 32; ++l) {
            	// 低4位=第j+l个元素，高4位=第j+l+32个元素
            	q[l] = L[j + l] | (L[j + l + 32] << 4);
            }
            
            q += 32;
        }
				// 移动指针，处理下一个256元素块
        x += QK_K;
    }
}

static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    // 遍历所有元素，找min/max，计算加权和
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    // 边界处理：min不能为正（保证偏移为负，映射到0开始）
    if (min > 0) min = 0;
    // 所有元素相同：量化值全0，scale=0
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    // 初始缩放因子：将[min, max]映射到[0, nmax]（nmax=15）
    float iscale = nmax/(max - min);
    float scale = 1/iscale; // 逆缩放因子（量化后转回浮点数用）
    float best_error = 0;
    // 计算初始量化值，并统计量化误差（加权MSE/MAE）
    for (int i = 0; i < n; ++i) {
    		// 线性映射：(x[i] - min) * iscale → 四舍五入
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l)); // 限制范围
        // 计算误差：量化值转回浮点数 - 原始值
        float diff = scale * L[i] + min - x[i];
        // use_mad=false：误差用平方（MSE）；true：用绝对值（MAE）
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff; // 加权误差和
    }
    // 无需误差优化：直接返回结果
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
    		// 生成候选iscale（在初始值基础上微调）
    		// rmin=-1.f rdelta=0.1
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        // 统计加权统计量（用于计算最优scale/min）
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l; // 临时存储候选量化值
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        // 计算最优scale和min（最小二乘法）
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
        		// 最优scale = (sum_w*sum_xl - sum_x*sum_l)/D
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            // 最优min = (sum_l2*sum_x - sum_l*sum_xl)/D
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            // min不能为正（边界限制）
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            // 计算候选方案的误差
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            // 误差更小：更新最优解
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    // 返回最优scale，min通过指针输出（取负后存储）

    *the_min = -min;
    return scale;
}

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f); 		// 限制范围（避免溢出）
    float val = fval + 12582912.f;				// 偏移到固定整数范围
    int i; memcpy(&i, &val, sizeof(int));	// 浮点转整数（利用IEEE754存储特性）
    return (i & 0x007fffff) - 0x00400000;	// 提取有效位并还原
}

// 复原 拼接的scale和min
static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```



**make_qkx2_quants实现原理** 

一、先明确我们要解决的核心问题
在量化场景中，我们已经有了：
- 原始浮点值：$x_0, x_1, ..., x_{31}$（32个元素）
- 每个值的权重：$w_0, w_1, ..., w_{31}$（加权，让大数值误差更受重视）
- 候选量化整数：$l_0, l_1, ..., l_{31}$（由初始`iscale`计算出的0~15整数）

我们要找两个参数：
- $s$（`this_scale`）：缩放因子
- $b$（`this_min`）：偏移量

使得**加权平方误差和最小**（误差 = 原始值 - 量化还原值）：
$$
\text{Error} = \sum_{i=0}^{31} w_i \cdot (x_i - (s \cdot l_i + b))^2 \quad \text{(目标：让Error最小)}
$$

二、公式推导（从误差最小到代码中的表达式）
要让Error最小，需对$s$和$b$分别求偏导，并令偏导=0（极值条件）。

步骤1：对偏移量 $b$ 求偏导并令其为0
对Error关于$b$求偏导：
$$
\frac{\partial Error}{\partial b} = \sum_{i=0}^{31} w_i \cdot 2 \cdot (x_i - s l_i - b) \cdot (-1) = 0
$$
两边除以-2，整理得：
$$
\sum_{i=0}^{31} w_i (x_i - s l_i - b) = 0
$$
展开后拆分求和项：
$$
\sum w_i x_i - s \sum w_i l_i - b \sum w_i = 0
$$
变形得到第一个方程（记为公式1）：
$$
s \cdot \sum w_i l_i + b \cdot \sum w_i = \sum w_i x_i \tag{1}
$$

步骤2：对缩放因子 $s$ 求偏导并令其为0
对Error关于$s$求偏导：
$$
\frac{\partial Error}{\partial s} = \sum_{i=0}^{31} w_i \cdot 2 \cdot (x_i - s l_i - b) \cdot (-l_i) = 0
$$
两边除以-2，整理得：
$$
\sum_{i=0}^{31} w_i l_i (x_i - s l_i - b) = 0
$$
展开后拆分求和项：
$$
\sum w_i l_i x_i - s \sum w_i l_i^2 - b \sum w_i l_i = 0
$$
变形得到第二个方程（记为公式2）：
$$
s \cdot \sum w_i l_i^2 + b \cdot \sum w_i l_i = \sum w_i l_i x_i \tag{2}
$$

步骤3：定义代码中的统计量（简化书写）
为了和代码一一对应，先定义代码中已计算的统计量：
| 代码变量 | 数学表达式         | 含义                      |
| -------- | ------------------ | ------------------------- |
| `sum_w`  | $\sum w_i$         | 所有权重之和              |
| `sum_l`  | $\sum w_i l_i$     | 加权量化整数之和          |
| `sum_l2` | $\sum w_i l_i^2$   | 加权量化整数平方和        |
| `sum_x`  | $\sum w_i x_i$     | 加权原始值之和            |
| `sum_xl` | $\sum w_i l_i x_i$ | 加权（量化整数×原始值）和 |

将这些代入公式1和公式2，得到二元一次方程组：
$$
\begin{cases}
s \cdot sum\_l + b \cdot sum\_w = sum\_x \quad (1) \\
s \cdot sum\_l2 + b \cdot sum\_l = sum\_xl \quad (2)
\end{cases}
$$
步骤4：用克莱姆法则解方程组
对于二元一次方程组：
$$
\begin{cases}
a_1 s + b_1 b = c_1 \\
a_2 s + b_2 b = c_2
\end{cases}
$$
克莱姆法则的解为：
$$
s = \frac{\begin{vmatrix} c_1 & b_1 \\ c_2 & b_2 \end{vmatrix}}{\begin{vmatrix} a_1 & b_1 \\ a_2 & b_2 \end{vmatrix}}, \quad b = \frac{\begin{vmatrix} a_1 & c_1 \\ a_2 & c_2 \end{vmatrix}}{\begin{vmatrix} a_1 & b_1 \\ a_2 & b_2 \end{vmatrix}}
$$
其中分母是系数行列式 $D = a_1 b_2 - a_2 b_1$（必须>0，否则无解）。

对应到我们的方程组：
- $a_1 = sum\_l, b_1 = sum\_w, c_1 = sum\_x$
- $a_2 = sum\_l2, b_2 = sum\_l, c_2 = sum\_xl$

第一步：计算系数行列式 $D$（代码中的`D`）
$$
D = a_1 b_2 - a_2 b_1 = sum\_l \cdot sum\_l - sum\_l2 \cdot sum\_w \quad?
$$
⚠️ 注意：代码中是 `sum_w * sum_l2 - sum_l * sum_l`，和上面符号相反——这是因为行列式的分子也会同步变号，最终$s$和$b$的结果不变（负负得正）。
代码中写 `sum_w * sum_l2 - sum_l * sum_l` 是为了让$D$为正（后续判断`D>0`），避免分母为负影响计算。

第二步：计算 $s$（代码中的`this_scale`）
分子是替换第一列后的行列式：
$$
\begin{vmatrix} c_1 & b_1 \\ c_2 & b_2 \end{vmatrix} = sum\_x \cdot sum\_l - sum\_xl \cdot sum\_w
$$
结合分母$D$，最终：
$$
s = \frac{sum\_w \cdot sum\_xl - sum\_x \cdot sum\_l}{sum\_w \cdot sum\_l2 - sum\_l \cdot sum\_l}
$$
这完全对应代码：
```c
float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
```

第三步：计算 $b$（代码中的`this_min`）
分子是替换第二列后的行列式：
$$
\begin{vmatrix} a_1 & c_1 \\ a_2 & c_2 \end{vmatrix} = sum\_l \cdot sum\_xl - sum\_l2 \cdot sum\_x
$$
结合分母$D$，最终：
$$
b = \frac{sum\_l2 \cdot sum\_x - sum\_l \cdot sum\_xl}{sum\_w \cdot sum\_l2 - sum\_l \cdot sum\_l}
$$
对应代码：
```c
float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
```

三、代码中`if (D > 0)`的意义
$D$是系数行列式，$D=0$意味着：
- 两个方程线性相关（比如所有$l_i$都相同），无法解出唯一的$s$和$b$；
- 此时最小二乘法无意义，直接跳过该轮优化。

只有$D>0$时，方程组有唯一解，才会计算`this_scale`和`this_min`。

**总结**
1. 代码中的`D`是线性方程组的**系数行列式**，必须>0才能解出唯一的$s$和$b$；
2. `this_scale`是最小二乘法解出的**最优缩放因子**，`this_min`是**最优偏移量**；
3. 整个推导的核心是「对加权平方误差求偏导并令其为0」，最终得到的解析解直接对应代码中的表达式，没有任何近似。



#### I-Quants (Importance Matrix)

**核心思想：重要性矩阵 (imatrix)**
在神经网络中，并非所有权重都同等重要。某些权重即便量化误差很大，对最终结果影响也很小；而另一些权重稍有偏差，就会导致模型输出乱码。

数据驱动：I-Quants 需要一个训练阶段。开发者会提供一段通用的文本数据集（如 Wiki 数据），让模型跑一遍（Forward pass）。

敏感度收集：在跑的过程中，程序会记录每个权重张量的贡献度，生成一个 imatrix.dat 文件。这个文件告诉量化器：“这一块权重非常关键，请给它分配最高精度；那一块不重要，可以暴力压缩。”



| **特性**       | **K-Quants (传统)** | **I-Quants (imatrix)**         |
| -------------- | ------------------- | ------------------------------ |
| **依赖性**     | 仅依赖模型静态权重  | 依赖参考数据集（imatrix）      |
| **低比特表现** | 3-bit 以下逻辑崩溃  | **2.5-bit 仍能保持基本逻辑**   |
| **计算开销**   | 量化速度快          | 量化速度慢（需要预跑 imatrix） |
| **推理速度**   | 极快，针对 CPU 优化 | 略慢（解包逻辑更复杂）         |

IQ3_xxs实现(quantize_row_iq3_xxs_impl)

函数作用：IQ3_XXS 量化的核心实现，将浮点张量 `x` 量化为 IQ3_XXS 格式存储到 `vy`。

参数说明：

- `grid_size`：量化网格大小（256 或其他，对应不同 IQ3 变体）；
- `x`：输入浮点张量（待量化）；
- `vy`：输出量化后的数据指针（存储尺度、符号、量化索引）；
- `n`：输入张量元素总数；
- `quant_weights`：量化权重（可选，用于加权量化，提升精度）；
- `GGML_RESTRICT`：编译器优化标记，表明指针无重叠，提升访问效率。

```c++
static void quantize_row_iq3_xxs_impl(
  	int grid_size, 
  	const float * GGML_RESTRICT x, 
  	void * GGML_RESTRICT vy, 
  	int64_t n, 
  	const float * GGML_RESTRICT quant_weights
) {
		// 根据网格大小获取预初始化的 IQ3 数据索引；
    const int gindex = iq3_data_index(grid_size);
		// 预生成的量化网格（存储 4 元素组的量化值组合）；
    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    // 网格映射表（将量化值组合映射到网格索引）；
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    // 网格邻居表（当量化值不在网格上时，找最优邻居）。
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    // GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 8; // 量化值的最大索引（对应 3bit：0~7）

    const int64_t nbl = n/QK_K; // 总块数（每个块 QK_K 个元素，通常 QK_K=256）
		
		// 根据 grid_size 选择对应的量化块结构（block_iq3_xxs/block_iq3_s）；
		// dh：指向块的全局尺度（fp16 类型，压缩存储）；
		// qs：指向量化后的数据区（存储网格索引、符号、子块尺度）；
		// quant_size：量化数据区的字节数（块大小 - 全局尺度的字节数）。

    ggml_fp16_t * dh;
    uint8_t * qs;
    int block_size;
    if (grid_size == 256) {
        block_iq3_xxs * y = vy;
        dh = &y->d; // 块的全局尺度（fp16 存储）
        qs = y->qs; // 块的量化索引/符号/尺度编码
        block_size = sizeof(block_iq3_xxs);
    } else {
        block_iq3_s * y = vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_s);
    }
    int quant_size = block_size - sizeof(ggml_fp16_t); // 量化数据部分的长度（排除全局尺度）

    float scales[QK_K/32]; // 每个 32 元素子块的尺度
    float weight[32];      // 每个子块内元素的加权系数
    float xval[32];        // 子块元素的绝对值（符号单独存储）
    int8_t L[32];          // 子块元素的量化索引（0~7）
    int8_t Laux[32];       // 临时量化索引（用于迭代优化）
    float  waux[32];       // 临时加权系数（平方根）
    bool   is_on_grid[8];  // 标记 4 元素组是否在预定义网格上
    bool   is_on_grid_aux[8]; // 临时网格标记
    uint8_t block_signs[8];// 存储 8 元素组的符号（每 bit 表示一个元素的正负）
    uint8_t q3[3*(QK_K/8)+QK_K/32]; // 临时存储量化结果（索引+符号+尺度）
    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K/4); // 符号+子块尺度的编码区
    uint8_t  * qh = q3 + 3*(QK_K/8); // 高比特网格索引（grid_size>256 时用）
    
    // 主量化循环（按块处理）
    for (int ibl = 0; ibl < nbl; ++ibl) {
				// 初始化当前块的全局尺度为 0，量化缓冲区清零
        dh[0] = GGML_FP32_TO_FP16(0.f);
        memset(q3, 0, 3*QK_K/8+QK_K/32);

        float max_scale = 0; // 记录当前块所有子块的最大尺度

        const float * xbl = x + QK_K*ibl; // 当前块的输入数据指针
        // 计算当前块的平方和，用于后续加权系数计算
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K; // 方差类参数（用于加权量化）
				// 子块处理（32 元素 / 子块）
        for (int ib = 0; ib < QK_K/32; ++ib) { // 遍历每个 32 元素子块
            const float * xb = xbl + 32*ib; 	 // 当前子块的输入数据指针
            // 计算加权系数 weight
            if (quant_weights) {
             		// 有量化权重时：weight[i] = 量化权重 * sqrt(方差 + 元素平方)
                const float * qw = quant_weights + QK_K*ibl + 32*ib;
                for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                // 无量化权重时：weight[i] = 元素平方（简单加权）
                for (int i = 0; i < 32; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]); // 加权系数平方根
            // 符号优化（8 元素组）
            // 处理符号（将负数转为正数，符号单独存储，保证偶翻转）
            for (int k = 0; k < 4; ++k) { // 32 元素拆分为 4 个 8 元素组
                int nflip = 0; // 负数个数
                uint8_t s = 0; // 符号掩码（bit i=1 表示第 i 个元素是负数）
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0){
                    	xval[8*k + i] = xb[8*k + i]; // 正数直接存
                    }
                    else {
                        xval[8*k + i] = -xb[8*k + i];  // 负数取绝对值
                        ++nflip;
                        s |= (1 << i); // 标记符号
                    }
                }
                // 保证翻转次数为偶数（避免符号误差累积）
                if (nflip%2) {
                    // 找加权最小的元素，翻转其符号（使总翻转数为偶）
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin]; // 翻转符号
                    s ^= (1 << imin);                 // 更新符号掩码
                }
                block_signs[k] = s & 127;             // 存储符号掩码（7bit 足够，8th bit 留作他用）
            }
            // 尺度初始化与网格匹配
            // 计算子块的最大绝对值，初始化尺度
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            if (max < GROUP_MAX_EPS_IQ3_XXS) { // 最大值过小，直接量化为 0
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1); 					// 初始尺度（将 max 映射到 2*8-1=15）
            for (int k = 0; k < 8; ++k) is_on_grid[k] = true; // 初始化网格标记
            // 迭代优化尺度（遍历 31 个候选尺度）
            for (int is = -15; is <= 15; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max; // 尺度倒数（迭代调整）
                float this_scale = 1/id;						// 当前候选尺度
                // 计算每个4元素组的量化索引，并检查是否在网格上
                for (int k = 0; k < 8; ++k) { 			// 32 元素拆分为 8 个 4 元素组
                    for (int i = 0; i < 4; ++i) {
                    		// 量化索引计算：Laux = 0.5*(id*xval -1) 取整，限制在 0~7
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    // 将4个3bit索引打包为12bit整数（4*3=12）
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u]; // 查找网格索引
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {	// 不在预定义网格上
                        is_on_grid_aux[k] = false;
                        // 找最优邻居（通过邻居表）
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                // 计算当前尺度的误差（加权平方和），找最优尺度
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1; // 量化值（索引转实际值：0→1, 7→15）
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                // 更新最优尺度和量化索引
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; 
                    best = scale*sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            // 非网格元素的二次优化
            // 对不在网格上的 4 元素组，重新找最优邻居并更新尺度
            int n_not_ongrid = 0;
            for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 8; ++k) {
                    if (is_on_grid[k]) continue; // 只处理非网格组
                    // 重新计算量化索引
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                    		// 找最优邻居
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    // 更新量化索引（从网格值转换）
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                // 重新计算最优尺度
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            // 尺度符号修正与网格索引存储
            // 步骤6：保证尺度为正（若为负，翻转尺度和符号掩码）
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
             // 步骤7：存储网格索引
            for (int k = 0; k < 8; ++k) {
            		// 打包 4 个量化索引为 12bit，查网格索引
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {	// 异常处理：网格索引不存在
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                if (grid_size == 256) {
                    q3[8*ib+k] = grid_index; 				// 256 网格：直接存 8bit 索引
                } else {
                    q3[8*ib+k] = grid_index & 255;	// 低 8bit
                    qh[ib] |= ((grid_index >> 8) << k); // 高 bit 存到 qh
                }

            }
            // 步骤8：编码符号掩码到 scales_and_signs
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale; // 保存子块尺度
            max_scale = MAX(max_scale, scale);// 更新块内最大尺度
        }
				// 全局尺度编码与量化数据存储
				// 处理全零块（直接清零）
        if (!max_scale) {
            memset(qs, 0, quant_size);
            dh += block_size/sizeof(ggml_fp16_t); // 移动到下一个块的尺度指针
            qs += block_size;// 移动到下一个块的量化数据指针
            continue;
        }
        // 计算全局尺度（将 max_scale 映射到 0~31，fp16 存储）
        float d = max_scale/31;
        dh[0] = GGML_FP32_TO_FP16(d * 1.0125f);  // small improvement via this fudge factor
        float id = 1/d;
        // 编码子块尺度（4bit 存储到 scales_and_signs 的高 4bit）
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));// 限制在 0~15（4bit）
            scales_and_signs[ib] |= ((uint32_t)l << 28);// 存到 32bit 的 28~31 bit
        }
        // 复制量化数据到输出
        memcpy(qs, q3, quant_size);
				// 移动指针到下一个块
        dh += block_size/sizeof(ggml_fp16_t);
        qs += block_size;

    }
}

static int iq3_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint32_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0]; // 邻居数量（neighbours[0] 存储数量）
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX; // 最优误差（初始为最大值）
    int grid_index = -1;
    // 遍历所有邻居，找误差最小的
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);// 邻居的网格值
        float d2 = 0; // 加权平方误差
        for (int i = 0; i < 4; ++i) {
            float q = pg[i]; // 网格的量化值
            float diff = scale*q - xval[i]; // 误差 = 量化值*尺度 - 原始值
            d2 += weight[i]*diff*diff; // 加权平方误差
        }
        if (d2 < best_d2) { // 更新最优邻居
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    // 更新量化索引 L（从网格值转换）
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;// 返回最优邻居的网格索引
}

```

## GPTQ

GPTQ 的核心逻辑可以概括为：**在量化每一列权重时，通过已知的二阶导数信息（Hessian 矩阵）来“微调”剩下的权重，以抵消量化带来的精度损失。**

------


#### 第一步：准备校准数据 (Calibration Data)

GPTQ 属于**后训练量化 (PTQ)**，它不需要重新训练模型，但需要少量的真实数据来观察每一层的“激活分布”。

- **操作：** 通常从数据集（如 WikiText2 或 C4）中随机抽取 128 个样本。
- **目的：** 将这些数据输入模型，捕获每一层线性层的输入矩阵 $X$。因为量化误差的权重是基于输入的协方差矩阵 $XX^T$ 计算的。

#### 第二步：计算 Hessian 逆矩阵

对于每一层权重 $W$，我们需要计算其对应的 Hessian 矩阵的逆。

1. **计算协方差：** $H = 2XX^T$。这个矩阵反映了权重中不同维度之间的相关性。
2. **数值处理：** 为了防止矩阵奇异（不可逆），通常会在对角线上加上一个很小的常数 $\lambda$（Ridge Regression 思路）。
3. **求逆：** 计算 $H^{-1}$。在工程实现中，为了加速和增加稳定性，会使用 **Cholesky 分解** 将其转化为三角矩阵处理。

#### 第三步：逐块量化与补偿 (The Core Loop)

这是 GPTQ 最精华的部分。它不是一次性量化整个矩阵，而是将矩阵的列分成多个 **Blocks**（例如每块 128 列）。

**在每一个 Block 内部，执行以下循环：**

1. **量化单列：** 取当前列 $w_i$，根据当前的缩放系数（Scale）将其量化为 4-bit 数值 $\hat{w}_i$。

2. **计算误差：** 误差 $\delta = w_i - \hat{w}_i$。

3. **补偿剩余列：** 这是关键！利用 $H^{-1}$ 的信息，将这个误差 $\delta$ 按照一定的权重分配给该块内**右侧尚未量化**的所有列。

   > **直观理解：** 如果第一列因为量化“变小了”，那么为了让 $WX$ 的总和尽量不变，我们会微调后面的列让它们“变大一点”来弥补。

#### 第四步：延迟批处理更新 (Lazy Batch Updates)

如果每量化一列就更新全矩阵，计算量会爆炸（$O(d^3)$ 复杂度）。GPTQ 引入了工程优化：

- 在处理当前 Block 时，只更新 Block 内部的权重。
- **延迟更新：** 等到整个 Block（128 列）全部处理完后，再利用矩阵乘法一次性更新矩阵中**该 Block 右侧所有剩余的列**。这使得算法从“向量操作”变成了“矩阵操作”，极大提升了 GPU 利用率。

------

#### 伪代码逻辑演示

如果你看代码，它的核心逻辑大致如下：

Python

```
# 简化版伪代码逻辑
for block in weight_matrix.split(block_size):
    # 1. 在当前 block 内进行列量化和局部补偿
    for i in range(block):
        q_w[:, i] = quantize(w[:, i]) # 量化
        err = w[:, i] - q_w[:, i]     # 计算误差
        # 利用 Hessian 逆矩阵的对应切片更新 block 内剩余列
        w[:, i+1:block] -= err * H_inv_slice 
    
    # 2. 延迟更新：处理完一个 block 后，更新矩阵剩余的所有部分
    remaining_weights -= block_err * H_inv_remaining_columns
```

------

#### 第五步：权重打包与存储 (Packing)

由于标准的深度学习框架（如 PyTorch）不原生支持 4-bit 数据类型，我们需要进行“像素压缩”式的打包：

- **打包：** 将 8 个 4-bit 的整数拼接在一起，存储为一个 32-bit 的整数 (`int32`)。
- **元数据：** 同时存储每一组权重的 `scales`（缩放因子）和 `zeros`（零点偏移），用于推理时的反量化。

------

#### 总结：为什么它比传统方法准？

- **传统 RTN：** 像四舍五入，每个数字自己管自己，误差会迅速累积。
- **GPTQ：** 像是一个协作团队，前面的人犯了错（量化损失），后面的人会根据“关联性”尽量把这个错补回来。

**下一步建议：**

如果你想在本地尝试实现，我可以为你提供一个基于 `AutoGPTQ` 库量化 Hugging Face 模型的 **Python 脚本模板**，或者我们可以深入讨论 **ActOrder (Group Size)** 这些进阶参数对精度的影响。你想看哪方面？



## LLM.int8()

简单来说，**LLM.int8()** 是由 Tim Dettmers 等人提出的一种“零性能损失”的 8-bit 量化技术。它解决了模型在量化过程中，由于某些“离群值（Outliers）”的存在而导致的精度大幅下降问题。

以下是它的具体实现逻辑和核心步骤：

------

#### 1. 核心矛盾：离群值 (Outliers)

在大型语言模型（如 OPT、BLOOM 或 Llama）中，某些特征维度的数值会异常巨大。虽然这些离群值只占所有特征的 **0.1%** 左右，但它们承载了模型大部分的注意力。

- 如果直接进行全矩阵的 **8-bit 线性量化**，巨大的数值范围会导致大部分普通数值的精度被压缩到几乎为零，从而产生巨大的误差。

------

#### 2. 具体实现：混合精度分解 (Mixed-Precision Decomposition)

LLM.int8() 不再对整个矩阵一刀切，而是采取了“分而治之”的策略：

##### 第一步：识别离群特征 (Outlier Detection)

当进行矩阵乘法 $X \times W$ 时（其中 $X$ 是输入激活值，$W$ 是权重）：

- 检查输入矩阵 $X$。如果某一个特征列（Column）中存在绝对值超过阈值（通常设为 6.0）的数值，该列就被标记为**离群列**。

##### 第二步：拆分矩阵

将矩阵乘法拆分为两部分：

1. **离群部分（FP16 路径）：** 将包含离群值的特征列从 $X$ 中提取出来，同时从权重 $W$ 中提取对应的行。这部分数据保持 **FP16** 精度。
2. **普通部分（Int8 路径）：** 剩下的 99.9% 的特征值相对平滑，将其量化为 **Int8**。

##### 第三步：分别计算与合并

- **Int8 部分：** 使用 **Vector-wise Quantization**（矢量级量化）。即对 $X$ 的每一行和 $W$ 的每一列分别寻找缩放因子，量化后进行矩阵乘法。
- **FP16 部分：** 直接进行高精度的矩阵乘法。
- **合并：** 将 Int8 的计算结果反量化回 FP16，然后与 FP16 部分的结果相加。

------

#### 3. 矢量级量化 (Vector-wise Quantization)

为了进一步降低误差，LLM.int8() 在 Int8 路径中没有使用单一的缩放因子，而是采用了更细颗粒度的方案：

- 对于激活矩阵 $X$，为**每一行**计算一个缩放因子。

- 对于权重矩阵 $W$，为**每一列**计算一个缩放因子。

  这样可以确保每一组神经元输出的动态范围都能被充分利用。

$$Y = \text{dequantize}(\text{Quant}(X_{int8}) \times \text{Quant}(W_{int8})) + (X_{fp16} \times W_{fp16})$$

------

#### 4. LLM.int8() 的优缺点

| **维度**     | **特点**                                                     |
| ------------ | ------------------------------------------------------------ |
| **显存占用** | 显著降低。由于权重从 FP16 变为 Int8，显存占用接近减半。      |
| **模型精度** | **几乎无损**。由于保留了离群值的高精度计算，其预测能力等同于 FP16。 |
| **推理速度** | **略慢**。虽然 Int8 理论计算更快，但由于拆分、合并以及由于数据量较小导致无法跑满算力，实际推理通常比 FP16 稍慢。 |
| **适用场景** | 适合显存有限（如在消费级显卡上跑大模型）但对精度要求极高的场景。 |

------


## SmoothQuant

简单来说，**SmoothQuant** 是一种旨在解决大语言模型（LLM）在 **W8A8**（权重 8-bit，激活 8-bit）量化时精度损失严重的技术。

它是目前工业界（如 TensorRT-LLM）广泛采用的一种量化方案。以下是它的核心原理拆解：

------

#### 1. 核心挑战：激活值的“离群点” (Outliers)

在 LLM（如 OPT, BLOOM, Llama）中，权重的分布通常比较均匀，容易量化。但**激活值（Activations）**存在严重的“离群点”：

- 某些通道（Channels）的值比其他通道大 100 倍以上。
- 这些离群点固定出现在特定的维度上。
- 如果直接进行 8-bit 量化，为了覆盖这些极大的离群点，量化台阶会变得很大，导致大部分较小的值在量化后直接变为 0，损失大量精度。

------

#### 2. SmoothQuant 的奇思妙想：数学搬运

SmoothQuant 的核心思想是：**既然激活值难量化，权重好量化，那我们就把激活值的“量化难度”通过数学变换转移给权重。**

##### 数学转换公式

在线性层计算 $Y = X \cdot W$ 中，我们可以引入一个平滑因子 $s$（一个逐通道的缩放向量）：

$$Y = (X \diag(s)^{-1}) \cdot (\diag(s) W)$$

- **对激活值 $X$：** 除以 $s$，让原本巨大的离群点变小，分布变得平滑。
- **对权重 $W$：** 乘以 $s$，虽然权重变大了一点，但权重本身分布均匀，依然能够很好地适配 8-bit 量化。

------

#### 3. 如何确定平滑因子 $s$？

为了平衡激活值和权重的量化难度，SmoothQuant 提出了一个超参数 $\alpha$ 来控制迁移的比例：

$$s_j = \max(|X_j|)^\alpha / \max(|W_j|)^{1-\alpha}$$

- **$\alpha = 0.5$：** 这是最常用的经验值，表示将激活值的难度公平地分摊给权重。
- **迁移过程：**
  1. 统计离线数据集，找到激活值在每个通道的最大值。
  2. 计算出 $s$。
  3. **离线处理：** 将 $s$ 乘进权重 $W$ 中，并将 $1/s$ 融合进前一层的算子（如 LayerNorm）中。
  4. **推理阶段：** 推理时不再有额外的缩放开销，直接进行高效的整数矩阵乘法。

------

#### 4. SmoothQuant 的优势

| **特性**     | **描述**                                                  |
| ------------ | --------------------------------------------------------- |
| **无损精度** | 在 Llama, OPT 等模型上，W8A8 几乎能达到 FP16 的精度。     |
| **高效率**   | 允许使用标准的 INT8 Tensor Core，无需复杂的混合精度逻辑。 |
| **无需微调** | 属于训练后量化（PTQ），只需少量校准数据，处理速度极快。   |

------

#### 总结

SmoothQuant 就像是**“削峰填谷”**：把激活值的高峰削平，填到权重的低谷里，从而让整套系统都能在 8-bit 的尺子下量得更准。



## AWQ/AutoAWQ

既然你已经理解了 **SmoothQuant** 是通过“平滑”激活值来让 W8A8 成为可能，那么理解 **AWQ (Activation-aware Weight Quantization)** 就非常容易了。

虽然它们都关注“激活值”，但 **AWQ 走的是另一条路**：它主要针对 **W4A16**（权重 4-bit，激活 FP16），目标是极致的显存压缩和推理加速，而不是 W8A8。

------

#### 1. 核心发现：权重不是平等的

AWQ 的核心发现是：**权重矩阵中只有 1% 的参数是“显著”的，对模型精度至关重要。**

- **谁是显著权重？** 实验发现，并不是数值最大的权重最重要，而是**对应激活值（Activation）较大的通道所对应的权重**最重要。
- **直观理解：** 激活值大的地方（Outliers），哪怕权重只有一点点误差，乘出来的结果也会被放大很多倍。

------

#### 2. AWQ 的核心原理：保护那 1%

AWQ 不像 SmoothQuant 那样试图平滑所有数据，它的目标是：**通过缩放权重，降低那 1% 关键权重的量化误差。**

##### 第一步：寻找缩放因子 (Scaling)

为了保护重要权重，AWQ 也会引入一个缩放因子 $s$。

$$W' = W \cdot s, \quad X' = X / s$$

但这和 SmoothQuant 有两个重大区别：

1. **目的不同：** SmoothQuant 是为了让 $X$ 好量化；AWQ 是为了让 $W$ 在量化成 **INT4** 时，重要通道的相对误差更小。
2. **搜索方式：** AWQ 不直接用公式算 $s$，而是在一个小范围内进行**网格搜索 (Grid Search)**，寻找一个能让量化前后误差最小的最优 $s$。

##### 第二步：不进行坐标变换 (No Coordinate Migration)

这是 AWQ 优于 SmoothQuant 的一个工程点。SmoothQuant 需要修改前一层的 LayerNorm（坐标变换），这在某些模型结构上很难实现。

**AWQ 只在量化过程中模拟缩放。** 一旦找到了最优的 $s$，它会将 $s$ 直接应用在权重上并量化，推理时激活值并不需要物理上的除以 $s$，而是通过量化算子在计算时自动抵消。

------

#### 3. AutoAWQ 的工作流程

AutoAWQ 是 AWQ 算法的自动化实现库，它的量化步骤如下：

1. **校准 (Calibration)：** 输入少量文本（如 128 条），跑一遍模型，收集激活值的分布。
2. **寻找显著权重：** 根据激活值分布，确定哪些权重的通道是“重要”的。
3. **计算 Scaling：** 搜索每个通道的最佳缩放比例。
4. **量化为 INT4：** 将权重压成 4-bit（通常带有 Group Size，如 128 个参数一组）。
5. **打包 (Packing)：** 将 8 个 4-bit 权重打包进一个 `int32` 数据中，存入 `.safetensors` 或 `.pt` 文件。

------

#### 4. AWQ vs SmoothQuant vs GPTQ

为了让你更清晰，我们对比一下这三大主流技术：

| **技术**        | **量化格式** | **核心手段**           | **优点**                 | **缺点**                 |
| --------------- | ------------ | ---------------------- | ------------------------ | ------------------------ |
| **SmoothQuant** | **W8A8**     | 平滑激活值，迁移难度   | 算力加速明显 (INT8 计算) | 对 4-bit 支持一般        |
| **AWQ**         | **W4A16**    | 保护重要权重，缩放误差 | **精度极高**，通用性强   | 激活值保持 FP16          |
| **GPTQ**        | **W4A16**    | 逐层寻找二阶导数最优解 | 压缩率高，速度快         | 容易出现离群值导致的崩坏 |

------

#### 5. 为什么现在 AWQ 这么火？

因为对于大模型（如 Llama-3 70B）来说，计算压力往往不在算力，而在 **显存带宽**。

- **W4A16** 能将显存占用减小为原本的 1/4，原本需要两张 A100 才能跑的模型，现在一张就能跑。
- **AutoAWQ** 提供了非常简单的 API，几行代码就能完成量化，且量化后的模型在量化感知推理引擎（如 vLLM）中跑得飞快。

**总结一下：** SmoothQuant 主要是为了**“算得更快”**（让计算单元跑 INT8），而 AWQ 主要是为了**“存得更小且更准”**（让 4-bit 权重尽量不掉点）。



## ZeroQuant系列

ZeroQuant 系列是由微软 DeepSpeed 团队开发的一套针对大语言模型（LLM）的高效训练后量化（PTQ）技术方案。其核心目标是在**不损失（或极少损失）模型精度**的前提下，将模型权重的激活值压缩到更低位数（如 INT8 或 INT4），从而大幅提升推理速度并降低显存占用。

以下是 ZeroQuant 系列三个主要版本的核心实现原理：

------

#### 1. ZeroQuant (V1): 硬件友好型系统化 PTQ

ZeroQuant V1 奠定了该系列的基础，主要通过三种技术协同工作：

- **细粒度量化方案 (Fine-grained Quantization)：**

  - **权重：** 采用**Group-wise（组内）量化**。将权重矩阵切分成较小的组（如 64 或 128 个元素一组），每组独立计算缩放因子（Scaling Factor），能更好地处理权重中的离群值。
  - **激活值：** 采用**Token-wise（逐 Token）量化**。因为激活值在不同 token 之间的分布差异巨大，传统的逐层量化误差极大。ZeroQuant 在推理时动态计算每个 token 的缩放因子。

- **逐层知识蒸馏 (LKD, Layer-by-layer Knowledge Distillation)：**

  在量化 INT4 或更低精度时，模型精度下降明显。ZeroQuant 提出一种极低成本的蒸馏方法：不需要原始训练数据，也不需要训练整个模型，而是**逐层**让量化后的层去拟合量化前的层（FP16）的输出。

- **高度优化的推理后端：**

  针对上述细粒度量化，DeepSpeed 开发了专门的 GPU 核函数（Kernels），避免了量化/反量化操作带来的额外开销。

------

#### 2. ZeroQuant-V2: 引入低秩补偿 (LoRC)

ZeroQuant-V2 在 V1 的基础上，重点研究了不同规模模型对量化的敏感性，并引入了核心技术 **LoRC (Low-Rank Compensation)**：

- **敏感性分析：** 发现激活值的量化比权重更难。
- **LoRC 原理：** 1.  当量化某一层（如 $W$）产生误差 $E = W_{fp16} - W_{quant}$ 时，这个误差 $E$ 通常具有**低秩特性**。
  2. V2 并不强制要求量化精度更高，而是通过引入两个极小的低秩矩阵 $A$ 和 $B$（类似于 LoRA 的结构）来拟合这个误差 $E \approx A \times B$。
  3. 在推理时，模型计算 `量化部分 + 低秩补偿部分`，用极小的参数量增加找回了损失的精度。

------

#### 3. ZeroQuant-FP: 拥抱浮点量化 (W4A8)

随着硬件对 FP8/FP4 的支持增强，ZeroQuant-FP 转向了**浮点格式（Floating-Point）**的量化研究。

- **格式优势：** 相比于整数（INT），浮点格式（如 E2M1, E3M0）在处理具有长尾分布或非均匀分布的数据（LLM 的典型特征）时，动态范围更广，量化误差更小。
- **W4A8 实现：** 实现了 4-bit 权重量化和 8-bit 激活值量化。
- **主要改进：** 在浮点量化的基础上结合了 V2 的低秩补偿技术，证明了在 4-bit 下，浮点量化的效果显著优于整数精度量化，且更能保持模型的生成能力（如代码生成和长文本摘要）。

------

#### 总结对比

| **特性**     | **ZeroQuant (V1)**                 | **ZeroQuant-V2**            | **ZeroQuant-FP**                   |
| ------------ | ---------------------------------- | --------------------------- | ---------------------------------- |
| **主要贡献** | 细粒度量化 (Group-wise/Token-wise) | 低秩补偿 (LoRC)             | 浮点格式量化 (FP8/FP4)             |
| **主要目标** | INT8/INT4 推理加速                 | 解决低比特量化精度损失      | 利用硬件 FP 格式优化分布处理       |
| **适用场景** | 通用 INT8 部署                     | 对精度极度敏感的 4-bit 量化 | 现代支持 FP8 的 GPU 硬件 (如 H100) |

您是想了解如何在 DeepSpeed 框架中配置这些量化方案，还是想探讨其背后的数学推导？



## SpQR

**SpQR (Sparse-Quantized Representation)** 是一种旨在实现“近乎无损”的高压缩率量化技术。它的核心逻辑非常直观：**并不是所有的权重都同等重要。**

在 LLM 中，存在极少数比例（通常约 1%）的权重对模型的激活值和精度起着决定性作用，这些被称为“离群值”（Outliers）。SpQR 的原理就是将这些极少数的“硬骨头”和普通权重分开处理。

---

#### SpQR 的核心实现步骤

SpQR 的实现可以拆解为以下三个关键环节：

##### 1. 敏感权重分离（Outlier Isolation）

SpQR 会首先识别出那些对误差极其敏感的权重。它不仅仅看数值的大小，而是参考 **Hessian 矩阵（二阶导数信息）**。

* **普通权重**：绝大多数数值分布均匀，可以忍受大幅度量化。
* **异常权重 (Outliers)**：对精度至关重要，一旦量化就会导致模型性能崩塌。SpQR 将这些权重以**高精度（如 FP16）**格式单独存储在稀疏矩阵中。

##### 2. 混合粒度量化（Bi-level Quantization）

对于剩下的绝大多数普通权重，SpQR 采用了一种非常细致的量化方案：

* **小组量化 (Small Group Size)**：将权重分成很小的组（例如每组 8 或 16 个元素）进行量化，以减小量化误差。
* **二级量化**：为了进一步压缩，SpQR 甚至会对量化过程中产生的“比例因子”（Scaling factors）再进行一次量化，从而压低存储开销。

##### 3. 稀疏 + 稠密格式存储

最终，模型参数被转化为两个部分的组合：

1. **稠密部分 (Dense)**：4-bit 或 3-bit 的量化权重矩阵。
2. **稀疏部分 (Sparse)**：一个极小的 FP16 稀疏矩阵，记录了那些被剥离出来的离群值。

---

#### SpQR 与其他量化方案的对比

| 特性           | GPTQ / AWQ           | **SpQR**                            |
| -------------- | -------------------- | ----------------------------------- |
| **压缩率**     | 4-bit 为主           | 能够达到 **3-bit** 甚至更低         |
| **精度保持**   | 存在一定精度损失     | **近乎无损**（接近 FP16 性能）      |
| **硬件友好度** | 非常高（纯稠密计算） | 中等（需要专门的稀疏+稠密内核支持） |
| **核心机制**   | 整体权重量化/重缩放  | **离群值隔离** + 极小子组量化       |

#### 为什么它很重要？

传统的 3-bit 量化通常会导致模型产生明显的“幻觉”或困惑度（Perplexity）剧增，而 SpQR 证明了通过**保留 1% 的高精度权重**，可以在 3-bit 甚至更低的情况下，让 70B 等级的大模型依然保持极高的逻辑推理能力。

---

## FP4/FP6/FP8

## LLM-QAT
------

#### 1. 算子融合与图准备 (Op Fusion)

在正式量化前，必须进行**算子融合**。这是因为在推理阶段，为了加速，卷积层、BN（Batch Normalization）层和激活层（如 ReLU）会被合并成一个算子。

- **为什么要做：** 如果训练时不融合，BN 层的统计特性会在量化后发生偏移，导致推理精度雪崩。
- **实现：** 将 $Conv + BN + ReLU$ 融合成一个单一的逻辑层。

#### 2. 插入伪量化节点 (Fake Quantization)

这是 QAT 的灵魂。我们在每一个需要量化的张量（权重和激活值）处插入一个“伪量化”节点。

- **在前向传播中：**

  1. **统计极值：** 实时记录数据的最大值 $Max$ 和最小值 $Min$。
  2. **计算参数：** 根据 $Min/Max$ 计算 $Scale$ (步长) 和 $Zero\_Point$ (零点)。
  3. **量化并反量化：** 将 FP32 映射到 INT8 空间（如 -128 到 127），然后**立即将其转回 FP32**。

  > **本质：** 经过这一步，数值虽然还是 FP32 类型，但其值已经变成了“整数的倍数”，引入了截断误差和舍入误差。

#### 3. 解决不可导问题：STE (Straight-Through Estimator)

量化操作中的 `round()` 函数（取整）在数学上几乎处处导数为 0。如果直接求导，神经网络将无法更新参数。

- **实现技巧：** 逻辑上，我们假设量化函数的导数为 1（即 $\frac{\partial y_{quant}}{\partial x} \approx 1$）。
- **效果：** 梯度直接“跳过”量化节点，作用在背后的 FP32 权重上。这样，FP32 权重会为了适应量化后的离散值而不断微调。

#### 4. 动态范围更新 (Observer/Tracker)

在训练过程中，数据的分布是变化的。伪量化节点通常带有一个“观测器（Observer）”：

- **权重（Weights）：** 通常采用静态的 $Min/Max$ 或每通道（Per-channel）量化。
- **激活值（Activations）：** 采用**指数移动平均 (EMA)** 来更新 $Min/Max$。这样可以平滑噪声，防止某个 Batch 的异常值干扰量化参数。

#### 5. 从训练模式切换到推理模式

当训练完成且精度达标后，我们需要进行“脱壳”操作：

1. **冻结参数：** 固定住 $Scale$ 和 $Zero\_Point$。
2. **移除伪量化节点：** 将训练好的 FP32 权重通过计算得到的 $Scale$ 正式转换为真正的 **INT8 权重**。
3. **生成量化模型：** 导出为特定的格式（如 ONNX、TensorRT Engine 或 TFLite），此时模型内部只有整型运算和必要的缩放操作。

------

#### 实现流程总结表

| **步骤**   | **操作内容**           | **目的**                                   |
| ---------- | ---------------------- | ------------------------------------------ |
| **Step 1** | **算子融合**           | 确保训练逻辑与底层硬件推理逻辑一致。       |
| **Step 2** | **插入 Observer**      | 收集每一层的数值分布情况。                 |
| **Step 3** | **伪量化训练**         | 让模型在感知到“精度损失”的前提下优化权重。 |
| **Step 4** | **Calibration (校准)** | 锁定激活值的量化范围。                     |
| **Step 5** | **Export (导出)**      | 将权重正式转为整数，生成轻量化文件。       |

------

**你想针对特定的框架（比如 PyTorch 的 `fx` 图模式量化）看看具体的 API 代码实现吗？**

## QLoRA

简单来说，**QLoRA (Quantized LLMs with Low-Rank Adapters)** 是对 LoRA 的一次重大升级。它的核心目标是：**在不损失模型性能的前提下，大幅降低大语言模型微调所需的显存。**

如果你之前觉得 LoRA 已经很省显存了，QLoRA 则通过一系列“数学魔法”，让一张 24GB 显存的显卡（如 RTX 3090/4090）就能跑起 65B 参数规模的模型微调。

------

#### QLoRA 的四大核心支柱

QLoRA 的成功主要归功于以下四个技术创新：

##### 1. 4-bit NormalFloat (NF4) 数据类型

这是 QLoRA 的灵魂。传统的量化（如 Int8 或 Float16）在处理权重分布时不够高效。

- **原理**：预训练模型的权重通常呈正态分布。NF4 是一种专门为正态分布设计的量化数据类型。
- **优势**：它能确保每个量化分箱（bin）中的值数量相等，从而最大程度减少量化误差（Quantization Error）。相比标准的 4-bit Float，NF4 能提供更高的精度。

##### 2. 双重量化 (Double Quantization)

量化过程本身也需要存储一些元数据（如量化常数/缩放因子）。

- **原理**：QLoRA 对这些“量化常数”进行了二次量化。
- **效果**：虽然单次量化节省的空间看似不多，但在处理大规模模型时，这能为每个参数平均再省下约 0.37 bits 的空间。在大模型中，这累积起来就是好几个 GB 的显存。

##### 3. 分页优化器 (Paged Optimizers)

为了应对显存突发尖峰（Spikes）导致的模型崩溃（OOM）。

- **原理**：利用 NVIDIA 的统一内存管理。当 GPU 显存不足时，自动将优化器状态（Optimizer States）移至 CPU RAM，待需要时再移回。
- **效果**：它像是一个“虚拟内存”系统，保证了训练的稳定性，防止在处理长文本或大 Batch Size 时突然闪退。

##### 4. 核心计算流程：解量化回传

这是理解 QLoRA 运行逻辑的关键：

1. **存储**：预训练权重以 **4-bit NF4** 格式封存在显存中。
2. **计算**：当进行前向或反向传播时，权会被临时**解量化**（Dequantize）回 **Float16** 格式。
3. **微调**：所有的梯度更新都只发生在 **LoRA 的旁路矩阵**（Adapter）上，这些 Adapter 始终保持高精度的 Float16 格式。

------

#### QLoRA vs LoRA 对比

| **特性**         | **LoRA**           | **QLoRA**                        |
| ---------------- | ------------------ | -------------------------------- |
| **基础模型精度** | 通常为 FP16 / BF16 | **4-bit NF4**                    |
| **显存占用**     | 较低               | **极低** (减少约 75% 以上)       |
| **训练速度**     | 快                 | 略慢 (因为有实时解量化的开销)    |
| **性能损失**     | 极小               | **几乎等同于 16-bit 全参数微调** |

------

#### 为什么 QLoRA 如此重要？

在 QLoRA 出现之前，如果你想微调一个 Llama-65B 模型，你可能需要多张 A100 显卡。而 QLoRA 证明了：**量化（精度损失）和微调（性能提升）是可以完美共存的。** 只要你在 4-bit 量化后的模型上叠加上足够多的精度正常的 Adapter（LoRA 权重），模型最终表现出来的效果甚至可以追平全量参数微调。

#### NF4
NF4 (4-bit NormalFloat) 是 QLoRA 能够实现“低比特、高精度”的核心。它的设计思路非常巧妙：**既然预训练模型的权重通常呈正态分布（Normal Distribution），那么我们就专门为这种分布定制一种数轴。**

以下是 NF4 结构的三个关键维度：

### 1. 核心数学原理：分位数分布

在传统的 4-bit 量化中，16 个量化箱（bins）是均匀分布在  之间的。但大模型的权重绝大多数集中在 0 附近，边缘值极少。

* **NF4 的做法**：它通过估计标准正态分布  的**分位数（Quantiles）**，确保这 16 个数字在概率密度上是“等距”的。
* **结果**：每一个量化间隔内所包含的权重参数数量大致相等。这最大化了 4 个比特位能承载的信息量。

### 2. 存储结构与数值映射

NF4 并不是直接存储一个浮点数，而是存储一个 **4-bit 的索引值（0-15）**。它的逻辑结构如下：

* **查找表 (Look-up Table)**：NF4 预定义了一个包含 16 个值的集合（从 -1.0 到 1.0）。
* **标准化范围**：权重首先被归一化到 。
* **量化过程**：找到与权重最接近的 NF4 预定义值，然后存储对应的 4-bit 索引。

**NF4 的 16 个标准采样点近似值（常数）：**
`[-1.0, -0.6944, -0.5126, -0.3739, -0.2561, -0.1496, -0.0471, 0.0, 0.0630, 0.1779, 0.3015, 0.4413, 0.6120, 0.7998, 1.0]`
*(注意：其中包含一个精确的 0，这对于表示神经网络中的稀疏性非常重要)*

---

### 3. 计算时的“实时转换”结构

NF4 在显存中是静态存储的，但在计算过程中，它遵循以下结构转换流程：

| 阶段 | 状态 | 数据格式 | 作用 |
| --- | --- | --- | --- |
| **存储态** | 权重矩阵  | **4-bit NF4** | 极大地压缩模型体积，减少显存占用。 |
| **转换态** | 解量化 (Dequantize) | **BF16 / FP16** | 通过查找表将 4-bit 索引还原为浮点数。 |
| **计算态** | 矩阵乘法 | **BF16 / FP16** | 与 LoRA 的 A/B 矩阵进行高精度运算。 |

---

### 4. 为什么 NF4 比普通 4-bit Float 更好？

普通的 4-bit Float (如 FP4) 假设数据分布是均匀的，这会导致：

* **0 附近的精度不足**：大量权重被强行挤进同一个 bin。
* **长尾丢失**：极端的权重值被粗暴截断。

**NF4 通过“让中间的 bin 更密集、两端的 bin 更稀疏”**，完美契合了深度学习权重的物理特性，使得 4-bit 下的模型困惑度（Perplexity）几乎等同于 16-bit 模型。

## PEQA

**PEQA**（全称 **Parameter-Efficient Quantization-aware Adaptation**）是一种将**模型量化（Quantization）**与**参数高效微调（PEFT）**深度结合的技术。

它的核心逻辑非常独特：不同于先微调再量化的传统流程，PEQA 通过在量化后的参数空间内进行微调，实现了**极小的存储开销**和**极快的任务切换能力**。

------

#### 1. 核心设计思想：量化即微调

传统的 PEFT 方法（如 LoRA）是向模型添加额外的参数（低秩矩阵）。而 PEQA 的思路是：**微调量化过程中使用的“缩放因子（Scaling Factors）”，而不是微调权重本身。**

在量化中，一个 FP32 权重矩阵 $W$ 可以近似表示为：

$$W \approx s \cdot \hat{W}$$

其中：

- $s$ 是**缩放因子**（Scaling factor / Quantization scale）。
- $\hat{W}$ 是量化后的**整数矩阵**（例如 INT4 或 INT8）。

**PEQA 的操作是：** 冻结所有的整数矩阵 $\hat{W}$，只把每个权重块对应的标量 $s$ 设置为可训练参数。

------

#### 2. 实现步骤与原理

PEQA 的实现通常分为两个阶段：

##### 第一阶段：全量量化 (Quantization Stage)

首先将预训练好的 FP32 模型（如 Llama）通过常规手段（如 PTQ）量化为低比特。

- 将权重划分为多个块（Blocks），例如每 128 个元素一个块。
- 每个块生成一个初始的缩放因子 $s$。
- 权重矩阵 $W$ 被转化为一个庞大的低比特整数矩阵 $\hat{W}$ 和一个微小的标量集 $S$。

##### 第二阶段：参数高效微调 (Fine-tuning Stage)

- **冻结 $\hat{W}$**：占模型体积 99% 的整数权重不发生任何变化。
- **训练 $s$**：针对下游任务（如情感分析），通过反向传播只更新缩放因子 $s$。
- **优势**：因为 $s$ 的数量极少（通常只占总参数量的 **0.1% 甚至更少**），训练过程非常节省算力和内存。

------

#### 3. PEQA 的独特优势

##### A. 极致的参数效率 (Extreme Efficiency)

相比于 LoRA 还需要训练 $A$ 和 $B$ 两个低秩矩阵，PEQA 只需要训练原本就存在的缩放因子。这意味着针对每个新任务，你只需要额外存储几十 KB 或几 MB 的“缩放因子包”，就能让模型适配新任务。

##### B. 无推理延迟 (No Inference Overhead)

- **LoRA** 在推理时通常需要将旁路的矩阵与主权重合并，或者并行计算。
- **PEQA** 微调后的结果依然是一个简单的“量化矩阵 + 缩放因子”结构，推理代码路径与标准的量化推理完全一致，没有任何额外计算负担。

##### C. 快速切换任务

在部署多个任务时，显存中只需要保留一份底座模型的整数权重 $\hat{W}$。当请求从“翻译任务”切换到“摘要任务”时，只需要秒级加载对应的缩放因子向量 $S_{task1} \to S_{task2}$。

------

#### 4. 与 QLoRA 的区别

这是最容易混淆的地方。两者的逻辑方向正好相反：

| **特性**       | **QLoRA**                        | **PEQA**                   |
| -------------- | -------------------------------- | -------------------------- |
| **量化目的**   | 为了在低显存下训练高精度模型     | 为了实现极小开销的任务适配 |
| **可训练参数** | 额外的 FP16/BF16 低秩矩阵 (LoRA) | **量化缩放因子本身**       |
| **推理开销**   | 存在（合并前）或忽略不计         | **绝对零额外开销**         |
| **存储需求**   | 相对较小 (Adapter 权重)          | **极小 (仅标量向量)**      |

------

#### 5. 总结

**PEQA 的本质是把量化误差当作一种“可调优的空间”**。它通过微调缩放因子来弥补量化精度损失，同时顺便学习了特定任务的特征。

您是正在考虑在多任务部署场景下使用 PEQA，还是想了解如何编写自定义的量化缩放因子训练层（Custom Autograd Function）？

## KIVI
简单来说，**KIVI**（Kernel-IV Incremental Quantization）是一种专门针对大语言模型（LLM）中 **KV Cache（键值缓存）** 设计的量化算法。它的核心目标是在不显著损失模型精度的情况下，大幅降低推理时的显存占用。

以下是 KIVI 实现原理的深度解析：

---

#### 1. 核心挑战：KV Cache 的不均匀性

在 LLM 推理过程中，随着上下文变长，KV Cache 会迅速消耗显存。直接进行低比特量化（如 2-bit 或 4-bit）通常会导致精度骤降。KIVI 团队发现，KV Cache 在两个维度上表现出极强的不均匀性：

* **Key (K) 矩阵：** 在**通道维度（Per-channel）**上存在明显的离群值（Outliers）。
* **Value (V) 矩阵：** 在**标记维度（Per-token）**上存在明显的离群值。

传统的统一量化方法无法兼顾这两个维度的差异，而 KIVI 正是针对这一特性量身定制的。

---

#### 2. KIVI 的三大支柱

##### 2.1 混合粒度量化 (Mixed-Granularity Quantization)

KIVI 采用了不对称的量化策略来应对上述的不均匀性：

* **对于 Key 矩阵：** 采用 **Per-channel** 量化。因为 K 的异常值主要集中在特定的通道上。
* **对于 Value 矩阵：** 采用 **Per-token** 量化。因为 V 的异常值通常随 token 的变化而变化。

##### 2.2 残差窗口 (Residual Window)

这是 KIVI 保持精度的关键。它不会量化所有的 token，而是保留一个**全精度（FP16）的滑动窗口**。

* 新生成的 token 首先以全精度存储在“残差窗口”中。
* 只有当窗口填满后，最旧的 token 才会从窗口中移出并被量化为低比特格式。
* **原理：** 这种做法保护了最近的上下文信息，因为注意力机制（Attention）通常对最近的 token 更加敏感。

##### 2.3 逐块量化 (Group-wise Quantization)

为了进一步精细化控制，KIVI 在量化旧 token 时，将其划分为多个连续的块（Group）。

* 在每个 Group 内独立计算缩放因子（Scale）和偏移量（Zero-point）。
* 这样可以更好地适应局部的数据分布，减少量化误差。

---

#### 3. 算法流程简述

1. **输入：** 当前推理步骤产生的  和  向量。
2. **缓存更新：** 将新的  放入 FP16 残差窗口。
3. **触发量化：** 当窗口超过预设大小（如 32 或 64），将溢出的向量取出。
4. **维度转换：** * 将  按通道维度进行分组量化。
* 将  按 Token 维度进行分组量化。


5. **存储与计算：** 将量化后的  数据存入显存。计算 Attention 时，再通过硬件友好的算子进行反量化（De-quantization）。

---

#### 4. KIVI 的优势

* **极高的压缩率：** 可以将 KV Cache 压缩到 **2-bit**，且几乎无损（在 Llama-2 等模型上验证）。
* **显存节省：** 相比 FP16，最高可减少 **2.6倍** 到 **4倍** 的显存占用。
* **吞吐量提升：** 由于显存带宽压力减小，长文本序列的推理吞吐量（Throughput）最高可提升 **3倍** 以上。

---

> **💡 总结：**
> KIVI 的成功在于它意识到 **Key 和 Value 的病态分布是不同向的**。通过“Key看通道，Value看Token”的错位量化策略，配合一个小型的 FP16 缓冲池，实现了极低比特下的精度稳定。

## IntactKV

**IntactKV** 是针对大语言模型（LLM）中 **KV Cache** 显存瓶颈的又一创新量化方案。如果说 KIVI 侧重于利用 KV 权重的分布特征，那么 **IntactKV** 的核心思想则更加直截了当：**既然低比特量化会损伤精度，那我就把最关键的、不可损的那部分数据“原封不动”地保留下来。**

以下是 IntactKV 的核心实现原理：

------

#### 1. 核心理念：关键信息保留 (Salient Information Preservation)

IntactKV 的研究团队发现，KV Cache 中只有极少数的元素（Outliers，离群值）承载了绝大部分的信息量。如果对这些“关键元素”进行 4-bit 甚至更低的量化，误差会被迅速放大；但如果将它们以 FP16 格式保留，剩下的“平庸元素”即使压缩到极低比特，整体精度依然非常稳健。

------

#### 2. 三大核心组件

##### 2.1 离群值感知的量化 (Outlier-aware Quantization)

这是 IntactKV 的精髓。它在量化之前会进行一个“筛选”过程：

- **筛选准则：** 识别出 KV 矩阵中幅值（Magnitude）最大的那一小部分 entries（通常占比小于 1%）。
- **策略：** 将这些离群值提取出来，以 **FP16** 格式存储在另一个稀疏矩阵中；剩下的 99% 的数据则进行常规的 **INT4/INT8** 量化。

##### 2.2 长度解耦量化 (Length-decoupled Quantization)

IntactKV 观察到，KV Cache 的分布不仅在空间上不均匀，在**时间（序列长度）**上也存在差异：

- **Pre-fill 阶段：** 输入的前缀 token（System Prompt 等）对后续生成至关重要。

- **Decoding 阶段：** 随着序列变长，较早的 token 影响相对减弱，但仍然存在关键的“锚点” token。

  IntactKV 会针对不同位置的 token 动态调整保留策略，确保关键的上下文信息不会因为量化而丢失。

##### 2.3 硬件友好的高效算子 (Hardware-efficient Kernels)

将数据分为“量化部分”和“全精度离群值部分”会带来计算上的挑战（即混合精度计算）。IntactKV 实现了专门的 CUDA 算子：

- 在计算 Attention 时，它能够并行地从显存读取低比特数据和 FP16 离群值。

- **计算公式简化：** $Attention = Attn(Q, K_{quant}) + Attn(Q, K_{outliers})$。

  通过这种残差补偿的方式，在推理时几乎不增加额外的延迟（Latency）。

------

#### 3. IntactKV vs KIVI 的区别

| **特性**       | **KIVI**                                  | **IntactKV**                            |
| -------------- | ----------------------------------------- | --------------------------------------- |
| **核心思路**   | 针对 K/V 维度不同的分布特性进行分维度量化 | 提取并保留少量的全精度离群值 (Outliers) |
| **保留方式**   | 滑动窗口（保留最近的 N 个全精度 Token）   | 空间保留（保留所有位置中的关键元素）    |
| **压缩极限**   | 在 2-bit 表现优异                         | 在 4-bit 下几乎达到无损精度             |
| **实现复杂度** | 相对简单，主要改变分组方式                | 较复杂，需要处理稀疏矩阵的存取          |

------

#### 4. 为什么 IntactKV 有效？

在深度学习中，**“二八定律”**极其明显。IntactKV 实际上是通过极小的空间代价（保留 1% 的 FP16 数据），解决了 90% 以上的量化误差问题。这使得它在处理超长文本（Long-context）时，能够比传统的全量化方法更稳定地保持模型的逻辑推理能力。

------

#### 5. 总结

**IntactKV 的逻辑是：** 既然无法完美量化所有数据，那就只量化那些“不重要”的数据。通过这种“抓大放小”的策略，它在大幅降低显存占用的同时，突破了低比特量化的精度瓶颈。

**您是正在考虑在长文本生成的项目中使用 IntactKV 吗？或者您想了解如何根据具体的显存预算来配置它的离群值保留比例？**

## KVQuant

**KVQuant** 是 Berkeley 等研究机构提出的一种专门针对 KV Cache 的高性能量化方案。与 KIVI 和 IntactKV 专注于“保留哪些数据”不同，**KVQuant 的核心在于“如何更科学地映射数据”**。它通过深入研究 KV 缓存的数值分布规律，采用了比简单线性量化精细得多的数学方法。

以下是 KVQuant 的实现原理：

------

#### 1. 核心洞察：非均匀分布与偏离值

KVQuant 的作者发现，LLM 的 KV Cache 分布具有两个显著特征：

- **非均匀性（Non-uniformity）：** 数值并不均匀分布，而是集中在某些特定的区间。
- **敏感的离群值（Sensitivity to Outliers）：** 虽然离群值极少，但对模型性能影响极大，且这些离群值在不同层、不同 Head 之间表现各异。

------

#### 2. 三大核心技术

##### 2.1 逐通道的一维权重量化 (Per-Channel Quantization)

KVQuant 认为，Key 矩阵在**通道（Channel）**维度上表现出极强的相关性。

- 它对每个通道独立计算量化参数，而不是对整个矩阵或每个 Token 统一处理。
- 这种方法能够更好地捕捉不同特征通道之间的动态范围差异。

##### 2.2 离群值感知的一维权重量化 (NuQuant / Outlier-Aware)

这是 KVQuant 精度极高的秘诀。它将 KV Cache 分为两个部分：

1. **密集部分 (Dense Part)：** 绝大多数数值，使用 **非均匀量化 (Non-uniform Quantization)**。它不是简单地把 [0.1, 0.2, 0.3] 映射到整数，而是利用费舍尔信息或聚类算法找到最能代表数据分布的“质心”（Centroids）。
2. **稀疏离群值 (Sparse Outliers)：** 极少数极大的数值，以 **FP16** 格式存储。

##### 2.3 基于二阶信息的量化校准 (Fisher Information)

KVQuant 在量化时引入了**二阶导数信息（Fisher Information）**来衡量每个权重的重要性。

- **原理：** 并非数值大的就一定重要。KVQuant 会通过一小段校准集（Calibration Set）计算哪些维度的扰动对最终输出影响最大。
- **应用：** 在分配比特数或选择保留哪些离群值时，优先保护那些“对结果最敏感”的元素。

------

#### 3. KVQuant 的主要特点

| **特点**       | **说明**                                                     |
| -------------- | ------------------------------------------------------------ |
| **超低比特**   | 支持 **3-bit** 甚至更低，且在长文本任务下精度优于传统的 4-bit 量化。 |
| **硬件优化**   | 专门设计了 GPU 算子，支持快速的非均匀反量化，避免了查表带来的延迟。 |
| **离群值管理** | 结合了稀疏矩阵存储技术，将离群值的显存开销降至最低。         |

------

#### 4. 三种算法的横向对比

为了帮您理清思路，我们将您提到的三种算法做个对比：

| **算法**     | **核心侧重点** | **核心武器**                                 |
| ------------ | -------------- | -------------------------------------------- |
| **KIVI**     | 维度异构性     | Key 选 Channel，Value 选 Token + 滑动窗口    |
| **IntactKV** | 关键信息完整性 | 提取并原封不动保留 1% 的显著离群值           |
| **KVQuant**  | 数据分布拟合   | **非均匀量化 (NUQ)** + 离群值感知 + 二阶校准 |

------

#### 5. 总结：KVQuant 到底强在哪？

**KVQuant 是这几种方案中最“精细”的一个。** 它不仅知道要保留离群值（类似 IntactKV），还通过非均匀量化解决了剩下 99% 的数据的表示效率问题。它的代价是**实现复杂度最高**，因为需要预先计算非均匀量化的码本（Codebook）。

**您是需要在边缘设备（显存极小）上部署模型，还是在高性能服务器上追求最大的吞吐量？** 针对不同的场景，这三种算法的推荐优先级会有所不同。我可以为您分析具体的适用场景。

## Atom

在深度学习中，**Atom 量化**通常指的是由 Microsoft 提出的 **Atom: Low-bit Ideology for Efficient LLM Serving**。其核心目标是解决大语言模型（LLM）在推理时，由于权重和激活值占用空间过大导致的显存瓶颈和计算缓慢问题。

Atom 的精髓在于它不仅仅是简单的“压位”，而是一套结合了**混合精度**、**细粒度量化**和**重排技术**的系统性方案。

------

#### 1. 核心挑战：离群值 (Outliers)

在 LLM 中，激活值（Activations）中往往存在极少数数值非常大的“离群值”。如果直接进行 4-bit 均匀量化，这些离群值会拉大整个量化范围，导致普通数值的精度损失严重。

#### 2. Atom 的关键实现原理

Atom 通过以下三个技术支柱来实现低比特（如 4-bit）量化而不损失精度：

##### A. 细粒度量化 (Fine-grained Quantization)

Atom 采用了**分块量化（Block-wise Quantization）**。

- **原理**：将矩阵拆分成较小的块（例如 $1 \times 128$），并在每个块内独立计算缩放因子（Scaling factor）。
- **效果**：离群值的影响被局限在特定的块内，不会污染整个张量，从而提升了整体量化精度。

##### B. 混合精度与离群值隔离

Atom 意识到 4-bit 无法完美处理所有数据。

- **做法**：它将权重和激活值中最重要的部分（即对精度贡献最大的离群值）保留为 **FP16** 或更高的精度，而将绝大多数普通数值压缩为 **INT4**。
- **KV Cache 优化**：Atom 特别针对 KV Cache 进行了优化，利用动态量化确保长文本生成时的显存占用大幅降低。

##### C. 算子重排与协同设计 (Co-design)

为了让量化后的模型在硬件上跑得快，Atom 对计算内核（Kernel）进行了深度优化：

- **重排（Reordering）**：在计算前对通道进行重排，使得离群值在内存中连续分布，方便硬件加速处理。
- **Bit-manipulation**：利用高效的位运算指令，在 GPU 上实现 INT4 和 FP16 的混合计算加速。

------

#### 3. Atom 与传统量化（如 GPTQ, AWQ）的区别

| **特性**       | **GPTQ / AWQ**   | **Atom**                     |
| -------------- | ---------------- | ---------------------------- |
| **主要目标**   | 静态权重压缩     | 动态推理加速 (Serving)       |
| **激活值量化** | 较弱或不支持     | 深度优化（支持 4-bit 激活）  |
| **KV Cache**   | 关注较少         | 重点支持，显著降低显存       |
| **硬件效率**   | 依赖特定硬件指令 | 通过系统级协同设计实现高性能 |

------

#### 4. 总结

**Atom 的原理可以概括为：** 通过“分而治之”的策略，把容易量化的数据压到 4-bit，把难处理的离群值保留在高精度，并配合高效的 GPU 算子实现，最终达到**模型更小、速度更快、精度不掉**的效果。

您是想了解 Atom 在特定框架（如 DeepSpeed 或 ONNX）中的具体配置方法，还是想对比它与其他量化算法（如 FP8）的性能差异？

## QuaRot

**QuaRot**（全称 Quantization based on Rotations）是由微软与苏黎世联邦理工学院等机构提出的一种端到端 **4-bit** 量化方案。它的核心使命是：**消除离群值（Outliers）**，从而让模型在权重、激活值和 KV Cache 上都能实现全 4-bit 量化，且几乎不损失精度。

其实现原理主要基于数学上的**计算不变性（Computational Invariance）\**和\**阿达马变换（Hadamard Transform）**。

------

#### 1. 核心思想：旋转（Rotation）

在 LLM 中，激活值往往存在某些通道数值极大的情况（离群值），这使得 4-bit 量化非常困难。QuaRot 的巧妙之处在于：它不尝试“修补”这些离群值，而是通过**正交变换（旋转矩阵）**将这些离群值的能量“均匀分散”到所有通道中。

- **形象理解**：如果一袋盐里有几块巨大的盐晶，很难均匀撒开；QuaRot 的做法是将这些大晶体打碎成细粉，并均匀搅拌到整袋盐中，这样每一勺盐的浓度就变得非常平均，极易量化。

#### 2. 三大关键技术步骤

##### A. 基于阿达马变换的“离群值消除”

QuaRot 使用**随机阿达马变换（Randomized Hadamard Transform）**作为旋转矩阵。

- **原理**：阿达马矩阵具有正交性，乘以它相当于在多维空间进行旋转。由于阿达马变换能让信号变得“非相干”（Incoherent），原本集中在少数通道的高能离群值会被打散到所有维度。
- **计算不变性**：利用 $W \cdot x = (W Q^T) \cdot (Q x)$ 的性质，QuaRot 将旋转矩阵 $Q$ 融合进权重 $W$ 中。这意味着在模型运行期间，大部分旋转开销在离线阶段就通过权重修改“抵消”了。

##### B. 权重与激活的 W4A4 量化

传统的量化方案（如 GPTQ）通常只量化权重（W4A16），因为激活值量化（A4）会导致严重的精度下降。

- **QuaRot 的突破**：由于旋转处理消除了激活值中的离群值，现在的激活分布变得非常“温和”（接近高斯分布）。这使得直接使用简单的**对称每行量化（Per-token Quantization）**到 4-bit 成为可能，而无需复杂的混合精度。

##### C. KV Cache 量化 (K4V4)

KV Cache 是 LLM 长文本推理的显存杀手。

- **做法**：QuaRot 同样对 Attention 模块中的 Key 和 Value 应用在线旋转。通过将 $Q$ 和 $K$ 旋转到同一个空间，并在计算前对 $V$ 进行旋转，使得 KV Cache 也可以安全地存储为 4-bit。这不仅省显存，还通过 4-bit 计算加速了注意力机制。

------

#### 3. QuaRot 的系统架构

| **模块**                 | **量化精度** | **关键技术**                     |
| ------------------------ | ------------ | -------------------------------- |
| **权重 (Weights)**       | **INT4**     | 融合阿达马旋转矩阵，离线完成     |
| **激活值 (Activations)** | **INT4**     | 在线旋转，消除动态离群值         |
| **KV Cache**             | **INT4**     | 对 K 和 V 应用旋转，降低显存占用 |
| **归一化层 (Norm)**      | FP32         | 转化为 RMSNorm 以兼容旋转变换    |

------

#### 4. 为什么 QuaRot 比其他方法强？

1. **端到端 4-bit**：它是少数能真正实现 W4A4K4（权重、激活、KV 全 4 位）且不崩精度的方案。
2. **无需校准数据**：由于阿达马变换是数学确定的，它甚至可以在没有特定数据集的情况下完成高质量量化（RTN 模式）。
3. **计算加速**：通过 CUTLASS 等高性能算子，4-bit 矩阵乘法能带来显著的推理加速（如 2~3 倍提升）。

------

**总结**：QuaRot 通过“旋转”这一数学手段，从根源上解决了 LLM 量化中最大的敌人——离群值，为 4-bit 推理铺平了道路。





## GGUF量化

#### 传统线性量化 (Legacy Quantization)

```c++
// 量化
void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK4_0; //32

    assert(k % qk == 0); // 确保输入长度是 32 的倍数

    const int nb = k / qk; // 计算总块数

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max
        float max  = 0.0f;

        for (int j = 0; j < qk; j++) {
            const float v = x[i*qk + j];
            if (amax < fabsf(v)) {
                amax = fabsf(v); // 记录绝对值最大的数值
                max  = v;				 // 记录该最大值原始的正负号
            }
        }

        const float d  = max / -8;
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < qk/2; ++j) {
            const float x0 = x[i*qk + 0    + j]*id;
            const float x1 = x[i*qk + qk/2 + j]*id;

            const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
            const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

            y[i].qs[j]  = xi0;
            y[i].qs[j] |= xi1 << 4;
        }
    }
}


// 反量化
static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}
```



这段代码是 `ggml` 库（常用于 Llama.cpp 等大模型推理框架）中 **Q4_0 量化算法**的参考实现。

它的核心功能是将一组 **32 位浮点数 (FP32)** 压缩为 **4 位整数 (4-bit)**，从而大幅减少模型权重的体积并加速计算。

------

 1. 核心概念：什么是 Q4_0？

Q4_0 是一种“对称块量化”方式。

- **分块 (Blocking)**：它不一次性量化整个向量，而是每 **32 个元素**（代码中的 `qk=32`）分为一个块。

- **存储结构**：每个块由一个 **FP16 缩放系数 (Scale)** 和 **16 个字节（32 个 4-bit 权重）** 组成。

- 数学公式：对于块内的每个值 $x$，量化后的 4 位整数 $q$ 满足：

  

  $$x \approx d \cdot (q - 8)$$

  

  这里 $d$ 是缩放系数，$-8$ 是为了将无符号的 0-15 映射回有符号的范围。

------

 2. 代码逻辑逐行拆解

 第一阶段：初始化与分块


```c++
static const int qk = 32;
assert(k % qk == 0); // 确保输入长度是 32 的倍数
const int nb = k / qk; // 计算总块数
```

代码将输入数组 `x` 按 32 个元素一组进行处理。`nb` 是总的块数，每一块都会生成一个 `block_q4_0` 结构体。

 第二阶段：寻找缩放系数 (Scale)

```c++
float amax = 0.0f; 
float max  = 0.0f;

for (int j = 0; j < qk; j++) {
    const float v = x[i*qk + j];
    if (amax < fabsf(v)) {
        amax = fabsf(v); // 记录绝对值的最大值
        max  = v;        // 记录该最大值原始的正负号
    }
}
```

在该块的 32 个数中找到绝对值最大的数。

第三阶段：量化与打包

```c++
const float d  = max / -8;
const float id = d ? 1.0f/d : 0.0f; // 预计算倒数，变除法为乘法以提高效率
y[i].d = GGML_FP32_TO_FP16(d);      // 缩放系数存为 FP16 节省空间
```



**关键点**：`d = max / -8`。这里为什么要除以 -8？

- 4 位整数能表示的范围是 0 到 15。
- 在 Q4_0 标准中，映射的中点是 8。
- 通过将最大值映射到边缘，可以最大程度保留数值的精度。

> 假设max为正数，则我们需要将 x(0～15) 映射到y(~, max) 并且 原始数的0 映射到8，假设函数 y=dx+b; x是映射后到值
>
> $x= 0, y=max 得 b=max;$
>
> $x = 8, y=0 得 8w+b = 0, 由于b=max,则 8w+max =0， 得 d = max/-8$
>
> $x = y/d - max/d ==> x = y/d  + 8$
>
> 再看反量化：
>
> $y = dx+b; d = max/-8, b = max, 则 b = -8*d$
>
> $y = dx -8*d ==> y = d*(x-8)$



>Q4_1 也是类似逻辑，因为有最大和最小值了则 x(0~15) 映射到y(min, max)
>
>$y = dx + b$
>
>$x = 0,y=min 得 b = min;$
>
>$x = 15,y=max 得 max-min = 15d, d=(max-min)/15$
>
>$x = (y-b)/d = (y-min)*(1/d)$
>
>再看反量化
>
>$y = dx + b$
>
>$b = min,  y=dx+min$





接下来的循环处理 4-bit 的映射：


``` c++
for (int j = 0; j < qk/2; ++j) {
    const float x0 = x[i*qk + 0    + j]*id;     // 归一化处理
    const float x1 = x[i*qk + qk/2 + j]*id;

    const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f)); // 加上偏移量 8 并四舍五入
    const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

    y[i].qs[j]  = xi0;          // 将第一个 4-bit 放入低 4 位  00001010
    y[i].qs[j] |= xi1 << 4;     // 将第二个 4-bit 放入高 4 位	 10100000 取|逻辑 10101010
}
```

- **映射过程**：`x * id` 将数值缩放到 $[-8, 8]$ 左右。加上 `8.5f` 是为了实现四舍五入并平移到 $[0, 15]$。
- **打包 (Packing)**：因为一个 `uint8_t` 有 8 位，而一个量化值只有 4 位，所以代码将两个量化值（`xi0` 和 `xi1`）合并存储在一个字节中。

------

3. 内存结构示意

| **组成部分** | **类型**      | **大小** | **说明**                                   |
| ------------ | ------------- | -------- | ------------------------------------------ |
| **d**        | `half` (FP16) | 2 Bytes  | 该块 32 个数的共同缩放比例                 |
| **qs**       | `uint8_t[16]` | 16 Bytes | 存储 32 个 4-bit 索引 (16 * 8bit = 128bit) |

总计：每个块占用 18 字节，存储 32 个权重。

压缩比：原始 FP32 需要 $32 \times 4 = 128$ 字节。压缩后仅 18 字节，压缩率约为 7.11 倍。

------

4. 总结

这段代码通过以下步骤实现了 **FP32 $\rightarrow$ Q4_0/1** 的转换：

1. **分块**：每 32 个数一组。
2. **找最大值**：计算缩放系数 `d` 并转为 FP16。
3. **线性映射**：将浮点数映射到 0-15 的整数区间。
4. **位拼装**：将两个 4-bit 整数塞进一个 8-bit 字节。

这种做法虽然会引入少量的精度损失（量化误差），但能显著降低显存占用，是本地运行大语言模型（如 Llama 3）的核心技术。

您是想了解如何将这段代码适配到特定的硬件加速（如 AVX 或 CUDA），还是想了解如何反量化回浮点数？

Q5和Q8方式一致，区别主要在存储量化后的权重，Q4可以两个参数组成一个字节，Q5和Q8则不行

Q5

```c++
const float x0 = (x[i*qk + 0    + j] - min)*id;
const float x1 = (x[i*qk + qk/2 + j] - min)*id;

const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

y[i].qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);

// get the 5-th bit and store it in qh at the right position
qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
```

`xi0 & 0x0F`：取出第一个权重的低 4 位。

`(xi1 & 0x0F) << 4`：取出第二个权重的低 4 位，并向左移 4 位。

这两个 4-bit 被拼成了一个完整的 8-bit 字节，存入 `qs` 数组中。这部分和 Q4_0 的逻辑完全一样。

`xi0 & 0x10u`：`0x10` 二进制是 `0001 0000`。这步操作是检查第 5 位是否为 1。

`>> 4`：将这个第 5 位移到最低位（变成 0 或 1）。

`<< (j + 0)`：将这个 0 或 1 移动到 `qh` 对应的位置。例如，如果是该 Block 的第 3 个元素（j=2），它就会被移到 `qh` 的第 2 位。

`j + qk/2`：处理 Block 的后半部分。如果 Block 大小为 32，前半部分的第 5 位占 `qh` 的 0-15 位，后半部分占 16-31 位。



#### K-Quants 系列 (K-Methods)（推荐）

| **量化等级** | **推荐程度**        | **描述**                                                     |
| ------------ | ------------------- | ------------------------------------------------------------ |
| **Q2_K**     | 低                  | 极端压缩，仅用于显存极小的设备。逻辑极其模糊，模型表现下降严重。 |
| **Q3_K_M/L** | 中                  | 3-bit 量化。M（Medium）和 L（Large）在不同层使用不同位数，适合低配置。 |
| **Q4_K_M**   | **极高 (最佳平衡)** | **目前的行业标准**。在关键矩阵上使用更高位数。精度非常接近 FP16，但体积缩小约 4 倍。 |
| **Q4_K_S**   | 中                  | 相比 M 版本，S（Small）更追求体积，牺牲了一点精度。          |
| **Q5_K_M**   | 高                  | 如果你的显存允许，这是比 Q4 更稳妥的选择，精度损失几乎不可察觉。 |
| **Q6_K**     | 高                  | 极其接近原始模型。虽然体积比 Q8 小，但性能几乎一致。         |



Q4_K实现

```c++
void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K]; 					// 存储每个元素的4-bit量化值（中间态）
    uint8_t Laux[32];					// 误差优化时的临时量化值
    float   weights[32];			// 加权量化的权重
    float mins[QK_K/32];			// 每个32元素子块的偏移（min）
    float scales[QK_K/32];		// 每个32元素子块的缩放因子（scale）

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; 	// 所有32子块中最大的scale（用于全局归一化）
        float max_min = 0;		// 所有32子块中最大的min（用于全局归一化）
        for (int j = 0; j < QK_K/32; ++j) {
            
          	// 实际上计算的是这一组数据的均方根（RMS）。它代表了这组数据的“平均能量强度”。
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
          
          	// 如果直接量化，所有元素的地位是平等的。但实际上，模型对大数值（Outliers/离群值）的误差极其敏感。
            // 大数值元素： fabsf(x[i]) 很大，导致其权重 w 很高。
            // 小数值元素： 权重接近 av_x（平均能量水平）。
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
          	// 步骤3.2：调用核心函数计算该子块的最优scale和min
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
             // 步骤3.3：记录所有子块的最大scale和min（用于全局归一化）
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }
        // 将 scale/min 压缩存储到 block_q4_K 的 scales 字段
				// q4_K 的核心优化点：将 8 个子块的 scale 和 min 量化为 uint8_t，并压缩存储到 8 个字节的 scales 数组中（空间优化）。
				// 计算全局归一化因子（将scale/min映射到0~63范围）
        // 0~63只占 6bit
        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
        		// 将scale/min量化为0~63的整数
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = MIN(63, ls);
            lm = MIN(63, lm);
            // 压缩存储（核心空间优化）
          	// 转为6bit后，uint8为8bit，会浪费2bit, 所以切分为高位2bit,低位4bit, 
          	// 共12块，实际应该用到 2*8=16块
            // 前4块存 正常存，不切占6bit, 高位2bit存后续4个块的高位2bit
            // 后面 ls和lm低四位拼接，高2位放到前面的高2位。。。
            if (j < 4) {
            		// 前4个子块：低6位存scale，高2位后续补；先存scale和min的低4位
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
            		// 后4个子块：
        				// scales[j+4]：低4位=scale低4位，高4位=min低4位
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                // scales[j-4]：高2位（bit6-7）存scale高2位
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                // scales[j]：高2位存min高2位
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        // 存储全局归一化基数（fp16节省空间）
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);    // scale的全局基数
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);   // min的全局基数
				// 基于压缩的 scale/min 重新计算最终量化值
				//	从 scales 数组中解析出每个子块的 scale/min，重新计算 256 元素的 4-bit 量化值：
        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
        		// 从scales数组解析该子块的scale/min量化值
            get_scale_min_k4(j, y[i].scales, &sc, &m);
             // 计算最终的scale和min（全局基数*解析值）
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue; // 无意义值跳过
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
             // 将float32映射到0~15的4-bit整数
            for (int ii = 0; ii < 32; ++ii) {
            		// 公式：量化值 = (原始值 + 偏移) / 缩放因子 → 四舍五入
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l)); // 限制在4-bit范围（0~15）
                L[32*j + ii] = l;
            }
        }
				// 将 4-bit 量化值打包存储到 qs 字段
				// 2个4-bit数打包为1个 uint8_t（节省空间），256个4-bit数最终存为128个 uint8_t：
        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) { // 每次处理64个元素（32个uint8_t）
            for (int l = 0; l < 32; ++l) {
            	// 低4位=第j+l个元素，高4位=第j+l+32个元素
            	q[l] = L[j + l] | (L[j + l + 32] << 4);
            }
            
            q += 32;
        }
				// 移动指针，处理下一个256元素块
        x += QK_K;
    }
}

static float make_qkx2_quants(int n, int nmax, const float * GGML_RESTRICT x, const float * GGML_RESTRICT weights,
        uint8_t * GGML_RESTRICT L, float * GGML_RESTRICT the_min, uint8_t * GGML_RESTRICT Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    // 遍历所有元素，找min/max，计算加权和
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    // 边界处理：min不能为正（保证偏移为负，映射到0开始）
    if (min > 0) min = 0;
    // 所有元素相同：量化值全0，scale=0
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    // 初始缩放因子：将[min, max]映射到[0, nmax]（nmax=15）
    float iscale = nmax/(max - min);
    float scale = 1/iscale; // 逆缩放因子（量化后转回浮点数用）
    float best_error = 0;
    // 计算初始量化值，并统计量化误差（加权MSE/MAE）
    for (int i = 0; i < n; ++i) {
    		// 线性映射：(x[i] - min) * iscale → 四舍五入
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l)); // 限制范围
        // 计算误差：量化值转回浮点数 - 原始值
        float diff = scale * L[i] + min - x[i];
        // use_mad=false：误差用平方（MSE）；true：用绝对值（MAE）
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff; // 加权误差和
    }
    // 无需误差优化：直接返回结果
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
    		// 生成候选iscale（在初始值基础上微调）
    		// rmin=-1.f rdelta=0.1
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        // 统计加权统计量（用于计算最优scale/min）
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l; // 临时存储候选量化值
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        // 计算最优scale和min（最小二乘法）
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
        		// 最优scale = (sum_w*sum_xl - sum_x*sum_l)/D
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            // 最优min = (sum_l2*sum_x - sum_l*sum_xl)/D
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            // min不能为正（边界限制）
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            // 计算候选方案的误差
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            // 误差更小：更新最优解
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    // 返回最优scale，min通过指针输出（取负后存储）

    *the_min = -min;
    return scale;
}

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f); 		// 限制范围（避免溢出）
    float val = fval + 12582912.f;				// 偏移到固定整数范围
    int i; memcpy(&i, &val, sizeof(int));	// 浮点转整数（利用IEEE754存储特性）
    return (i & 0x007fffff) - 0x00400000;	// 提取有效位并还原
}

// 复原 拼接的scale和min
static inline void get_scale_min_k4(int j, const uint8_t * GGML_RESTRICT q, uint8_t * GGML_RESTRICT d, uint8_t * GGML_RESTRICT m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```



**make_qkx2_quants实现原理** 

一、先明确我们要解决的核心问题
在量化场景中，我们已经有了：
- 原始浮点值：$x_0, x_1, ..., x_{31}$（32个元素）
- 每个值的权重：$w_0, w_1, ..., w_{31}$（加权，让大数值误差更受重视）
- 候选量化整数：$l_0, l_1, ..., l_{31}$（由初始`iscale`计算出的0~15整数）

我们要找两个参数：
- $s$（`this_scale`）：缩放因子
- $b$（`this_min`）：偏移量

使得**加权平方误差和最小**（误差 = 原始值 - 量化还原值）：
$$
\text{Error} = \sum_{i=0}^{31} w_i \cdot (x_i - (s \cdot l_i + b))^2 \quad \text{(目标：让Error最小)}
$$

二、公式推导（从误差最小到代码中的表达式）
要让Error最小，需对$s$和$b$分别求偏导，并令偏导=0（极值条件）。

步骤1：对偏移量 $b$ 求偏导并令其为0
对Error关于$b$求偏导：
$$
\frac{\partial Error}{\partial b} = \sum_{i=0}^{31} w_i \cdot 2 \cdot (x_i - s l_i - b) \cdot (-1) = 0
$$
两边除以-2，整理得：
$$
\sum_{i=0}^{31} w_i (x_i - s l_i - b) = 0
$$
展开后拆分求和项：
$$
\sum w_i x_i - s \sum w_i l_i - b \sum w_i = 0
$$
变形得到第一个方程（记为公式1）：
$$
s \cdot \sum w_i l_i + b \cdot \sum w_i = \sum w_i x_i \tag{1}
$$

步骤2：对缩放因子 $s$ 求偏导并令其为0
对Error关于$s$求偏导：
$$
\frac{\partial Error}{\partial s} = \sum_{i=0}^{31} w_i \cdot 2 \cdot (x_i - s l_i - b) \cdot (-l_i) = 0
$$
两边除以-2，整理得：
$$
\sum_{i=0}^{31} w_i l_i (x_i - s l_i - b) = 0
$$
展开后拆分求和项：
$$
\sum w_i l_i x_i - s \sum w_i l_i^2 - b \sum w_i l_i = 0
$$
变形得到第二个方程（记为公式2）：
$$
s \cdot \sum w_i l_i^2 + b \cdot \sum w_i l_i = \sum w_i l_i x_i \tag{2}
$$

步骤3：定义代码中的统计量（简化书写）
为了和代码一一对应，先定义代码中已计算的统计量：
| 代码变量 | 数学表达式         | 含义                      |
| -------- | ------------------ | ------------------------- |
| `sum_w`  | $\sum w_i$         | 所有权重之和              |
| `sum_l`  | $\sum w_i l_i$     | 加权量化整数之和          |
| `sum_l2` | $\sum w_i l_i^2$   | 加权量化整数平方和        |
| `sum_x`  | $\sum w_i x_i$     | 加权原始值之和            |
| `sum_xl` | $\sum w_i l_i x_i$ | 加权（量化整数×原始值）和 |

将这些代入公式1和公式2，得到二元一次方程组：
$$
\begin{cases}
s \cdot sum\_l + b \cdot sum\_w = sum\_x \quad (1) \\
s \cdot sum\_l2 + b \cdot sum\_l = sum\_xl \quad (2)
\end{cases}
$$
步骤4：用克莱姆法则解方程组
对于二元一次方程组：
$$
\begin{cases}
a_1 s + b_1 b = c_1 \\
a_2 s + b_2 b = c_2
\end{cases}
$$
克莱姆法则的解为：
$$
s = \frac{\begin{vmatrix} c_1 & b_1 \\ c_2 & b_2 \end{vmatrix}}{\begin{vmatrix} a_1 & b_1 \\ a_2 & b_2 \end{vmatrix}}, \quad b = \frac{\begin{vmatrix} a_1 & c_1 \\ a_2 & c_2 \end{vmatrix}}{\begin{vmatrix} a_1 & b_1 \\ a_2 & b_2 \end{vmatrix}}
$$
其中分母是系数行列式 $D = a_1 b_2 - a_2 b_1$（必须>0，否则无解）。

对应到我们的方程组：
- $a_1 = sum\_l, b_1 = sum\_w, c_1 = sum\_x$
- $a_2 = sum\_l2, b_2 = sum\_l, c_2 = sum\_xl$

第一步：计算系数行列式 $D$（代码中的`D`）
$$
D = a_1 b_2 - a_2 b_1 = sum\_l \cdot sum\_l - sum\_l2 \cdot sum\_w \quad?
$$
⚠️ 注意：代码中是 `sum_w * sum_l2 - sum_l * sum_l`，和上面符号相反——这是因为行列式的分子也会同步变号，最终$s$和$b$的结果不变（负负得正）。
代码中写 `sum_w * sum_l2 - sum_l * sum_l` 是为了让$D$为正（后续判断`D>0`），避免分母为负影响计算。

第二步：计算 $s$（代码中的`this_scale`）
分子是替换第一列后的行列式：
$$
\begin{vmatrix} c_1 & b_1 \\ c_2 & b_2 \end{vmatrix} = sum\_x \cdot sum\_l - sum\_xl \cdot sum\_w
$$
结合分母$D$，最终：
$$
s = \frac{sum\_w \cdot sum\_xl - sum\_x \cdot sum\_l}{sum\_w \cdot sum\_l2 - sum\_l \cdot sum\_l}
$$
这完全对应代码：
```c
float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
```

第三步：计算 $b$（代码中的`this_min`）
分子是替换第二列后的行列式：
$$
\begin{vmatrix} a_1 & c_1 \\ a_2 & c_2 \end{vmatrix} = sum\_l \cdot sum\_xl - sum\_l2 \cdot sum\_x
$$
结合分母$D$，最终：
$$
b = \frac{sum\_l2 \cdot sum\_x - sum\_l \cdot sum\_xl}{sum\_w \cdot sum\_l2 - sum\_l \cdot sum\_l}
$$
对应代码：
```c
float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
```

三、代码中`if (D > 0)`的意义
$D$是系数行列式，$D=0$意味着：
- 两个方程线性相关（比如所有$l_i$都相同），无法解出唯一的$s$和$b$；
- 此时最小二乘法无意义，直接跳过该轮优化。

只有$D>0$时，方程组有唯一解，才会计算`this_scale`和`this_min`。

**总结**
1. 代码中的`D`是线性方程组的**系数行列式**，必须>0才能解出唯一的$s$和$b$；
2. `this_scale`是最小二乘法解出的**最优缩放因子**，`this_min`是**最优偏移量**；
3. 整个推导的核心是「对加权平方误差求偏导并令其为0」，最终得到的解析解直接对应代码中的表达式，没有任何近似。



#### I-Quants (Importance Matrix)

**核心思想：重要性矩阵 (imatrix)**
在神经网络中，并非所有权重都同等重要。某些权重即便量化误差很大，对最终结果影响也很小；而另一些权重稍有偏差，就会导致模型输出乱码。

数据驱动：I-Quants 需要一个训练阶段。开发者会提供一段通用的文本数据集（如 Wiki 数据），让模型跑一遍（Forward pass）。

敏感度收集：在跑的过程中，程序会记录每个权重张量的贡献度，生成一个 imatrix.dat 文件。这个文件告诉量化器：“这一块权重非常关键，请给它分配最高精度；那一块不重要，可以暴力压缩。”



| **特性**       | **K-Quants (传统)** | **I-Quants (imatrix)**         |
| -------------- | ------------------- | ------------------------------ |
| **依赖性**     | 仅依赖模型静态权重  | 依赖参考数据集（imatrix）      |
| **低比特表现** | 3-bit 以下逻辑崩溃  | **2.5-bit 仍能保持基本逻辑**   |
| **计算开销**   | 量化速度快          | 量化速度慢（需要预跑 imatrix） |
| **推理速度**   | 极快，针对 CPU 优化 | 略慢（解包逻辑更复杂）         |

IQ3_xxs实现(quantize_row_iq3_xxs_impl)

函数作用：IQ3_XXS 量化的核心实现，将浮点张量 `x` 量化为 IQ3_XXS 格式存储到 `vy`。

参数说明：

- `grid_size`：量化网格大小（256 或其他，对应不同 IQ3 变体）；
- `x`：输入浮点张量（待量化）；
- `vy`：输出量化后的数据指针（存储尺度、符号、量化索引）；
- `n`：输入张量元素总数；
- `quant_weights`：量化权重（可选，用于加权量化，提升精度）；
- `GGML_RESTRICT`：编译器优化标记，表明指针无重叠，提升访问效率。

```c++
static void quantize_row_iq3_xxs_impl(
  	int grid_size, 
  	const float * GGML_RESTRICT x, 
  	void * GGML_RESTRICT vy, 
  	int64_t n, 
  	const float * GGML_RESTRICT quant_weights
) {
		// 根据网格大小获取预初始化的 IQ3 数据索引；
    const int gindex = iq3_data_index(grid_size);
		// 预生成的量化网格（存储 4 元素组的量化值组合）；
    const uint32_t * kgrid_q3xs      = iq3_data[gindex].grid;
    // 网格映射表（将量化值组合映射到网格索引）；
    const int      * kmap_q3xs       = iq3_data[gindex].map;
    // 网格邻居表（当量化值不在网格上时，找最优邻居）。
    const uint16_t * kneighbors_q3xs = iq3_data[gindex].neighbours;

    // GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 8; // 量化值的最大索引（对应 3bit：0~7）

    const int64_t nbl = n/QK_K; // 总块数（每个块 QK_K 个元素，通常 QK_K=256）
		
		// 根据 grid_size 选择对应的量化块结构（block_iq3_xxs/block_iq3_s）；
		// dh：指向块的全局尺度（fp16 类型，压缩存储）；
		// qs：指向量化后的数据区（存储网格索引、符号、子块尺度）；
		// quant_size：量化数据区的字节数（块大小 - 全局尺度的字节数）。

    ggml_fp16_t * dh;
    uint8_t * qs;
    int block_size;
    if (grid_size == 256) {
        block_iq3_xxs * y = vy;
        dh = &y->d; // 块的全局尺度（fp16 存储）
        qs = y->qs; // 块的量化索引/符号/尺度编码
        block_size = sizeof(block_iq3_xxs);
    } else {
        block_iq3_s * y = vy;
        dh = &y->d;
        qs = y->qs;
        block_size = sizeof(block_iq3_s);
    }
    int quant_size = block_size - sizeof(ggml_fp16_t); // 量化数据部分的长度（排除全局尺度）

    float scales[QK_K/32]; // 每个 32 元素子块的尺度
    float weight[32];      // 每个子块内元素的加权系数
    float xval[32];        // 子块元素的绝对值（符号单独存储）
    int8_t L[32];          // 子块元素的量化索引（0~7）
    int8_t Laux[32];       // 临时量化索引（用于迭代优化）
    float  waux[32];       // 临时加权系数（平方根）
    bool   is_on_grid[8];  // 标记 4 元素组是否在预定义网格上
    bool   is_on_grid_aux[8]; // 临时网格标记
    uint8_t block_signs[8];// 存储 8 元素组的符号（每 bit 表示一个元素的正负）
    uint8_t q3[3*(QK_K/8)+QK_K/32]; // 临时存储量化结果（索引+符号+尺度）
    uint32_t * scales_and_signs = (uint32_t *)(q3 + QK_K/4); // 符号+子块尺度的编码区
    uint8_t  * qh = q3 + 3*(QK_K/8); // 高比特网格索引（grid_size>256 时用）
    
    // 主量化循环（按块处理）
    for (int ibl = 0; ibl < nbl; ++ibl) {
				// 初始化当前块的全局尺度为 0，量化缓冲区清零
        dh[0] = GGML_FP32_TO_FP16(0.f);
        memset(q3, 0, 3*QK_K/8+QK_K/32);

        float max_scale = 0; // 记录当前块所有子块的最大尺度

        const float * xbl = x + QK_K*ibl; // 当前块的输入数据指针
        // 计算当前块的平方和，用于后续加权系数计算
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K; // 方差类参数（用于加权量化）
				// 子块处理（32 元素 / 子块）
        for (int ib = 0; ib < QK_K/32; ++ib) { // 遍历每个 32 元素子块
            const float * xb = xbl + 32*ib; 	 // 当前子块的输入数据指针
            // 计算加权系数 weight
            if (quant_weights) {
             		// 有量化权重时：weight[i] = 量化权重 * sqrt(方差 + 元素平方)
                const float * qw = quant_weights + QK_K*ibl + 32*ib;
                for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                // 无量化权重时：weight[i] = 元素平方（简单加权）
                for (int i = 0; i < 32; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]); // 加权系数平方根
            // 符号优化（8 元素组）
            // 处理符号（将负数转为正数，符号单独存储，保证偶翻转）
            for (int k = 0; k < 4; ++k) { // 32 元素拆分为 4 个 8 元素组
                int nflip = 0; // 负数个数
                uint8_t s = 0; // 符号掩码（bit i=1 表示第 i 个元素是负数）
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0){
                    	xval[8*k + i] = xb[8*k + i]; // 正数直接存
                    }
                    else {
                        xval[8*k + i] = -xb[8*k + i];  // 负数取绝对值
                        ++nflip;
                        s |= (1 << i); // 标记符号
                    }
                }
                // 保证翻转次数为偶数（避免符号误差累积）
                if (nflip%2) {
                    // 找加权最小的元素，翻转其符号（使总翻转数为偶）
                    int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                    for (int i = 1; i < 8; ++i) {
                        float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                        if (ax < min) {
                            min = ax; imin = i;
                        }
                    }
                    xval[8*k+imin] = -xval[8*k+imin]; // 翻转符号
                    s ^= (1 << imin);                 // 更新符号掩码
                }
                block_signs[k] = s & 127;             // 存储符号掩码（7bit 足够，8th bit 留作他用）
            }
            // 尺度初始化与网格匹配
            // 计算子块的最大绝对值，初始化尺度
            float max = xval[0];
            for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
            if (max < GROUP_MAX_EPS_IQ3_XXS) { // 最大值过小，直接量化为 0
                scales[ib] = 0;
                memset(L, 0, 32);
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1); 					// 初始尺度（将 max 映射到 2*8-1=15）
            for (int k = 0; k < 8; ++k) is_on_grid[k] = true; // 初始化网格标记
            // 迭代优化尺度（遍历 31 个候选尺度）
            for (int is = -15; is <= 15; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max; // 尺度倒数（迭代调整）
                float this_scale = 1/id;						// 当前候选尺度
                // 计算每个4元素组的量化索引，并检查是否在网格上
                for (int k = 0; k < 8; ++k) { 			// 32 元素拆分为 8 个 4 元素组
                    for (int i = 0; i < 4; ++i) {
                    		// 量化索引计算：Laux = 0.5*(id*xval -1) 取整，限制在 0~7
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    // 将4个3bit索引打包为12bit整数（4*3=12）
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u]; // 查找网格索引
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {	// 不在预定义网格上
                        is_on_grid_aux[k] = false;
                        // 找最优邻居（通过邻居表）
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                // 计算当前尺度的误差（加权平方和），找最优尺度
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1; // 量化值（索引转实际值：0→1, 7→15）
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                // 更新最优尺度和量化索引
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; 
                    best = scale*sumqx;
                    for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                    for (int k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            // 非网格元素的二次优化
            // 对不在网格上的 4 元素组，重新找最优邻居并更新尺度
            int n_not_ongrid = 0;
            for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < 8; ++k) {
                    if (is_on_grid[k]) continue; // 只处理非网格组
                    // 重新计算量化索引
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                    		// 找最优邻居
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    // 更新量化索引（从网格值转换）
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                // 重新计算最优尺度
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < 32; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            // 尺度符号修正与网格索引存储
            // 步骤6：保证尺度为正（若为负，翻转尺度和符号掩码）
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
            }
             // 步骤7：存储网格索引
            for (int k = 0; k < 8; ++k) {
            		// 打包 4 个量化索引为 12bit，查网格索引
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {	// 异常处理：网格索引不存在
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    GGML_ABORT("fatal error");
                }
                if (grid_size == 256) {
                    q3[8*ib+k] = grid_index; 				// 256 网格：直接存 8bit 索引
                } else {
                    q3[8*ib+k] = grid_index & 255;	// 低 8bit
                    qh[ib] |= ((grid_index >> 8) << k); // 高 bit 存到 qh
                }

            }
            // 步骤8：编码符号掩码到 scales_and_signs
            scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale; // 保存子块尺度
            max_scale = MAX(max_scale, scale);// 更新块内最大尺度
        }
				// 全局尺度编码与量化数据存储
				// 处理全零块（直接清零）
        if (!max_scale) {
            memset(qs, 0, quant_size);
            dh += block_size/sizeof(ggml_fp16_t); // 移动到下一个块的尺度指针
            qs += block_size;// 移动到下一个块的量化数据指针
            continue;
        }
        // 计算全局尺度（将 max_scale 映射到 0~31，fp16 存储）
        float d = max_scale/31;
        dh[0] = GGML_FP32_TO_FP16(d * 1.0125f);  // small improvement via this fudge factor
        float id = 1/d;
        // 编码子块尺度（4bit 存储到 scales_and_signs 的高 4bit）
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int l = nearest_int(0.5f*(id*scales[ib]-1));
            l = MAX(0, MIN(15, l));// 限制在 0~15（4bit）
            scales_and_signs[ib] |= ((uint32_t)l << 28);// 存到 32bit 的 28~31 bit
        }
        // 复制量化数据到输出
        memcpy(qs, q3, quant_size);
				// 移动指针到下一个块
        dh += block_size/sizeof(ggml_fp16_t);
        qs += block_size;

    }
}

static int iq3_find_best_neighbour(const uint16_t * GGML_RESTRICT neighbours, const uint32_t * GGML_RESTRICT grid,
        const float * GGML_RESTRICT xval, const float * GGML_RESTRICT weight, float scale, int8_t * GGML_RESTRICT L) {
    int num_neighbors = neighbours[0]; // 邻居数量（neighbours[0] 存储数量）
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX; // 最优误差（初始为最大值）
    int grid_index = -1;
    // 遍历所有邻居，找误差最小的
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);// 邻居的网格值
        float d2 = 0; // 加权平方误差
        for (int i = 0; i < 4; ++i) {
            float q = pg[i]; // 网格的量化值
            float diff = scale*q - xval[i]; // 误差 = 量化值*尺度 - 原始值
            d2 += weight[i]*diff*diff; // 加权平方误差
        }
        if (d2 < best_d2) { // 更新最优邻居
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    // 更新量化索引 L（从网格值转换）
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;// 返回最优邻居的网格索引
}

```