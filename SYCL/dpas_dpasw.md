# Intel XMX `DPAS` / `DPASW` 笔记

## 1. 背景：DPAS 是什么

`DPAS` 全称可以理解为：

```text
Dot Product Accumulate Systolic
```

它是 Intel XMX / Matrix Engine 用来做低精度矩阵乘加的核心指令。

数学语义：

```text
D = C + A × B
```

在 Intel vISA 文档里的 operand 对应关系是：

```text
Dst  = D = M × N
Src0 = C = M × N
Src2 = A = M × K
Src1 = B = K × N
```

最容易搞混的一点：

```text
Src2 是 A
Src1 是 B
```

不是反过来。

---

## 2. DPAS 的核心参数

vISA 形式大概是：

```text
DPAS Exec_size Dst Src0 Src1 Src2 W A SD RC
```

你写 SYCL ESIMD 的 `xmx::dpas` 时，不一定直接按这个汇编格式写，但底层概念是一致的。

关键参数：

```text
RC = Repeat Count
SD = Systolic Depth
Exec_size = N 方向的执行宽度
Src1 precision = B 的数据类型
Src2 precision = A 的数据类型
```

矩阵形状：

```text
D / C : M × N
A     : M × K
B     : K × N
```

其中：

```text
M = RC
N = Exec_size，通常是 8 或 16
K = SD × OPS_PER_CHAN
```

`OPS_PER_CHAN` 取决于数据类型：

```text
TF32:         OPS_PER_CHAN = 1
BF16 / FP16: OPS_PER_CHAN = 2
8-bit:       OPS_PER_CHAN = 4
<8-bit:      OPS_PER_CHAN = 8
```

所以如果：

```text
SD = 8
```

那么一次 DPAS 的 K tile 是：

```text
TF32:         K = 8
BF16 / FP16: K = 16
INT8 / FP8:  K = 32
INT4 / INT2: K = 64
```

注意：这里的 K 是“一次 DPAS 指令处理的 Ktile”，不是完整 GEMM 的总 K。

---

## 3. 如何制定 M / N / K 大小

### 3.1 M 怎么来

```text
M = RC = Repeat Count
```

例如你想一次得到：

```text
C: 4 × 8
```

那可以考虑：

```text
RC = 4
Exec_size = 8
```

这时：

```text
Dst/Src0 = 4 × 8
```

### 3.2 N 怎么来

```text
N = Exec_size
```

常见是：

```text
N = 8 或 16
```

可以粗略理解为：

```text
Exec_size=8  -> 一次输出 8 列
Exec_size=16 -> 一次输出 16 列
```

从 SIMD channel 角度看：

```text
channel 0 负责 C 的第 0 列
channel 1 负责 C 的第 1 列
...
channel N-1 负责 C 的第 N-1 列
```

### 3.3 K 怎么来

```text
Ktile = SD × OPS_PER_CHAN
```

对 FP16/BF16：

```text
OPS_PER_CHAN = 2
```

所以常见：

```text
SD = 8
Ktile = 16
```

这和 CUDA 里常见的：

```text
mma ... k16
```

比较像。

如果完整 GEMM 的 K 是 128，不是把 `SD` 设置成 64，而是循环多个 DPAS：

```cpp
for (int kk = 0; kk < 128; kk += 16) {
    // A tile: M × 16
    // B tile: 16 × N
    // C += A × B
    acc = dpas(acc, a_tile, b_tile);
}
```

也就是：

```text
总 K = 128
每次 Ktile = 16
循环 8 次
```

### 3.4 一个 FP16 例子

想算：

```text
A: 4 × 16
B: 16 × 8
C: 4 × 8
```

可以设计为：

```text
RC = 4
Exec_size = 8
SD = 8
FP16/BF16 => OPS_PER_CHAN = 2
Ktile = 8 × 2 = 16
```

也就是一次 DPAS 完成一个：

```text
4 × 8 输出 tile
```

如果是：

```text
A: 4 × 128
B: 128 × 8
C: 4 × 8
```

那就是：

```text
RC = 4
Exec_size = 8
SD = 8
Ktile = 16
循环 8 次
```

---

## 4. DPAS layout 总结

DPAS 最容易绕的地方是 operand layout。

先记住：

```text
Dst/Src0/Src2 比较正常
Src1 最特殊
```

也就是：

```text
Dst  = D/C 输出 accumulator，row-major
Src0 = C 输入 accumulator，row-major
Src2 = A，row-major
Src1 = B，特殊 packed layout
```

### 4.1 Dst / Src0 layout

`Dst/Src0` 表示：

```text
C/D: M × N
```

其中：

```text
M = RC
N = Exec_size
```

可以按行理解：

```text
C row 0: c00 c01 c02 ... c0(N-1)
C row 1: c10 c11 c12 ... c1(N-1)
C row 2: c20 c21 c22 ... c2(N-1)
...
```

例如：

```text
RC = 4
Exec_size = 8
```

那么：

```text
Dst/Src0 = 4 × 8
```

直观理解：

```text
Repeat 0 -> C 第 0 行
Repeat 1 -> C 第 1 行
Repeat 2 -> C 第 2 行
Repeat 3 -> C 第 3 行
```

### 4.2 Src2 / A layout

`Src2 = A = M × K`

它也是 row-major：

```text
A row 0: a00 a01 a02 ... a0(K-1)
A row 1: a10 a11 a12 ... a1(K-1)
A row 2: a20 a21 a22 ... a2(K-1)
...
```

每个 repeat 处理 C 的一行，对应使用 A 的一行。

所以：

```text
C[0, :] 用 A[0, :] × B
C[1, :] 用 A[1, :] × B
C[2, :] 用 A[2, :] × B
...
```

### 4.3 Src1 / B layout

`Src1 = B = K × N`

这是 DPAS 最特殊的部分。

普通 row-major 的 B 是：

```text
B row 0: b00 b01 b02 ...
B row 1: b10 b11 b12 ...
B row 2: b20 b21 b22 ...
...
```

但是 DPAS 的 `Src1/B` 不是这么放的。

Intel 文档建议把 GRF 看成二维空间：

```text
一行 = 一个 GRF register
一列 = 一个 32-bit DW lane
```

其中：

```text
DW = Double Word = 32 bit = 4 byte
DW lane = 一个 GRF 里的第几个 32-bit 槽位
```

假设一个 GRF 是 64 byte，那么：

```text
64 byte / 4 byte = 16 DW
```

可以画成：

```text
          DW0   DW1   DW2   DW3   ...   DW15
GRF0      x     x     x     x           x
GRF1      x     x     x     x           x
GRF2      x     x     x     x           x
...
```

在理解 DPAS `Src1/B` layout 时，可以近似认为：

```text
DW lane n 对应 SIMD channel n
SIMD channel n 对应 C 的第 n 列
```

所以：

```text
B 的第 0 列 -> 放到 DW0 这一列
B 的第 1 列 -> 放到 DW1 这一列
B 的第 2 列 -> 放到 DW2 这一列
...
```

### 4.4 FP16 的 Src1/B 例子

假设：

```text
B: 16 × 8
K = 16
N = 8
FP16
```

因为：

```text
一个 DW = 32 bit
一个 FP16 = 16 bit
所以一个 DW 可以放 2 个 FP16
```

那么 `Src1/B` 可以粗略理解成：

```text
                 DW0              DW1              DW2          ...   DW7
                 B第0列           B第1列           B第2列              B第7列

GRF0        B[0,0],B[1,0]    B[0,1],B[1,1]    B[0,2],B[1,2]    ...  B[0,7],B[1,7]

GRF1        B[2,0],B[3,0]    B[2,1],B[3,1]    B[2,2],B[3,2]    ...  B[2,7],B[3,7]

GRF2        B[4,0],B[5,0]    B[4,1],B[5,1]    B[4,2],B[5,2]    ...  B[4,7],B[5,7]

...

GRF7        B[14,0],B[15,0]  B[14,1],B[15,1]  B[14,2],B[15,2] ...  B[14,7],B[15,7]
```

也就是：

```text
N 方向 -> 映射到 DW lane / SIMD channel
K 方向 -> 沿 GRF row 往下走
每个 DW -> pack 多个低精度元素
```

对 FP16/BF16：

```text
每个 DW pack 2 个元素
```

对 INT8：

```text
每个 DW pack 4 个元素
```

对 INT4：

```text
每个 DW pack 8 个元素
```

---

## 5. SIMD channel / DW lane / sub-group lane 的关系

几个词不要混：

```text
SIMD channel = SIMD 指令里的 lane
DW lane      = GRF register 里的第几个 32-bit 槽位
sub-group lane = SYCL sub-group 里的 local id
```

在 DPAS `Src1/B` layout 里，可以把它们对应起来理解：

```text
sub_group.get_local_linear_id()
≈ SIMD channel id
≈ DPAS 里的第几个输出列
≈ Src1/B 里对应的 DW lane
```

但严格说：

```text
DW lane 是寄存器存储视角
SIMD channel 是执行视角
sub-group lane 是 SYCL 编程视角
```

---

## 6. DPASW 是什么

`DPASW` 可以理解成：

```text
DPAS Wide
```

数学语义不变：

```text
D = C + A × B
```

operand 对应关系也不变：

```text
Dst  = D
Src0 = C
Src2 = A
Src1 = B
```

它不是：

```text
更大 K 的 DPAS
```

也不是：

```text
新的矩阵乘语义
```

它的核心变化是：

```text
在 fused EU / paired EU 场景下，两个 DPAS pipeline 共享 Src2/A 的读取
```

目的：

```text
减少 GRF read bandwidth 压力
改善 operand feed
```

---

## 7. DPASW layout 和 DPAS 的区别

DPASW 的大部分 layout 与 DPAS 相同：

```text
Dst/Src0 layout = 同 DPAS
Src1/B layout   = 同 DPAS，仍然是特殊 packed layout
```

真正特殊的是：

```text
Src2/A 的物理来源不同
```

普通 DPAS：

```text
一个 EU 自己提供完整 Src2/A
```

DPASW：

```text
两个 paired/fused EU 一起提供完整 Src2/A
EU0 提供前一部分 GRF
EU1 提供后一部分 GRF
然后组合成完整 Src2/A
```

可以理解为：

```text
DPAS:
    EU0 读一份完整 A
    EU1 读一份完整 A

DPASW:
    EU0 读 A 的前半部分
    EU1 读 A 的后半部分
    两个 DPAS pipeline 共享组合后的 A
```

所以 DPASW 的收益点是：

```text
减少 Src2/A 的重复 GRF 读取
```

不是改变 B layout，也不是让 K 变大。

---

## 8. GRF bandwidth / operand feed 是什么

### 8.1 GRF

```text
GRF = General Register File
```

也就是 EU / Vector Engine 里的通用寄存器文件。

DPAS 运行前，需要从 GRF 里读取：

```text
Src0/C accumulator
Src1/B tile
Src2/A tile
```

运行后还要写回：

```text
Dst/D accumulator
```

所以 DPAS 很吃寄存器读写带宽。

### 8.2 operand feed

`operand feed` 就是：

```text
把 A/B/C 操作数以正确 layout 从寄存器喂给 XMX/DPAS pipe
```

如果计算单元很快，但寄存器读数据、pack/unpack、shuffle、move 跟不上，就会出现：

```text
XMX 算力没满
DPAS pipe 等数据
```

这就是：

```text
GRF bandwidth / operand feed bottleneck
```

DPASW 的设计目的之一就是减少 Src2/A 的 GRF 读取压力。

---

## 9. DPAS vs DPASW 对比

| 项目       | DPAS                                     | DPASW                        |
| -------- | ---------------------------------------- | ---------------------------- |
| 数学语义     | `D = C + A × B`                          | `D = C + A × B`              |
| Dst/Src0 | C/D accumulator                          | 同 DPAS                       |
| Src2     | A                                        | A，但由 paired EU 分摊/共享         |
| Src1     | B，特殊 layout                              | 同 DPAS                       |
| 是否改变 K   | 不改变                                      | 不改变                          |
| 主要目的     | 基础 XMX 矩阵乘加                              | 减少 fused EU 场景下 Src2 GRF 读带宽 |
| 使用难度     | 已经较高                                     | 更高                           |
| 初学建议     | 先用这个                                     | 不建议一开始碰                      |
| 适合场景     | 自定义 GEMM / fused matmul / dequant matmul | 高级优化，且确认架构支持 DPASW           |

---

## 10. 支持架构

按 Intel IGC vISA 指令列表：

```text
DPAS  : XEHP+
DPASW : DG2, XEHP
```

需要注意：

```text
DPASW 不是所有有 DPAS/XMX 的架构都有
```

官方 DPASW 文档还特别说明：

```text
PVC does not have DPASW
```

所以不能简单认为：

```text
新架构一定有 DPASW
```

大致理解：

```text
Xe-LP / UHD / 老 Iris Xe-LP:
    没有 XMX DPAS
    主要走 SIMD ALU / DP4A 等路径

Xe-HPG / DG2 / Arc A 系列:
    有 XMX
    支持 DPAS
    vISA 标注支持 DPASW

Xe-HPC / PVC / Data Center GPU Max:
    有 XMX / DPAS
    但 PVC 没有 DPASW

Xe2 / Arc B 系列 / Lunar Lake / Battlemage:
    有 XMX 的产品可以走 DPAS 类路径
    但 DPASW 不能默认假设可用，要看编译器和设备支持
```

---

## 11. 用户层面如何选择 API

### 11.1 普通用户 / 框架用户

优先：

```text
oneDNN
oneMKL
深度学习框架
```

这些库在支持硬件和数据类型时会使用 XMX/DPAS。

### 11.2 自己写 SYCL GEMM

优先考虑：

```text
joint_matrix
joint_matrix_mad
```

它比直接写 `xmx::dpas` 更高层，更适合作为手写 GEMM 的入口。

### 11.3 自己写极致优化 kernel

才考虑：

```text
ESIMD xmx::dpas
ESIMD xmx::dpasw
```

适合：

```text
自定义 GEMM micro-kernel
fused GEMM + dequant
LLM int4/int8/fp16/bf16 matmul
需要手工控制寄存器 layout
需要手工 pack A/B operand
```

---

## 12. 和 CUDA MMA 的类比

CUDA：

```text
mma.sync.aligned.m16n8k16
```

Intel DPAS：

```text
通过 RC / Exec_size / SD / precision 组合出 tile
```

粗略对应：

```text
CUDA MMA 的 M 方向  -> DPAS 的 RC
CUDA MMA 的 N 方向  -> DPAS 的 Exec_size
CUDA MMA 的 K 方向  -> DPAS 的 SD × OPS_PER_CHAN
```

例如 FP16：

```text
SD=8
OPS_PER_CHAN=2
Ktile=16
```

类似 CUDA MMA 的 `k16`。

但注意：

```text
Intel DPAS 不直接写 m4n8k16 这种形式
```

而是通过参数和 operand layout 隐式表达。

---

## 13. Intel GPU 层级与术语

几个词的关系：

```text
GPU
  └── Xe-core / Subslice / Dual Subslice
        ├── EU / Vector Engine
        │     ├── SIMD ALU
        │     ├── SIMD channel
        │     ├── hardware threads
        │     └── GRF
        ├── XMX / Matrix Engine
        ├── SLM / local memory
        └── thread dispatch
```

类比 CUDA：

```text
CUDA SM
≈ Intel Xe-core / Subslice / Dual Subslice

CUDA SM 内部的执行单元 / ALU cluster
≈ Intel EU / Vector Engine

CUDA Tensor Core
≈ Intel XMX / Matrix Engine

CUDA warp
≈ SYCL sub-group

CUDA warp lane
≈ SIMD channel
≈ sub_group.get_local_linear_id()
```

注意：

```text
EU 不是 CUDA SM
EU 是 Xe-core/Subslice 内部的执行单元
```

---

## 14. 最容易搞混的点

### 14.1 Src1 和 Src2 容易反

正确是：

```text
Src2 = A = M × K
Src1 = B = K × N
```

### 14.2 K 不是完整 GEMM 的 K

正确是：

```text
Ktile = SD × OPS_PER_CHAN
```

完整 K 大于 Ktile 时，用循环累加。

例如 FP16：

```text
SD=8
Ktile=16
完整 K=128
=> 循环 8 次 DPAS
```

### 14.3 Src1/B 不是 row-major

正确是：

```text
Dst/Src0/Src2 是 row-major
Src1/B 是特殊 layout
```

Src1/B 的直觉：

```text
B 的 N 方向列 -> 映射到 DW lane / SIMD channel
B 的 K 方向 -> 沿 GRF row 方向推进
```

### 14.4 DW lane 不是新执行单元

```text
DW = 32-bit 槽位
DW lane = GRF 里的第几个 32-bit 位置
SIMD channel = 执行时的 lane
```

在 DPAS Src1 layout 中：

```text
DW lane n 对应 SIMD channel n
```

### 14.5 DPASW 不是 K 更大

错误理解：

```text
DPASW = 更大 K 的 DPAS
```

正确理解：

```text
DPASW = wide DPAS
主要变化是 paired/fused EU 共享 Src2/A
```

### 14.6 DPASW 不一定所有 XMX 架构都有

正确理解：

```text
DPAS 支持范围更广
DPASW 只在特定架构标注支持
PVC 明确没有 DPASW
```

### 14.7 work-group size 不是 DPAS 的核心参数

对普通 SYCL kernel，work-group size 可以从 128/256/512 测。

但对 DPAS/XMX kernel，真正核心是：

```text
sub-group size
RC / Exec_size / SD
A/B/C tile layout
GRF packing
每个 sub-group 负责的 C tile
```

---

## 15. 实战建议

### 15.1 初学路径

建议顺序：

```text
1. 先理解 joint_matrix
2. 再理解 DPAS 的 M/N/K 参数
3. 再理解 Src1/B 特殊 layout
4. 最后再看 DPASW
```

不要一开始直接上 DPASW。

### 15.2 FP16/BF16 matmul 推荐心智模型

先按：

```text
RC = 输出 M
Exec_size = 输出 N
SD = 8
Ktile = 16
```

例如：

```text
C: 4 × 8
A: 4 × 16
B: 16 × 8
```

对应：

```text
RC = 4
Exec_size = 8
SD = 8
FP16/BF16 => Ktile = 16
```

如果 K 更大：

```text
for kk += 16
```

循环累加。

### 15.3 DPASW 使用前提

只有满足这些条件再考虑：

```text
1. 已经把 DPAS 跑通
2. 目标架构确认支持 DPASW
3. 理解 paired/fused EU 的 Src2 sharing
4. layout 能让两个 EU 复用 A/Src2
5. profiling 显示 operand feed / GRF read 是瓶颈
```

否则 DPASW 很可能只是增加复杂度，不一定更快。

---

## 16. 一句话总结

```text
DPAS 是 Intel XMX 的基础矩阵乘加指令：
    D = C + A × B
    RC 决定 M
    Exec_size 决定 N
    SD × OPS_PER_CHAN 决定一次指令的 Ktile
    Src2 是 A，row-major
    Src1 是 B，特殊 packed layout

DPASW 是 DPAS Wide：
    数学语义不变
    Src1/B layout 不变
    特殊点是 paired/fused EU 共享 Src2/A
    目的是减少 GRF 读取和 operand feed 压力
    只适合特定架构和高级优化场景
```
