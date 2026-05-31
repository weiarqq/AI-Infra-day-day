# Intel SYCL 7 天学习路线：面向 llama.cpp 后端与算子融合

## 适用目标

这份学习路线面向已经熟悉 C/C++、了解 CUDA 基本概念、希望学习 Intel SYCL / oneAPI，并最终能阅读 llama.cpp SYCL backend、手写简单 fused kernel 的开发者。

重点不是泛泛学习 SYCL 语法，而是围绕大模型推理常见算子展开：

- 数据如何从 host 到 device
- kernel 如何组织并行
- work-group / sub-group 如何映射到硬件
- local memory / barrier 什么时候有价值
- reduce / shuffle 如何做规约与数据交换
- joint_matrix / DPAS / XMX 如何理解
- 如何阅读 llama.cpp 的 SYCL 后端
- 如何实现 fused RMSNorm
- 如何实现 fused MoE gate + up 的雏形

---

# 第 0 天：环境准备与心智模型

## 学习目标

在开始写 kernel 之前，先建立一个简单但非常重要的心智模型：

```text
C++ host code
    ↓ submit
SYCL queue
    ↓ dispatch
SYCL kernel
    ↓ mapped to
Intel GPU EUs / subslices / XMX
```

SYCL 程序一般由两部分组成：

1. host 侧 C++ 代码：负责分配内存、提交 kernel、同步结果。
2. device 侧 kernel：真正运行在 GPU 上的并行代码。

## 推荐环境

建议使用 Intel oneAPI Base Toolkit，并优先使用 Level Zero 后端。

常用环境变量：

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export SYCL_PI_TRACE=1
export SYCL_UR_TRACE=1
```

新版本 oneAPI 更推荐 Unified Runtime，因此很多场景会看到 `SYCL_UR_TRACE` 输出。

## 最小测试程序

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q{sycl::gpu_selector_v};

    std::cout << "Device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    return 0;
}
```

编译：

```bash
icpx -fsycl test.cpp -o test
./test
```

## 今日验收标准

你应该能够：

- 成功编译并运行一个 SYCL 程序。
- 知道 `queue` 是 host 向 device 提交任务的入口。
- 知道 Intel GPU 下通常优先走 Level Zero 后端。

---

# 第 1 天：USM、parallel_for、nd_range

## 今日目标

掌握最基本的 SYCL kernel 写法，理解以下概念：

- USM：Unified Shared Memory
- `parallel_for`
- `range`
- `nd_range`
- global id / local id / group id

---

## 1.1 USM 是什么

USM 可以理解为一种更接近 CUDA `cudaMalloc` 风格的内存模型。

常见分配方式：

```cpp
float *x = sycl::malloc_shared<float>(n, q);
float *y = sycl::malloc_device<float>(n, q);
float *z = sycl::malloc_host<float>(n, q);
```

三者区别：

| 类型            | host 可访问 |  device 可访问 | 常见用途                   |
| --------------- | ----------: | -------------: | -------------------------- |
| `malloc_shared` |          是 |             是 | 入门、原型验证             |
| `malloc_device` |          否 |             是 | 性能更明确的 device buffer |
| `malloc_host`   |          是 | 可通过拷贝访问 | pinned host memory         |

入门时可以先用 `malloc_shared`，后续优化时再转为 `malloc_device` + `memcpy`。

---

## 1.2 parallel_for：一维向量加法

```cpp
#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    constexpr int N = 1024;

    float *a = sycl::malloc_shared<float>(N, q);
    float *b = sycl::malloc_shared<float>(N, q);
    float *c = sycl::malloc_shared<float>(N, q);

    for (int i = 0; i < N; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }

    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> idx) {
        int i = idx[0];
        c[i] = a[i] + b[i];
    }).wait();

    std::cout << c[10] << std::endl;

    sycl::free(a, q);
    sycl::free(b, q);
    sycl::free(c, q);
}
```

这里的 `parallel_for(range<1>(N))` 表示启动 N 个 work-item。

每个 work-item 处理一个元素。

---

## 1.3 nd_range：显式指定 work-group

```cpp
constexpr int N = 1024;
constexpr int WG = 256;

q.parallel_for(
    sycl::nd_range<1>{sycl::range<1>(N), sycl::range<1>(WG)},
    [=](sycl::nd_item<1> item) {
        int gid = item.get_global_id(0);
        int lid = item.get_local_id(0);
        int group = item.get_group(0);

        c[gid] = a[gid] + b[gid];
    }
).wait();
```

`nd_range` 把并行空间拆成：

```text
global range = 总 work-item 数
local range  = 每个 work-group 的 work-item 数
```

对应关系：

```text
global_id = group_id * local_size + local_id
```

---

## 1.4 和 CUDA 的类比

| CUDA          | SYCL                                |
| ------------- | ----------------------------------- |
| grid          | global range                        |
| block         | work-group                          |
| thread        | work-item                           |
| blockDim.x    | local range                         |
| threadIdx.x   | local id                            |
| blockIdx.x    | group id                            |
| cudaMalloc    | sycl::malloc_device / malloc_shared |
| kernel launch | queue.parallel_for                  |

---

## 今日练习

### 练习 1：vector add

实现：

```cpp
c[i] = a[i] + b[i]
```

要求：

- 用 `malloc_shared`
- 用 `parallel_for(range)` 写一版
- 用 `parallel_for(nd_range)` 写一版

### 练习 2：scale kernel

实现：

```cpp
y[i] = alpha * x[i]
```

### 练习 3：处理越界

把 N 改成不是 work-group size 整数倍，比如：

```cpp
N = 1000;
WG = 256;
global = round_up(N, WG);
```

kernel 中加：

```cpp
if (gid < N) { ... }
```

## 今日验收标准

你应该能够：

- 解释 `range` 和 `nd_range` 的区别。
- 解释 global id、local id、group id 的关系。
- 独立写出一个一维 SYCL kernel。

---

# 第 2 天：work-group、local memory、barrier

## 今日目标

理解 work-group 的能力：

- group 内 work-item 可以协作。
- group 内 work-item 可以共享 local memory。
- group 内 work-item 可以用 barrier 同步。
- 不同 work-group 之间不能直接同步。

---

## 2.1 work-group 是什么

一个 work-group 是一组 work-item。

它的特点是：

```text
同一个 work-group 内：
    可以共享 local memory
    可以 barrier 同步
    可以做 group reduce

不同 work-group 之间：
    不能直接通信
    不能在一个 kernel 内全局同步
```

这和 CUDA block 基本对应。

---

## 2.2 local memory

SYCL 中常见写法：

```cpp
sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);
```

完整示例：

```cpp
q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>{sycl::range<1>(N), sycl::range<1>(WG)},
        [=](sycl::nd_item<1> item) {
            int gid = item.get_global_id(0);
            int lid = item.get_local_id(0);

            scratch[lid] = x[gid];
            item.barrier(sycl::access::fence_space::local_space);

            y[gid] = scratch[lid] * 2.0f;
        }
    );
}).wait();
```

---

## 2.3 barrier 的作用

barrier 保证同一个 work-group 内所有 work-item 都执行到这个位置后，才继续往下执行。

典型用途：

```cpp
scratch[lid] = x[gid];
item.barrier(sycl::access::fence_space::local_space);

// 此时可以安全读取其他 work-item 写入 scratch 的数据
float neighbor = scratch[(lid + 1) % WG];
```

如果没有 barrier，某些 work-item 可能还没来得及写，其他 work-item 就已经开始读了。

---

## 2.4 local memory 什么时候有价值

local memory 适合：

- group 内复用数据
- tile-based matrix multiplication
- convolution tile
- block reduction
- stencil / neighbor access

不适合：

- 每个元素只读一次、没有复用
- 数据访问本来就是完全连续且简单
- local memory 引入了额外同步但没有减少 global memory 访问

在大模型推理里，local memory 常见于：

- 小规模 reduce
- softmax 局部缓存
- matmul tile
- 某些 norm kernel

---

## 2.5 block reduce 示例

目标：每个 work-group 对一段数据求和。

```cpp
q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);

    cgh.parallel_for(
        sycl::nd_range<1>{sycl::range<1>(num_groups * WG), sycl::range<1>(WG)},
        [=](sycl::nd_item<1> item) {
            int gid = item.get_global_id(0);
            int lid = item.get_local_id(0);
            int group = item.get_group(0);

            scratch[lid] = x[gid];
            item.barrier(sycl::access::fence_space::local_space);

            for (int stride = WG / 2; stride > 0; stride >>= 1) {
                if (lid < stride) {
                    scratch[lid] += scratch[lid + stride];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (lid == 0) {
                partial[group] = scratch[0];
            }
        }
    );
}).wait();
```

---

## 今日练习

### 练习 1：neighbor copy

实现：

```cpp
y[i] = x[i] + x[i + 1]
```

要求：

- 用 local memory 缓存一个 work-group 内的数据。
- 思考 group 边界如何处理。

### 练习 2：block reduce

实现每个 work-group 输出一个 partial sum。

### 练习 3：两阶段 reduce

第一阶段：每个 group 输出 partial sum。

第二阶段：再对 partial sum 做一次 reduce。

## 今日验收标准

你应该能够：

- 解释 work-group 的能力边界。
- 解释 local memory 和 global memory 的区别。
- 解释为什么 barrier 只能同步同一个 work-group。
- 写出一个 block reduce kernel。

---

# 第 3 天：sub_group、reduce_over_group、shuffle

## 今日目标

理解 sub-group 的特殊性：

- sub-group 是 work-group 内更小的执行单元。
- sub-group 通常更接近硬件 SIMD / wave / warp 的执行粒度。
- sub-group 内通信通常比 local memory + barrier 更轻量。
- sub-group 适合小范围 reduce、broadcast、shuffle。

---

## 3.1 work-group 和 sub-group 的区别

| 概念       | SYCL                           | CUDA 类比 | 特点                        |
| ---------- | ------------------------------ | --------- | --------------------------- |
| work-item  | 单个执行实例                   | thread    | 最小编程单元                |
| sub-group  | 一组 lockstep 执行的 work-item | warp      | 适合 shuffle/reduce         |
| work-group | 一组 work-item                 | block     | 可用 local memory + barrier |

sub-group 是 work-group 内部的分组。

一个 work-group 里通常会包含多个 sub-group。

---

## 3.2 获取 sub_group

```cpp
cgh.parallel_for(
    sycl::nd_range<1>{sycl::range<1>(N), sycl::range<1>(WG)},
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        sycl::sub_group sg = item.get_sub_group();

        int sg_local_id = sg.get_local_id()[0];
        int sg_size = sg.get_local_range()[0];
    }
);
```

`[[sycl::reqd_sub_group_size(16)]]` 表示请求 sub-group size 为 16。

注意：

- 不是所有设备都支持所有 sub-group size。
- Intel GPU 上常见 size 包括 8、16、32，具体要查询设备能力。
- 对性能敏感时，不要盲目指定，要结合硬件和 kernel 特征。

---

## 3.3 查询支持的 sub-group size

```cpp
auto dev = q.get_device();

auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();

for (auto s : sizes) {
    std::cout << "sub_group size: " << s << std::endl;
}
```

---

## 3.4 reduce_over_group

SYCL 提供 group algorithm：

```cpp
float sum = sycl::reduce_over_group(sg, value, sycl::plus<float>());
```

这会在 sub-group 内做 reduce。

示例：

```cpp
q.parallel_for(
    sycl::nd_range<1>{sycl::range<1>(N), sycl::range<1>(WG)},
    [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        int gid = item.get_global_id(0);
        sycl::sub_group sg = item.get_sub_group();

        float v = x[gid];
        float s = sycl::reduce_over_group(sg, v, sycl::plus<float>());

        if (sg.get_local_id()[0] == 0) {
            partial[item.get_global_id(0) / sg.get_local_range()[0]] = s;
        }
    }
).wait();
```

---

## 3.5 shuffle

shuffle 可以在 sub-group 内直接交换寄存器数据。

```cpp
float other = sycl::select_from_group(sg, value, 0);
```

含义：把 sub-group 内 lane 0 的 `value` 广播给所有 lane。

常见模式：

```cpp
float lane0 = sycl::select_from_group(sg, value, 0);
```

这类似 CUDA 的 `__shfl_sync`。

---

## 3.6 sub-group 适合什么

适合：

- warp-level reduce
- 小向量 dot product
- softmax 局部 max/sum
- RMSNorm 中的局部平方和
- MoE gate top-k 的局部候选规约
- matmul 内部小 tile 协作

不适合：

- 需要跨 work-group 同步的任务
- 超大范围 reduce，除非分阶段
- 复杂随机访存为主、sub-group 协作价值低的任务

---

## 今日练习

### 练习 1：sub-group reduce sum

每个 sub-group 对一段输入求和，输出到 partial array。

### 练习 2：sub-group max

每个 sub-group 求最大值。

### 练习 3：broadcast

让每个 sub-group 的 lane 0 读取一个 scale，然后广播给其他 lane。

伪代码：

```cpp
float scale = 0;
if (sg_lid == 0) {
    scale = scales[sg_id];
}
scale = sycl::select_from_group(sg, scale, 0);
y[gid] = x[gid] * scale;
```

## 今日验收标准

你应该能够：

- 解释 sub-group 和 work-group 的区别。
- 使用 `reduce_over_group` 做 sub-group 规约。
- 使用 `select_from_group` 做广播。
- 理解为什么 sub-group 通信通常比 local memory 更轻量。

---

# 第 4 天：joint_matrix、DPAS、XMX

## 今日目标

理解 Intel GPU 上矩阵乘法加速的基本路径：

```text
普通标量 / SIMD FMA
    ↓
sub-group 协作
    ↓
joint_matrix
    ↓
DPAS instruction
    ↓
XMX hardware
```

---

## 4.1 XMX 是什么

XMX 可以理解为 Intel GPU 上面向矩阵运算的硬件加速单元。

在大模型推理中，最核心的计算通常是矩阵乘法：

```text
Y = XW
```

如果硬件有 XMX，那么高性能 matmul 通常会通过 DPAS 类指令利用 XMX。

---

## 4.2 DPAS 是什么

DPAS 可以粗略理解为 Intel GPU 上用于矩阵点积累加的底层指令族。

你可以把它类比成 NVIDIA Tensor Core 背后的 MMA 指令。

从编程层次看：

```text
joint_matrix API
    ↓ compiler lowering
DPAS instruction
    ↓ hardware execution
XMX
```

也就是说，通常你不会在标准 SYCL 代码里直接手写 DPAS 指令，而是通过更高层的 API 或库触发它。

---

## 4.3 joint_matrix 是什么

`joint_matrix` 是 SYCL / Intel 扩展中用于表达矩阵 tile 协作计算的 API。

它更接近 CUDA WMMA / MMA 编程模型，而不是普通的 scalar kernel。

核心思路：

```text
一个 sub-group 共同持有一个矩阵 tile
一个 sub-group 共同加载 A tile / B tile
一个 sub-group 共同执行 matrix multiply accumulate
一个 sub-group 共同 store C tile
```

---

## 4.4 普通 matmul 和 joint_matrix 的区别

普通写法：

```cpp
float acc = 0;
for (int k = 0; k < K; ++k) {
    acc += A[m * K + k] * B[k * N + n];
}
C[m * N + n] = acc;
```

joint_matrix 写法的抽象：

```cpp
joint_matrix<sub_group, T, use::a, M, K, layout> sub_a;
joint_matrix<sub_group, T, use::b, K, N, layout> sub_b;
joint_matrix<sub_group, Tacc, use::accumulator, M, N> sub_c;

joint_matrix_load(sg, sub_a, ptr_a, stride_a);
joint_matrix_load(sg, sub_b, ptr_b, stride_b);
joint_matrix_mad(sg, sub_c, sub_a, sub_b, sub_c);
joint_matrix_store(sg, sub_c, ptr_c, stride_c, layout);
```

重点不是每个 work-item 算一个 C 元素，而是整个 sub-group 协作计算一个 tile。

---

## 4.5 joint_matrix 适合什么

适合：

- GEMM
- batched GEMM
- attention projection
- FFN up/down/gate projection
- MoE expert matmul
- 小矩阵融合 kernel 中的 tile matmul 部分

不适合：

- 元素级简单操作
- RMSNorm 这种以 reduce + elementwise 为主的 kernel
- 访存模式很乱且无法形成规则 tile 的计算

---

## 4.6 oneMKL GEMM vs 手写 joint_matrix

`oneapi::mkl::blas::gemm` 通常是单独的高性能 GEMM kernel。

优点：

- 稳定
- 调优成熟
- 适合大矩阵
- 通常能较好利用 XMX

缺点：

- kernel 边界固定
- GEMM 前后的轻量操作不容易融合
- 对特殊小矩阵 / MoE 多 expert / batch=1 场景，可能有额外 launch 和访存开销

手写 fused kernel 的价值通常不在于超过大型 GEMM，而在于：

```text
减少 kernel launch
减少中间结果写回 global memory
融合小矩阵前后的 elementwise / activation / routing
适配 batch=1、token 数小、expert 数少的特殊形状
```

---

## 今日练习

### 练习 1：理解 tile matmul

先不用 joint_matrix，手写一个 tiled matmul：

- 使用 work-group
- 使用 local memory
- 每个 work-item 算一个 C 元素

### 练习 2：阅读 joint_matrix 示例

关注以下问题：

- 一个 sub-group 负责多大的 C tile？
- A/B/C 的 layout 分别是什么？
- K 维循环如何推进？
- accumulator 用什么类型？

### 练习 3：思考 fused matmul

以：

```text
Y = silu(XW_gate) * (XW_up)
```

为例，思考两种实现：

1. 两个 GEMM + 一个 elementwise kernel。
2. 一个 fused kernel 同时算 gate/up 并做 silu 乘法。

## 今日验收标准

你应该能够：

- 解释 joint_matrix、DPAS、XMX 三者关系。
- 理解 joint_matrix 更接近 MMA/WMMA，而不是普通 kernel。
- 理解为什么库 GEMM 通常快，但 fused kernel 在特殊场景仍有价值。

---

# 第 5 天：阅读 llama.cpp SYCL backend

## 今日目标

开始阅读真实项目，不要求第一天读懂全部，而是建立索引地图。

建议关注：

```text
ggml-sycl
SYCL backend initialization
buffer allocation
kernel launch wrapper
matmul path
norm path
rope / softmax / elementwise kernels
oneMKL / custom kernel 边界
```

---

## 5.1 阅读顺序

推荐不要从最复杂的 matmul 开始。

建议顺序：

1. backend 初始化
2. device / queue 管理
3. buffer 分配和拷贝
4. 简单 elementwise kernel
5. norm kernel
6. softmax kernel
7. matmul / dequant / quantized matmul
8. graph 调度与 op 分派

---

## 5.2 你要找的关键问题

阅读每个 kernel 时，记录以下信息：

```text
这个 kernel 对应哪个 ggml op？
输入 tensor layout 是什么？
输出 tensor layout 是什么？
每个 work-item 负责什么？
每个 work-group 负责什么？
有没有使用 local memory？
有没有使用 sub-group？
有没有显式 barrier？
有没有 vectorized load/store？
有没有调用 oneMKL / oneDNN？
是否存在可融合机会？
```

---

## 5.3 llama.cpp 后端阅读模板

建议你建一个笔记表：

| op      | 文件位置 | kernel 名称 | 并行粒度           | 是否用 local memory | 是否用 sub-group | 可融合机会       |
| ------- | -------- | ----------- | ------------------ | ------------------- | ---------------- | ---------------- |
| RMSNorm | 待填写   | 待填写      | 每 row 一个 group? | 可能                | 可能             | norm + scale     |
| softmax | 待填写   | 待填写      | 每 row 一个 group? | 可能                | 可能             | mask + softmax   |
| matmul  | 待填写   | 待填写      | tile               | 可能                | 可能             | dequant + matmul |
| MoE     | 待填写   | 待填写      | expert/token       | 可能                | 可能             | gate + up        |

---

## 5.4 阅读重点：kernel launch 形状

看到 kernel 时优先问：

```cpp
sycl::nd_range<1>{global, local}
```

或者：

```cpp
sycl::nd_range<2>{global_2d, local_2d}
```

然后把它翻译成：

```text
global range 是什么？
local range 是什么？
一个 work-group 处理一个 token？一个 row？一个 tile？
一个 work-item 处理一个元素？多个元素？
```

如果能完成这个翻译，kernel 已经读懂了一半。

---

## 5.5 阅读重点：tensor stride

llama.cpp / ggml tensor 经常不是简单连续矩阵。

要特别关注：

```text
ne: number of elements per dimension
nb: byte stride per dimension
```

读 kernel 时要把指针计算写成公式：

```text
ptr = base + i0 * nb0 + i1 * nb1 + i2 * nb2 + ...
```

很多 kernel 难读，不是因为 SYCL 难，而是因为 tensor layout 和 stride 复杂。

---

## 今日练习

### 练习 1：找 3 个简单 kernel

在 llama.cpp SYCL backend 中找 3 个简单 kernel：

- add
- mul
- scale
- rope
- norm

任选三个。

对每个 kernel 写出：

```text
输入是什么？
输出是什么？
每个 work-item 做什么？
global/local range 如何设置？
```

### 练习 2：找 matmul 路径

回答：

```text
什么时候走 oneMKL？
什么时候走自定义 kernel？
量化 matmul 有没有单独路径？
```

### 练习 3：找一个可融合点

例如：

```text
RMSNorm + scale
Gate projection + activation
Dequant + matmul
MoE routing + expert projection
```

## 今日验收标准

你应该能够：

- 找到 SYCL backend 的主要文件。
- 看懂一个简单 elementwise kernel。
- 能把 `nd_range` 翻译成实际并行任务分配。
- 知道 llama.cpp 中哪些部分可能调用库，哪些是自定义 kernel。

---

# 第 6 天：自己写一个 fused RMSNorm

## 今日目标

实现一个简化版 fused RMSNorm kernel。

RMSNorm 公式：

```text
rms = sqrt(mean(x_i^2) + eps)
y_i = x_i / rms * weight_i
```

对应大模型中常见的逐 token、逐 hidden dimension 归一化。

---

## 6.1 输入输出设计

假设输入为二维：

```text
x:      [num_tokens, hidden_size]
weight: [hidden_size]
y:      [num_tokens, hidden_size]
```

每一行是一个 token 的 hidden state。

---

## 6.2 并行设计

简单设计：

```text
一个 work-group 处理一个 token row
一个 work-item 处理 hidden 维上的多个元素
work-group 内做 reduce sum(x_i^2)
算出 inv_rms
再写出 y_i
```

---

## 6.3 kernel 骨架

```cpp
template<int WG>
void rmsnorm_kernel(
    sycl::queue &q,
    const float *x,
    const float *weight,
    float *y,
    int num_tokens,
    int hidden_size,
    float eps
) {
    sycl::range<1> global(num_tokens * WG);
    sycl::range<1> local(WG);

    q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(WG), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>{global, local},
            [=](sycl::nd_item<1> item) {
                int token = item.get_group(0);
                int lid = item.get_local_id(0);

                const float *row = x + token * hidden_size;
                float *out = y + token * hidden_size;

                float sum = 0.0f;

                for (int i = lid; i < hidden_size; i += WG) {
                    float v = row[i];
                    sum += v * v;
                }

                scratch[lid] = sum;
                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = WG / 2; stride > 0; stride >>= 1) {
                    if (lid < stride) {
                        scratch[lid] += scratch[lid + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                float inv_rms = sycl::rsqrt(scratch[0] / hidden_size + eps);

                for (int i = lid; i < hidden_size; i += WG) {
                    out[i] = row[i] * inv_rms * weight[i];
                }
            }
        );
    }).wait();
}
```

---

## 6.4 可以继续优化的方向

### 方向 1：sub-group reduce

先在 sub-group 内 reduce，再把每个 sub-group 的结果写入 local memory，减少 barrier 次数。

### 方向 2：vectorized load/store

让每个 work-item 一次处理多个连续元素，例如 `float4` 或 half vector。

### 方向 3：半精度输入，float 累加

大模型推理常见：

```text
input: fp16 / bf16
accumulator: fp32
output: fp16 / bf16
```

### 方向 4：融合 residual / scale

例如：

```text
y = rmsnorm(x + residual) * weight
```

或者：

```text
y = rmsnorm(x) * weight
```

---

## 今日练习

### 练习 1：实现 CPU reference

```cpp
for token in tokens:
    sum = 0
    for i in hidden:
        sum += x[token][i] * x[token][i]
    inv = 1 / sqrt(sum / hidden_size + eps)
    for i in hidden:
        y[token][i] = x[token][i] * inv * weight[i]
```

### 练习 2：实现 SYCL kernel

使用 work-group reduce。

### 练习 3：验证误差

比较 CPU 和 GPU 输出：

```cpp
max_abs_error < 1e-4
```

### 练习 4：benchmark

测试不同 hidden size：

```text
1024
2048
4096
8192
```

测试不同 work-group size：

```text
128
256
512
```

## 今日验收标准

你应该能够：

- 写出一个可运行的 RMSNorm SYCL kernel。
- 解释为什么一个 row 用一个 work-group 是合理的起点。
- 解释 reduce 的两阶段：先每个 work-item 局部累加，再 group 内归约。
- 知道下一步如何用 sub-group 优化。

---

# 第 7 天：自己写一个 fused MoE gate + up

## 今日目标

实现一个简化版 fused MoE gate + up kernel 的设计雏形。

这里重点是理解融合思路，不追求第一版就超过 oneMKL GEMM。

---

## 7.1 FFN / MoE 中的 gate + up

常见 SwiGLU / gated FFN 形式：

```text
gate = X W_gate
up   = X W_up
out  = silu(gate) * up
```

其中：

```text
silu(x) = x / (1 + exp(-x))
```

如果不融合，可能是：

```text
GEMM gate
GEMM up
elementwise silu_mul
```

融合目标：

```text
一个 kernel 中同时计算 gate 和 up，并直接输出 silu(gate) * up
```

---

## 7.2 为什么 batch=1 / 小 token 数有融合价值

大 batch、大矩阵时，库 GEMM 往往非常强。

但 batch=1 或 token 数很小时，瓶颈可能变成：

- kernel launch overhead
- 中间 tensor 写回 global memory
- 多个小 GEMM 无法充分吃满硬件
- MoE expert 分散导致矩阵规模更小

此时 fused kernel 可能有探索价值。

---

## 7.3 简化问题设定

先不要一上来处理完整 MoE routing。

第一版只做：

```text
输入：
X:       [num_tokens, hidden_size]
W_gate:  [hidden_size, inter_size]
W_up:    [hidden_size, inter_size]
Y:       [num_tokens, inter_size]

输出：
Y[t, j] = silu(sum_i X[t, i] * W_gate[i, j])
          * sum_i X[t, i] * W_up[i, j]
```

这其实是 fused gated FFN projection。

等这个跑通后，再加 expert 维度。

---

## 7.4 最简单 kernel 设计

```text
二维 nd_range：
    dim0: token
    dim1: output channel j

每个 work-item 负责一个 Y[token, j]
```

伪代码：

```cpp
int token = item.get_global_id(0);
int j = item.get_global_id(1);

float gate = 0;
float up = 0;

for (int i = 0; i < hidden_size; ++i) {
    float xv = X[token * hidden_size + i];
    gate += xv * W_gate[i * inter_size + j];
    up   += xv * W_up[i * inter_size + j];
}

float s = gate / (1.0f + sycl::exp(-gate));
Y[token * inter_size + j] = s * up;
```

优点：

- 最容易写。
- 最容易验证。
- 清楚表达融合含义。

缺点：

- 每个 output channel 重复读取 X。
- 没有 tile。
- 没有利用 local memory / sub-group / XMX。
- 性能通常不会好。

---

## 7.5 第一版 kernel 骨架

```cpp
void fused_gate_up_kernel(
    sycl::queue &q,
    const float *x,
    const float *w_gate,
    const float *w_up,
    float *y,
    int num_tokens,
    int hidden_size,
    int inter_size
) {
    sycl::range<2> global(num_tokens, inter_size);

    q.parallel_for(global, [=](sycl::id<2> id) {
        int token = id[0];
        int j = id[1];

        const float *xrow = x + token * hidden_size;

        float gate = 0.0f;
        float up = 0.0f;

        for (int i = 0; i < hidden_size; ++i) {
            float xv = xrow[i];
            gate += xv * w_gate[i * inter_size + j];
            up   += xv * w_up[i * inter_size + j];
        }

        float silu = gate / (1.0f + sycl::exp(-gate));
        y[token * inter_size + j] = silu * up;
    }).wait();
}
```

---

## 7.6 第二版：tile 输出通道

更合理的设计：

```text
一个 work-group 负责：
    一个 token
    一段 output channels

local memory 缓存 X 的一部分
多个 work-item 协作计算多个 j
```

思路：

```text
for k_tile in hidden_size:
    load X tile into local memory
    barrier
    each work-item accumulates gate/up for its output j
    barrier
apply silu and store
```

这可以减少 X 的重复读取。

---

## 7.7 第三版：引入 MoE expert

MoE 情况下，一般有：

```text
topk_expert_ids[token, k]
topk_weights[token, k]
```

每个 token 只进入少数几个 expert。

简化公式：

```text
for each token t:
    for each selected expert e:
        gate = X[t] W_gate[e]
        up   = X[t] W_up[e]
        tmp  = silu(gate) * up
        output += topk_weight[t, e] * tmp W_down[e]
```

你第 7 天只做 gate + up，不做 down projection，也可以。

---

## 7.8 fused MoE gate + up 的难点

主要难点不是公式，而是 shape 和调度：

```text
不同 token 选择不同 expert
不同 expert 负载不均衡
batch=1 时 token 数少，并行度不足
每个 expert 是小 GEMM
权重量化格式可能复杂
中间结果是否要落 global memory
```

这也是为什么 MoE 推理优化经常围绕：

- expert batching
- token grouping by expert
- grouped GEMM
- fused activation
- fused dequant
- persistent kernel
- 专门处理 batch=1 的小矩阵路径

---

## 今日练习

### 练习 1：CPU reference

实现：

```text
gate = X W_gate
up = X W_up
Y = silu(gate) * up
```

### 练习 2：最简单 SYCL fused kernel

每个 work-item 算一个输出元素。

### 练习 3：和非融合版本比较

非融合版本：

1. kernel 1 算 gate
2. kernel 2 算 up
3. kernel 3 算 silu(gate) * up

比较：

```text
kernel launch 次数
中间 tensor 写入量
总耗时
```

### 练习 4：加 expert 维度

先固定每个 token 只有一个 expert：

```text
expert_id[token]
```

然后权重变成：

```text
W_gate: [num_experts, hidden_size, inter_size]
W_up:   [num_experts, hidden_size, inter_size]
```

## 今日验收标准

你应该能够：

- 写出一个 fused gate + up 的最小 kernel。
- 解释为什么第一版性能可能不好，但教学价值高。
- 解释 MoE 融合的核心瓶颈。
- 明确下一步优化方向：tile、sub-group、joint_matrix、expert batching。

---

# 7 天之后：继续深入方向

## 方向 A：从 RMSNorm 进入高质量 kernel 优化

继续优化：

- sub-group reduce
- vectorized load/store
- fp16/bf16 input + fp32 accumulate
- 多 token per work-group
- residual add + RMSNorm fusion

## 方向 B：从 gate + up 进入 FFN fusion

继续优化：

```text
X W_gate
X W_up
silu(gate) * up
optional quant/dequant
optional down projection
```

关注：

- 中间结果是否落 global memory
- 是否用 oneMKL 做大 GEMM
- 是否为 batch=1 写专用小矩阵 kernel

## 方向 C：从 joint_matrix 进入 XMX

继续学习：

- joint_matrix tile shape
- sub-group size 和 tile 的关系
- accumulator 类型
- bf16/fp16/int8 路径
- DPAS 是否被编译器生成
- 用 `ocloc` / `llvm-objdump` / VTune 查看实际指令和性能

## 方向 D：结合 llama.cpp 做真实贡献

可以尝试：

1. 找一个简单 op 加 SYCL kernel。
2. 找一个已有 kernel 做小优化。
3. 给某个 shape 加 fast path。
4. 做 benchmark，对比 CPU / Vulkan / SYCL。
5. 写 profiling 记录，确认瓶颈是 bandwidth、launch overhead 还是 compute。

---

# 建议的每日学习节奏

每天建议按下面节奏进行：

```text
30 分钟：读概念
60 分钟：敲代码
30 分钟：调试和验证正确性
30 分钟：benchmark
30 分钟：写笔记
```

学习 SYCL 不要只看 API。

每一天都应该产出一个能运行的小程序。

---

# 最终项目目标

完成 7 天后，你应该至少拥有以下代码：

```text
01_vector_add.cpp
02_block_reduce.cpp
03_subgroup_reduce.cpp
04_tiled_matmul_or_joint_matrix_demo.cpp
05_llama_sycl_backend_notes.md
06_fused_rmsnorm.cpp
07_fused_gate_up.cpp
```

最终你应该能回答：

```text
这个 kernel 的并行粒度是什么？
每个 work-group 做什么？
每个 sub-group 做什么？
有没有 local memory？
有没有 barrier？
有没有减少 global memory 读写？
有没有减少 kernel launch？
有没有机会使用 XMX？
这个融合是否真的值得？
```

如果能回答这些问题，你就已经从“会写 SYCL 语法”进入了“能做 SYCL 后端优化”的阶段。

