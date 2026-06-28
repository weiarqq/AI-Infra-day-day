# SYCL / ESIMD 常见 API 速查笔记

## 1. `ndi.get_local_id(dim)`

### API

```cpp
auto lid = ndi.get_local_id(dim);
```

### 要点

`get_local_id(dim)` 用来获取当前 work-item 在 **当前 work-group 内某一维度上的局部编号**。

在 1D kernel 中：

```cpp
auto lid = ndi.get_local_id(0);
```

基本等价于 CUDA 的：

```cpp
threadIdx.x
```

在 2D kernel 中：

```cpp
auto ly = ndi.get_local_id(0);
auto lx = ndi.get_local_id(1);
```

可以类比 CUDA：

```cpp
threadIdx.y
threadIdx.x
```

### 示例

```cpp
sycl::nd_range<2>(
    sycl::range<2>(64, 64),
    sycl::range<2>(8, 16)
)
```

一个 work-group 内是 `8 x 16` 个 work-item。

对于某个 work-item：

```cpp
auto ly = ndi.get_local_id(0);
auto lx = ndi.get_local_id(1);
```

可能得到：

```cpp
ly = 3;
lx = 5;
```

说明它在当前 work-group 内的位置是第 3 行、第 5 列。

---

## 2. `ndi.get_local_linear_id()`

### API

```cpp
int localLinearId = ndi.get_local_linear_id();
```

### 要点

`get_local_linear_id()` 获取当前 work-item 在 **当前 work-group 内的一维线性编号**。

范围是：

```cpp
0 ~ work_group_size - 1
```

在 1D kernel 中，它通常和：

```cpp
ndi.get_local_id(0)
```

结果一样。

### 1D 示例

```cpp
sycl::nd_range<1>(
    sycl::range<1>(1024),
    sycl::range<1>(256)
)
```

每个 work-group 有 256 个 work-item。

那么：

```cpp
ndi.get_local_linear_id()
```

的范围是：

```cpp
0, 1, 2, ..., 255
```

每个 work-group 都会重新从 0 开始编号。

### 2D 示例

```cpp
sycl::nd_range<2>(
    sycl::range<2>(64, 64),
    sycl::range<2>(8, 16)
)
```

当前 work-item：

```cpp
ly = ndi.get_local_id(0); // 3
lx = ndi.get_local_id(1); // 5
```

那么线性编号可以理解为：

```cpp
linear = ly * local_range(1) + lx;
```

也就是：

```cpp
linear = 3 * 16 + 5 = 53;
```

所以：

```cpp
ndi.get_local_linear_id()
```

大致返回：

```cpp
53
```

---

## 3. `get_local_id()` 和 `get_local_linear_id()` 的区别

### API 对比

```cpp
ndi.get_local_id(dim)
ndi.get_local_linear_id()
```

### 要点

| API                     | 含义                       |
| ----------------------- | ------------------------ |
| `get_local_id(dim)`     | 获取某一维上的 local id         |
| `get_local_linear_id()` | 获取当前 work-group 内的一维线性编号 |

### 1D 情况

在 1D kernel 中：

```cpp
ndi.get_local_id(0) == ndi.get_local_linear_id()
```

例如 local size = 256：

```cpp
get_local_id(0)       = 0 ~ 255
get_local_linear_id() = 0 ~ 255
```

### 2D / 3D 情况

在多维 work-group 中：

```cpp
get_local_id(0)
get_local_id(1)
get_local_id(2)
```

表示多维坐标。

而：

```cpp
get_local_linear_id()
```

表示把多维坐标展开成一维后的编号。

### CUDA 类比

```cpp
// CUDA
int tid = threadIdx.y * blockDim.x + threadIdx.x;
```

对应 SYCL：

```cpp
int tid = ndi.get_local_linear_id();
```

或者手写：

```cpp
int tid = ndi.get_local_id(0) * ndi.get_local_range(1)
        + ndi.get_local_id(1);
```

### 什么时候用哪个？

如果是矩阵、图像、tile 这种二维结构，通常用：

```cpp
auto row = ndi.get_local_id(0);
auto col = ndi.get_local_id(1);
```

如果是归约、初始化 local memory、按线程编号分工，通常用：

```cpp
auto tid = ndi.get_local_linear_id();
```

---

## 4. `simd<float, 1024>`

### API

```cpp
sycl::ext::intel::esimd::simd<float, 1024>
```

常见简写：

```cpp
using namespace sycl::ext::intel::esimd;

simd<float, 1024> data;
```

### 要点

`simd<float, 1024>` 是 Intel ESIMD 中的显式 SIMD 向量类型。

它表示一个包含 1024 个 `float` 元素的 SIMD 向量。

可以近似理解为：

```cpp
float data[1024];
```

但它不是普通数组，而是 ESIMD 的向量对象。

### 数据大小

```cpp
1024 * sizeof(float) = 1024 * 4 = 4096 bytes
```

也就是 4 KB。

### 示例

```cpp
simd<float, 1024> a;
simd<float, 1024> b;
simd<float, 1024> c;

c = a + b;
```

语义上类似：

```cpp
for (int i = 0; i < 1024; ++i) {
    c[i] = a[i] + b[i];
}
```

### 注意

`simd<float, 1024>` 不代表硬件一定用一条 SIMD 指令完成 1024 个 float 的操作。

实际编译后，编译器可能会拆成多条底层向量指令执行。

---

## 5. `simd<float, 1024> inputData = 0;`

### API

```cpp
simd<float, 1024> inputData = 0;
```

### 要点

这表示定义一个包含 1024 个 `float` 元素的 ESIMD 向量，并把所有元素初始化为 0。

等价理解：

```cpp
float inputData[1024];

for (int i = 0; i < 1024; ++i) {
    inputData[i] = 0.0f;
}
```

也可以写成：

```cpp
simd<float, 1024> inputData(0.0f);
```

或者：

```cpp
simd<float, 1024> inputData = 0.0f;
```

### 重点

这里不是只初始化第一个元素，而是：

```cpp
0.0f -> 广播到所有 1024 个 lane
```

即：

```cpp
inputData[0]    = 0.0f;
inputData[1]    = 0.0f;
...
inputData[1023] = 0.0f;
```

### 常见用途

常用于清零累加器或临时 buffer：

```cpp
simd<float, 1024> acc = 0;
```

例如矩阵乘法中：

```cpp
acc += a * b;
```

---

## 6. `select<N, Stride>(Offset)`

### API

```cpp
data.select<N, Stride>(Offset)
```

### 要点

`select` 用来从 ESIMD 向量中选择一段子向量。

含义是：

```cpp
从 Offset 开始，
选择 N 个元素，
每次下标增加 Stride
```

### 示例 1：步长为 1

```cpp
inputData.select<64, 1>(j * 64)
```

表示选择：

```cpp
inputData[j * 64 + 0]
inputData[j * 64 + 1]
inputData[j * 64 + 2]
...
inputData[j * 64 + 63]
```

因为步长是 `1`，所以是连续选择。

### 示例 2：步长为 2

```cpp
inputData.select<64, 2>(128)
```

表示选择：

```cpp
inputData[128]
inputData[130]
inputData[132]
...
```

也就是每隔一个元素取一个。

### 示例 3：步长为 4

```cpp
inputData.select<64, 4>(128)
```

表示选择：

```cpp
inputData[128]
inputData[132]
inputData[136]
...
```

也就是每隔 4 个位置取一个。

### 对 `select<64, 1>` 的直观理解

```cpp
inputData.select<64, 1>(j * 64)
```

近似等价于数组切片：

```cpp
inputData[j * 64 : j * 64 + 64]
```

或者：

```cpp
for (int k = 0; k < 64; ++k) {
    inputData[j * 64 + k]
}
```

---

## 7. `block_load<T, N>(ptr)`

### API

```cpp
block_load<T, N>(ptr)
```

### 要点

`block_load` 从连续内存中读取 `N` 个类型为 `T` 的元素，返回一个 ESIMD SIMD 向量。

例如：

```cpp
block_load<float, 64>(inputs + readOffset)
```

表示：

```cpp
从 inputs + readOffset 开始，
连续读取 64 个 float
```

返回结果大致是：

```cpp
simd<float, 64>
```

### 等价理解

```cpp
simd<float, 64> tmp;

for (int k = 0; k < 64; ++k) {
    tmp[k] = inputs[readOffset + k];
}
```

### 注意

`block_load` 是连续内存读取，适合读取一整段连续数据。

---

## 8. `inputData.select<64, 1>(j * 64) = block_load<float, 64>(inputs + readOffset);`

### API

```cpp
inputData.select<64, 1>(j * 64) =
    block_load<float, 64>(inputs + readOffset);
```

### 要点

这句表示：

从 global memory 中连续读取 64 个 `float`，然后写入 `inputData` 的某一段。

右边：

```cpp
block_load<float, 64>(inputs + readOffset)
```

表示：

```cpp
inputs[readOffset + 0]
inputs[readOffset + 1]
...
inputs[readOffset + 63]
```

左边：

```cpp
inputData.select<64, 1>(j * 64)
```

表示：

```cpp
inputData[j * 64 + 0]
inputData[j * 64 + 1]
...
inputData[j * 64 + 63]
```

### 整句等价理解

```cpp
for (int k = 0; k < 64; ++k) {
    inputData[j * 64 + k] = inputs[readOffset + k];
}
```

### 示例

如果：

```cpp
j = 2;
readOffset = 1000;
```

那么：

```cpp
inputData.select<64, 1>(128) =
    block_load<float, 64>(inputs + 1000);
```

等价于：

```cpp
inputData[128] = inputs[1000];
inputData[129] = inputs[1001];
...
inputData[191] = inputs[1063];
```

### 常见用途

常用于把多次连续读取的数据拼进一个大的 SIMD 向量中：

```cpp
simd<float, 1024> inputData = 0;

for (int j = 0; j < 16; ++j) {
    inputData.select<64, 1>(j * 64) =
        block_load<float, 64>(inputs + readOffset);
}
```

如果 `readOffset` 每次增长 64，那么整体就是把连续 1024 个 float 读入 `inputData`。

---

## 9. `block_store<T, N>(ptr, data)`

### API

```cpp
block_store<T, N>(ptr, data);
```

### 要点

`block_store` 把 SIMD 向量中的数据连续写回内存。

例如：

```cpp
block_store<fp16, 256>(outputs + writeOffset, data);
```

表示从 `outputs + writeOffset` 开始，连续写入 256 个 `fp16` 元素。

### 等价理解

```cpp
for (int k = 0; k < 256; ++k) {
    outputs[writeOffset + k] = data[k];
}
```

### 和 `block_load` 对比

```cpp
block_load  : memory -> simd
block_store : simd   -> memory
```

---

## 10. `block_store<fp16, 256>(outputs + writeOffset, outputData.select<256, 1>(j * 256));`

### API

```cpp
block_store<fp16, 256>(
    outputs + writeOffset,
    outputData.select<256, 1>(j * 256)
);
```

### 要点

这句表示：

从 `outputData` 中取出一段连续 256 个 `fp16` 元素，然后写回到 `outputs + writeOffset` 开始的连续内存中。

### 左边内存位置

```cpp
outputs + writeOffset
```

表示写入位置从：

```cpp
outputs[writeOffset]
```

开始。

写入范围：

```cpp
outputs[writeOffset + 0]
outputs[writeOffset + 1]
...
outputs[writeOffset + 255]
```

### 右边 SIMD 子向量

```cpp
outputData.select<256, 1>(j * 256)
```

表示选择：

```cpp
outputData[j * 256 + 0]
outputData[j * 256 + 1]
...
outputData[j * 256 + 255]
```

### 整句等价理解

```cpp
for (int k = 0; k < 256; ++k) {
    outputs[writeOffset + k] = outputData[j * 256 + k];
}
```

### 示例

如果：

```cpp
j = 2;
writeOffset = 10000;
```

那么：

```cpp
outputData.select<256, 1>(512)
```

选中：

```cpp
outputData[512] ~ outputData[767]
```

然后：

```cpp
block_store<fp16, 256>(outputs + 10000, ...)
```

写入：

```cpp
outputs[10000] ~ outputs[10255]
```

### 注意类型

这里使用的是：

```cpp
block_store<fp16, 256>
```

所以 `outputs` 一般应该是：

```cpp
fp16* outputs
```

或者等价的半精度类型指针。

如果 `outputData` 是：

```cpp
simd<fp16, N> outputData;
```

则类型匹配。

如果 `outputData` 是：

```cpp
simd<float, N> outputData;
```

可能需要先转换成 `fp16` 再 store。

---

# 常见模式总结

## 模式 1：从内存读入 SIMD 向量

```cpp
simd<float, 1024> inputData = 0;

inputData.select<64, 1>(j * 64) =
    block_load<float, 64>(inputs + readOffset);
```

含义：

```cpp
从 inputs + readOffset 读取 64 个 float，
写入 inputData[j * 64] ~ inputData[j * 64 + 63]
```

---

## 模式 2：从 SIMD 向量写回内存

```cpp
block_store<fp16, 256>(
    outputs + writeOffset,
    outputData.select<256, 1>(j * 256)
);
```

含义：

```cpp
从 outputData[j * 256] ~ outputData[j * 256 + 255]
取 256 个 fp16，
写入 outputs[writeOffset] ~ outputs[writeOffset + 255]
```

---

## 模式 3：用 `select` 做 SIMD 向量切片

```cpp
data.select<N, 1>(offset)
```

表示连续切片：

```cpp
data[offset]
data[offset + 1]
...
data[offset + N - 1]
```

如果步长不是 1：

```cpp
data.select<N, 2>(offset)
```

表示：

```cpp
data[offset]
data[offset + 2]
data[offset + 4]
...
```

---

# 快速记忆表

| API                            | 作用                           | 类比                                       |
| ------------------------------ | ---------------------------- | ---------------------------------------- |
| `get_local_id(dim)`            | 当前 work-item 在某一维上的 local id | `threadIdx.x/y/z`                        |
| `get_local_linear_id()`        | 当前 work-group 内的一维线性编号       | `threadIdx.y * blockDim.x + threadIdx.x` |
| `simd<float, 1024>`            | 1024 个 float 的 ESIMD 向量      | `float data[1024]`                       |
| `simd<float, 1024> x = 0`      | 1024 个 lane 全部清零             | `memset / for 清零`                        |
| `select<N, S>(offset)`         | 从 SIMD 向量中切出 N 个元素，步长 S      | 数组切片                                     |
| `block_load<T, N>(ptr)`        | 从连续内存读 N 个 T                 | memory → simd                            |
| `block_store<T, N>(ptr, data)` | 向连续内存写 N 个 T                 | simd → memory                            |

---

# 一句话总览

ESIMD 代码里经常是这种模式：

```cpp
simd<float, 1024> inputData = 0;

inputData.select<64, 1>(j * 64) =
    block_load<float, 64>(inputs + readOffset);

block_store<fp16, 256>(
    outputs + writeOffset,
    outputData.select<256, 1>(j * 256)
);
```

可以理解为：

1. 用 `simd<T, N>` 表示一个大的显式 SIMD 向量；
2. 用 `select` 从 SIMD 向量里切片；
3. 用 `block_load` 从连续内存读入 SIMD；
4. 用 `block_store` 把 SIMD 数据连续写回内存；
5. 用 `get_local_id` / `get_local_linear_id` 给 work-item 编号，决定每个 work-item 处理哪一块数据。

