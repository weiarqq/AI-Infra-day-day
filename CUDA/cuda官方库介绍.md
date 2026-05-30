## CUDA API References





cuBLAS





Tensor Core



cute dsl



CUTLASS

> CUTLASS本质上不是一个二进制库，而是一套头文件库。
>
> 它更像是一个**内核开发框架**：
>
> - **它是“积木”**：它提供了线程块（Threadblock）、线程束（Warp）、线程（Thread）各个层级的 GEMM 实现细节。
> - **它是“规范”**：它定义了数据怎么从显存搬到共享内存（Shared Memory），再怎么喂给 Tensor Core 的标准路径。
> - **它是“生成器”**：当你写下 `Gemm<128, 64, 32>` 时，编译器会根据这些模板参数，为你“量身定制”一段汇编代码。



FMA/WMMA/MMA

| **特性**       | **FMA**                                                     | **WMMA**           | **MMA (PTX)**                       |
| -------------- | ----------------------------------------------------------- | ------------------ | ----------------------------------- |
| **计算对象**   | 标量 (1x1)                                                  | 矩阵 (如 16x16)    | 矩阵 (如 16x8)                      |
| **硬件载体**   | **CUDA Core**                                               | **Tensor Core**    | **Tensor Core**                     |
| **开发者层级** | 基础 (所有程序员)                                           | 中级 (调用 API)    | 高级 (算子开发)                     |
| **效率**       | 1倍 (基准)                                                  | ~8x-16x (针对矩阵) | ~8x-16x (针对矩阵)                  |
| **大模型用途** | 处理非矩阵运算（如激活函数、LayerNorm、Softmax 的指数部分） | 基本不用           | **核心：处理 Attention 和全连接层** |



NCCL



NVSHMEM



拓扑感知



NVIDIA Nsight Systems / Compute



Roofline Model



