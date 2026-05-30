# llama.cpp















### 后端支持

|backend|支持平台|矩阵加速方案|使用场景|环境||
| :--- | :--- | :--- | :--- | :--- | :--- |
|CPU|||||
|CUDA|NVIDIA GPU||||
|BLAS|主要为CPU|Accelerate/Intel MKL/OpenBLAS/|||
|CANN|华为昇腾GPU||||
|hexagon|高通骁龙||||
|hip|AMD GPU||||
|metal|Apple Silicon M系列芯片||||
|musa|摩尔线程芯片||||
|opencl|||||
|openvino|||||
|rpc|||||
|sycl|||||
|virtgpu|||||
|vulkan|||||
|webgpu|||||
|zdnn|||||
|zendnn|||||













| Backend | 平台 | 用途 | 矩阵加速方案 (Kernels) | 构建要求 / 环境 | 使用场景 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CUDA** | NVIDIA GPU | 旗舰级加速 | **cuBLAS**, Tensor Cores (fp16/int8) | CUDA Toolkit (>=12.x) | 算力最强的首选方案，适合所有 NVIDIA 显卡 |
| **Metal** | Apple Silicon | 苹果全家桶 | **MPS**, Apple Neural Engine (ANE) | macOS Xcode / CLI Tools | MacBook, iPad 等移动办公及 Apple 生态 |
| **Vulkan** | 跨平台 GPU | 通用加速 | Compute Shaders, **CoopMat** | Vulkan SDK, 兼容驱动 | AMD/Intel Windows 用户及 Linux 跨显卡混合环境 |
| **SYCL** | Intel GPU | Intel 专用 | **oneMKL**, Intel XMX (Xe Matrix Ext) | DPC++ Compiler (oneAPI) | Intel Arc 独显、Core Ultra 集显及 Max 加速卡 |
| **HIP** | AMD GPU | AMD 原生加速 | **hipBLAS**, rocWMMA (RDNA3+) | ROCm Stack (Linux) | Linux 环境下的 Radeon 与 Instinct 系列加速 |
| **RPC** | 网络分布式 | **跨机器集群** | 远程后端转发 (通过 TCP/RDMA) | `GGML_RPC=ON`, 客户端/服务端模式 | 显存不足时，利用多台旧电脑或服务器组成算力池 |
| **OpenVINO** | Intel 全栈 | AI PC 深度优化 | OpenVINO Graph 翻译, NPU 加速 | OpenVINO Toolkit | 笔记本上的 NPU 推理、Intel 边缘计算节点 |
| **CANN** | 华为昇腾 | 国产算力 | **AscendCL**, NPU 算子 | 华为 CANN 软件栈 | 昇腾 310/910 算力机房、国产国产化替代项目 |
| **BLAS** | 通用 CPU | 基础/辅助 | **OpenBLAS**, Intel MKL, Accelerate | 对应的数学库 (.lib / .so) | 纯 CPU 环境或需要极高精度的非量化任务 |
| **KleidiAI** | Arm64 | 移动/服务器 | **NEON**, SVE / SVE2 优化 | Arm KleidiAI 库集成 | 现代 Android 旗舰手机 (骁龙/天玑) 及 Arm 服务器 |









## 性能分析工具





 #### btop

支持：

- CPU / 内存 / 磁盘 / 网络 全部可视化
- 流畅动画 UI（非常接近 nvtop 风格）
- 每核使用率 + 进程详细信息

安装：

```shell
sudo apt install btop
```



![btop](/Users/wangqi/workspace/workspace/AI-Infra-day-day/llama.cpp/images/btops.jpg)













Llama-simple
