# AMD 推理平台

| **硬件类型**                  | **核心算子库 / 后端**           | **主流部署方案**                         | **评价**                                                     |
| ----------------------------- | ------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| **AMD CPU**                   | **ZenDNN**, **AOCL**            | **ONNX Runtime**, **llama.cpp**          | **ZenDNN** 是专门为 Zen 架构优化的算子库，但在通用性上略逊于 OpenVINO 的 CPU 后端。 |
| **AMD iGPU** (Radeon)         | **ROCm / MiOpen**               | **ONNX Runtime (ROCm)**, **Vulkan**      | 消费级 iGPU 在 Linux 下主要靠 ROCm，Windows 下推荐使用 DirectML 或 Vulkan。 |
| **AMD GPU** (Instinct/Radeon) | **ROCm (MiOpen / RCCL)**        | **vLLM (ROCm)**, **PyTorch**, **Triton** | **最优方案**。ROCm 是 AMD 追赶 CUDA 的核心，目前 vLLM 对 ROCm 的支持已非常成熟，是数据中心部署的首选。 |
| **AMD NPU** (XDNA)            | **Vitis AI** / **XDNA Runtime** | **Ryzen AI Software**, **ONNX Runtime**  | 针对集成在 Ryzen 处理器中的 NPU，必须使用 AMD 提供的专用软件栈。 |