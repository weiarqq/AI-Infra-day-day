# intel 推理平台









| **硬件类型**                 | **核心算子库 / 后端**                              | **主流部署方案**                 | **评价**                                                     |
| ---------------------------- | -------------------------------------------------- | -------------------------------- | ------------------------------------------------------------ |
| **Intel CPU**                | **oneDNN** (MKL-DNN)                               | **OpenVINO**, ONNX Runtime (CPU) | **最优方案**。配合 AMX (Xeon) 或 AVX-512，性能极强，是目前 CPU 推理的标杆。 |
| **Intel iGPU**               | **oneDNN** / **clDNN** (OpenCL)                    | **OpenVINO**                     | **最优方案**。利用集成显卡的执行单元 (EU) 加速，非常适合端侧设备。 |
| **Intel GPU** (Arc/Flex/Max) | **oneDNN**, **Intel Extension for PyTorch (IPEX)** | **OpenVINO**, **DeepSpeed**      | 离散 GPU 推荐使用 IPEX 进行大模型微调与推理，生产环境部署首选 OpenVINO。 |
| **Intel NPU** (Core Ultra)   | **Level Zero** / **NPU Plugin**                    | **OpenVINO**                     | NPU 是低功耗 AI 推理的核心，OpenVINO 是目前唯一成熟的封装方案。 |