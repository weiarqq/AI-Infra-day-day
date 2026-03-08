## NCCL

### 什么是 NCCL？

**NCCL**（NVIDIA Collective Communications Library，读作 "Nickel"）是 NVIDIA 开发的一套开源库，专门用于多 GPU 之间的高性能通信。

在深度学习训练中，当模型太大无法装入单个 GPU，或者需要通过增加 GPU 数量来缩短训练时间时（即**分布式训练**），不同 GPU 之间需要频繁地交换梯度、权重等数据。NCCL 正是为此而生的“高速翻译官”。

------

### NCCL 的核心优势

为什么在英伟达显卡上训练，大家都默认用 NCCL 而不是 MPI 或其他框架？

1. **拓扑感知（Topology Awareness）**：它能自动探测系统内的硬件连接（如 PCIe 槽位、NVLink、InfiniBand 等），并自动选择最优的路径。
2. **高带宽利用率**：针对 NVLink 进行了深度优化，能够实现接近理论极限的传输速率。
3. **易用性**：它高度集成在 PyTorch (`torch.distributed`)、TensorFlow 和 Horovod 中。
4. **异步执行**：通信操作与计算可以重叠（Overlap），减少显卡的等待时间。

------

### 常见的集体通信操作（Collectives）

NCCL 实现了一系列经典的并行计算原语：

- **AllReduce**：最常用的操作。每个 GPU 都有部分数据（如梯度），AllReduce 将所有数据求和，并把结果分发给所有 GPU。
- **Broadcast**：将数据从一个指定的“根” GPU 拷贝到所有其他 GPU。
- **Reduce**：收集所有 GPU 的数据，进行求和等操作，但只把结果放在“根” GPU 上。
- **AllGather**：每个 GPU 有不同的数据，操作结束后，每个 GPU 都拥有所有 GPU 数据的汇总。
- **ReduceScatter**：Reduce 操作的变体，结果被均匀分散到各个 GPU 上。

------

### 技术架构与原理

#### 1. 环形算法（Ring Algorithm）

在早期的多 GPU 通信中，如果所有 GPU 都向一个中心点发送数据，带宽会迅速成为瓶颈。NCCL 常用 **Ring** 拓扑：

- 所有 GPU 连成一个环。
- 数据被切分成小块（Chunks），在环中依次传递。
- 这种方式使得无论有多少个 GPU，每个 GPU 只需负责与相邻的两个 GPU 通信，极大地提高了带宽利用率。

#### 2. 树形算法（Tree Algorithm）

对于跨节点（多台服务器）的小规模数据传输，环形算法的延迟（Latency）可能较高。NCCL 后来引入了 **Tree** 算法，在某些特定规模下能提供更低的延迟。

#### 3. 硬件支持

NCCL 支持多种底层互联技术：

- **NVLink**：单机内 GPU 之间极速互联。
- **NVSwitch**：更大规模的交换架构。
- **InfiniBand (IB) / RoCE**：多机之间的高性能网络。

------

### NCCL 常见环境变量（调试利器）

如果你在训练模型时遇到卡死或性能不佳，这些环境变量通常是排查问题的关键：

| **变量名**                | **作用**                                              |
| ------------------------- | ----------------------------------------------------- |
| `NCCL_DEBUG=INFO`         | 打印详细日志，显示正在使用的协议（如 NVLink 或 IP）。 |
| `NCCL_P2P_DISABLE=1`      | 禁用点对点通信（在某些 PCIe 兼容性出现问题时使用）。  |
| `NCCL_IB_DISABLE=1`       | 强制禁用 InfiniBand 走以太网。                        |
| `NCCL_SOCKET_IFNAME=eth0` | 指定多网卡环境下用于通信的网卡名称。                  |

------

### 使用建议

如果您正在使用 PyTorch 进行分布式训练，通常只需要简单地指定后端：

```Python
import torch.distributed as dist

# 在初始化分布式环境时指定 nccl
dist.init_process_group(backend='nccl', init_method='env://')
```

**需要注意的是：** NCCL 仅支持 NVIDIA GPU。如果你在 CPU 或 AMD 环境下工作，则需要分别选择 Gloo 或 RCCL。

---

### NCCL 多场景使用示例

NCCL 的使用通常分为两个层面：**深度学习框架集成层**（以 PyTorch 为代表，易用性高）和 **原生 C++ 库调用层**（高性能自定义通信，复杂性高）。

------

### 一、 PyTorch 中的分布式通信 (Python)

PyTorch 的 `torch.distributed` 模块是对 NCCL 的高级封装。在 NVIDIA GPU 环境下，只需指定 `backend='nccl'` 即可。

#### 1. All-Reduce (梯度同步最常用操作)

每个进程（对应一个 GPU）持有一个张量，操作后所有张量变为所有进程原始张量的总和。

```Python
import torch
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 核心：初始化进程组，后端使用 nccl
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def demo_all_reduce(rank, world_size):
    setup(rank, world_size)
    
    # 每个 GPU 准备一个数据
    data = torch.tensor([1.0 + rank]).cuda(rank)
    print(f"Rank {rank} before AllReduce: {data.item()}")

    # 执行 AllReduce (默认操作是求和)
    dist.all_reduce(data, op=dist.ReduceOp.SUM)
    
    print(f"Rank {rank} after AllReduce: {data.item()}")
    dist.destroy_process_group()

if __name__ == "__main__":
    # 假设你有 2 张显卡，通常通过 torchrun 启动
    # torchrun --nproc_per_node=2 your_script.py
    import sys
    world_size = 2
    rank = int(os.environ['RANK'])
    demo_all_reduce(rank, world_size)
```

#### 2. All-Gather (收集所有节点数据)

常用于评估阶段收集所有 GPU 的预测结果。

```Python
def demo_all_gather(rank, world_size):
    setup(rank, world_size)
    tensor = torch.tensor([rank]).cuda(rank)
    
    # 准备一个空列表接收所有 GPU 的数据
    gather_list = [torch.zeros(1).cuda(rank) for _ in range(world_size)]
    
    dist.all_gather(gather_list, tensor)
    
    if rank == 0:
        print(f"Rank 0 gathered: {gather_list}")
    dist.destroy_process_group()
```

------

### 二、 原生 C++ 接口调用 (C++ API)

直接使用 NCCL C++ 库需要处理通信器（Communicator）的生命周期和 CUDA 流（Stream）的管理。这通常用于底层算子开发或自定义并行框架。

#### 1. 单机多 GPU 示例

以下是使用 NCCL C++ API 进行简单 `allReduce` 的核心逻辑框架：

```C++
#include <stdio.h>
#include "nccl.h"
#include <cuda_runtime.h>

#define CHECK_CUDA(cmd) do { \
    cudaError_t res = cmd; \
    if (res != cudaSuccess) { \
        printf("CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(res)); \
        exit(1); \
    } \
} while(0)

#define CHECK_NCCL(cmd) do { \
    ncclResult_t res = cmd; \
    if (res != ncclSuccess) { \
        printf("NCCL Error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(res)); \
        exit(1); \
    } \
} while(0)

int main() {
    int nDevices = 2; // 假设使用 2 个 GPU
    int devs[2] = {0, 1};
    size_t size = 1024 * 1024; // 传输 1M 个 float

    ncclComm_t comms[2];
    cudaStream_t s[2];
    float** sendbuff = (float**)malloc(nDevices * sizeof(float*));
    float** recvbuff = (float**)malloc(nDevices * sizeof(float*));

    // 1. 初始化 NCCL 和 CUDA 流
    CHECK_NCCL(ncclCommInitAll(comms, nDevices, devs));
    
    for (int i = 0; i < nDevices; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&sendbuff[i], size * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&recvbuff[i], size * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&s[i]));
    }

    // 2. 启动集体通信操作 (Group API 可以合并多个操作)
    CHECK_NCCL(ncclGroupStart());
    for (int i = 0; i < nDevices; ++i) {
        CHECK_NCCL(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, 
                                 ncclFloat, ncclSum, comms[i], s[i]));
    }
    CHECK_NCCL(ncclGroupEnd());

    // 3. 同步并清理
    for (int i = 0; i < nDevices; ++i) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(s[i]));
    }

    for (int i = 0; i < nDevices; ++i) {
        ncclCommDestroy(comms[i]);
        cudaFree(sendbuff[i]);
        cudaFree(recvbuff[i]);
    }
    
    printf("Success!\n");
    return 0;
}
```

------

### 三、 关键操作对照表

无论是用 Python 还是 C++，其核心逻辑是一致的，下表对比了常见的操作名称及用途：

| **操作名 (Collective)** | **描述**                       | **应用场景**                       |
| ----------------------- | ------------------------------ | ---------------------------------- |
| **AllReduce**           | 所有人数据求和后再分发给所有人 | 数据并行训练中的梯度同步（最核心） |
| **Broadcast**           | 节点 0 发送数据给所有人        | 初始权重同步、配置分发             |
| **AllGather**           | 所有人拼接彼此的数据           | 模型并行中的特征聚合               |
| **ReduceScatter**       | 先求和，再将结果分块发给不同人 | ZeRO 优化器（Deepspeed 核心）      |
| **Send/Recv**           | 点对点传输                     | 流水线并行 (Pipeline Parallelism)  |

### 进阶提示

- **编译 C++ 代码**：需要链接库文件，通常使用 `nvcc example.cpp -o example -lnccl -lcudart`。
- **多机环境**：如果是多机（Multi-node），C++ 需要配合 `ncclGetUniqueId` 并在各节点间通过 Socket 共享该 ID 才能完成初始化；PyTorch 则通过 `MASTER_ADDR` 自动处理了这一过程。