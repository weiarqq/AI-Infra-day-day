



| CUDA          | Intel                              |
| ------------- | ---------------------------------- |
| GPU           | GPU                                |
| SM            | Xe-core / Subslice / Dual Subslice |
| CUDA core     | SIMD lane / Vector ALU lane        |
| Tensor Core   | XMX / Matrix Engine                |
| Warp          | Sub-group                          |
| Warp lane     | SIMD channel                       |
| warp lane id       | SIMD channel(SIMD channel = SIMD lane = vector lane) / sub-group local id |
| Thread        | Work-item                          |
| Shared memory | SLM / local memory                 |
| Register file | GRF                                |
| block              | work-group                        |
| blockDim.x         | work-group size                   |
| threadIdx.x        | work-group local id               |
| `threadIdx.x % 32` | `sub_group.get_local_linear_id()` |


GRF = General Register File，通用寄存器文件
一个 GRF = 一个寄存器
DW = Double Word = 32 bit = 4 byte
DW lane = 这个 GRF 里的第几个 32-bit 槽位
SIMD channel = 一条 SIMD 指令里的第几个 lane
