### ggml 对象



#### ggml_tallocr





## Tensor Representation

#### ggml_tensor

```c++
struct ggml_tensor {
        enum ggml_type type;
        struct ggml_backend_buffer * buffer; 
        int64_t ne[GGML_MAX_DIMS]; 
        size_t  nb[GGML_MAX_DIMS]; 
        enum ggml_op op;
        int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
        int32_t flags;
        struct ggml_tensor * src[GGML_MAX_SRC];
        struct ggml_tensor * view_src;
        size_t               view_offs;
        void * data;
        char name[GGML_MAX_NAME];
        void * extra; 
        char padding[8];
    };
```

**1. 基础类型与存储相关**

| 变量名              | 类型                          | 含义与作用                                                   |
| ------------------- | ----------------------------- | ------------------------------------------------------------ |
| `type`              | `enum ggml_type`              | 张量的数据类型，决定了存储格式和计算方式。<br>例如：<br>- `GGML_TYPE_F32`：32位浮点数<br>- `GGML_TYPE_Q4_0`：4位量化类型（压缩存储）<br>- `GGML_TYPE_BF16`：bfloat16浮点数<br>支持量化类型（如Q系列）、整数类型（I系列）和浮点数类型，共40种（`GGML_TYPE_COUNT = 40`）。 |
| `buffer`            | `struct ggml_backend_buffer*` | 指向后端缓冲区（如CPU、CUDA、Metal等硬件的内存缓冲区），用于管理张量数据的存储位置（不同硬件的内存分配）。 |
| `ne[GGML_MAX_DIMS]` | `int64_t`                     | 张量各维度的元素数量（shape），`GGML_MAX_DIMS` 为最大维度数（通常为4）。<br>例如：2D张量 `ne[0] = 3, ne[1] = 4` 表示3列4行。 |
| `nb[GGML_MAX_DIMS]` | `size_t`                      | 各维度的**字节步长**（stride），用于计算内存中元素的偏移量，支持非连续内存布局（如切片、转置）。<br>规则：<br>- `nb[0]` 为单个元素的字节数（由 `type` 决定）<br>- `nb[i] = nb[i-1] * ne[i-1]`（高维步长基于低维计算，可能包含填充）。 |
| `data`              | `void*`                       | 指向张量的原始数据内存地址。对于量化类型，数据按对应格式压缩存储；对于视图（view）张量，可能指向源张量的 `data` 偏移位置。 |

**2. 计算图与运算相关**

| 变量名              | 类型                                            | 含义与作用                                                   |
| ------------------- | ----------------------------------------------- | ------------------------------------------------------------ |
| `op`                | `enum ggml_op`                                  | 张量关联的运算类型，表示当前张量是某个运算的输出。<br>例如：<br>- `GGML_OP_ADD`：加法运算<br>- `GGML_OP_MUL_MAT`：矩阵乘法<br>- `GGML_OP_RMS_NORM`：RMS归一化<br>支持超过50种运算（如算术、卷积、激活函数、注意力机制等）。 |
| `op_params`         | `int32_t[GGML_MAX_OP_PARAMS / sizeof(int32_t)]` | 运算的参数数组，长度由 `GGML_MAX_OP_PARAMS` 限制，存储运算所需的额外配置。<br>例如：<br>- 卷积运算的核大小、步长<br>- 归一化的epsilon参数<br>- 激活函数的类型（如 `GGML_UNARY_OP_RELU`）。 |
| `flags`             | `int32_t`                                       | 张量的属性标志，通过位运算组合，定义张量在计算图中的角色：<br>- `GGML_TENSOR_FLAG_INPUT`：计算图的输入张量<br>- `GGML_TENSOR_FLAG_OUTPUT`：计算图的输出张量<br>- `GGML_TENSOR_FLAG_PARAM`：可训练的参数张量（如权重）<br>- `GGML_TENSOR_FLAG_LOSS`：损失函数张量（用于优化）。 |
| `src[GGML_MAX_SRC]` | `struct ggml_tensor*`                           | 运算的输入张量列表，`GGML_MAX_SRC` 为最大输入数（通常为2或3）。<br>例如：加法运算 `a + b` 中，`src[0] = a`，`src[1] = b`。 |

**3. 视图（View）机制相关**

| 变量名      | 类型                  | 含义与作用                                                   |
| ----------- | --------------------- | ------------------------------------------------------------ |
| `view_src`  | `struct ggml_tensor*` | 若当前张量是某个张量的**视图**（无需复制数据的子集），则指向源张量。<br>例如：对张量切片后得到的新张量，`view_src` 指向原始张量。 |
| `view_offs` | `size_t`              | 视图在源张量数据中的**字节偏移量**，即 `data = view_src->data + view_offs`，用于定位视图数据在源张量中的起始位置。 |

**4. 辅助信息**

| 变量名    | 类型                  | 含义与作用                                                   |
| --------- | --------------------- | ------------------------------------------------------------ |
| `name`    | `char[GGML_MAX_NAME]` | 张量的名称（可选），用于调试、标识或模型加载时的张量匹配（如 `blk.0.attn_q.weight`）。 |
| `extra`   | `void*`               | 额外数据指针，用于硬件后端（如CUDA、Metal）的扩展信息（如设备端内存句柄、优化参数等）。 |
| `padding` | `char[8]`             | 结构体填充字节，确保 `ggml_tensor` 按内存对齐要求（如64位对齐）分配，避免内存访问错误。 |









ggml_type_traits_cpu







ggml_context







ggml_backend

| Opaque Handle                | Concrete Struct            | Interface Struct             | Role                                                        |
| ---------------------------- | -------------------------- | ---------------------------- | ----------------------------------------------------------- |
| `ggml_backend_reg_t`         | `ggml_backend_reg`         | `ggml_backend_reg_i`         | A compiled-in or dynamically loaded backend plugin.         |
| `ggml_backend_dev_t`         | `ggml_backend_dev`         | `ggml_backend_dev_i`         | A specific physical device (e.g., NVIDIA GeForce RTX 4090). |
| `ggml_backend_buffer_type_t` | `ggml_backend_buffer_type` | `ggml_backend_buffer_type_i` | Factory for allocating memory on a specific device.         |
| `ggml_backend_buffer_t`      | `ggml_backend_buffer`      | `ggml_backend_buffer_i`      | A concrete allocated region of device memory.               |
| `ggml_backend_t`             | `ggml_backend`             | `ggml_backend_i`             | A running compute context (stream/queue) bound to a device. |







### ggml_cgraph