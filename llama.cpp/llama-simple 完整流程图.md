llama-simple 完整流程图
┌─────────────────────────────────────────────────────────────────────────────┐
│                           阶段1: 初始化与模型加载                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ggml_backend_load_all()                                                    │
│      │                                                                      │
│      └── 遍历 ggml_backend_registry() 中的所有后端                            │
│           (CPU, GPU, IGPU, RPC 等)                                          │
│           通过 ggml_backend_dev_count() 获取可用设备数量                       │
│                                                                             │
│  llama_model_default_params()                                               │
│      │                                                                      │
│      └── 返回默认模型参数 (n_gpu_layers = 0)                                  │
│           用户可通过 model_params.n_gpu_layers 指定 GPU offload 层数          │
│                                                                             │
│  llama_model_load_from_file(model_path, model_params)                        │
│      │                                                                      │
│      ├── llama_model_loader ml()                                            │
│      │       ├── gguf_init_from_file()  // 解析 GGUF 格式模型文件              │
│      │       ├── 读取模型架构 (arch_name) → LLM_KV() 获取 llm_arch 枚举       │
│      │       ├── 遍历 ggml_get_first_tensor() 创建 tensor 元数据               │
│      │       ├── load_arch()      // 加载模型架构信息                         │
│      │       ├── load_hparams()   // 加载超参数 (config.json)                  │
│      │       ├── load_vocab()     // 加载词表                                │
│      │       ├── load_tensors()   // 创建权重 tensor (仅元数据,不分配内存)      │
│      │       │       └── create_tensor() 创建 ggml_tensor 结构               │
│      │       ├── ml.done_getting_tensors()                                  │
│      │       └── ml.init_mappings()    // 建立内存映射                         │
│      │                                                                      │
│      ├── make_cpu_buft_list()  // 构建 CPU/GPU buffer 列表                    │
│      │                                                                      │
│      └── load_all_data()       // 实际加载权重数据到内存                        │
│              ├── ggml_backend_buffer_alloc() 为每个 tensor 分配 GPU 显存      │
│              └── 使用 mmap 或直接读取方式加载权重数据                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           阶段2: 上下文初始化                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  llama_model_get_vocab(model)                                                │
│      └── 返回词表指针                                                        │
│                                                                             │
│  llama_tokenize(vocab, prompt, ...)                                         │
│      └── 将 prompt 文本转换为 token 序列                                      │
│                                                                             │
│  llama_context_default_params()                                              │
│      └── 获取默认上下文参数                                                  │
│           - n_ctx: 上下文窗口大小                                            │
│           - n_batch: batch 大小                                             │
│           - rope_scaling_type, pool_type 等                                 │
│                                                                             │
│  llama_init_from_model(model, ctx_params)                                   │
│      │                                                                      │
│      ├── 检查模型配置是否有效                                                │
│      │                                                                      │
│      ├── new llama_context                                                  │
│      │       ├── 获取可用后端 ggml_backend_dev_by_type()                     │
│      │       ├── 创建 scheduler: ggml_backend_sched_new()                   │
│      │       │                                                                      │
│      │       └── model.create_memory(ctx_params)  // KV cache               │
│      │               ├── llm_arch_is_recurrent() → llama_memory_recurrent   │
│      │               ├── llm_arch_is_hybrid()   → llama_memory_hybrid        │
│      │               │       ├── llama_kv_cache    // 自注意力 KV           │
│      │               │       └── llama_memory_recurrent  // MoE 状态        │
│      │               └── ggml_backend_alloc_ctx_tensors_from_buft_impl()   │
│      │                       // 为 KV cache 分配 GPU 显存                    │
│      │                                                                      │
│      └── sched_reserve()                                                    │
│              ├── graph_max_nodes()    // 估算计算图最大节点数                 │
│              └── ggml_backend_sched_new() // 创建调度器                      │
│                      └── res->reset()                                       │
│                              └── model.build_graph(gparams)                 │
│                                      ├── 根据 arch 选择对应 LLM_ARCH_*       │
│                                      │   (如 LLM_ARCH_LLAMA, LLM_ARCH_QWEN) │
│                                      └── llm_build_xxx() 构建完整计算图      │
│                                              ├── 初始化 transformer 层       │
│                                              ├── KV cache 更新               │
│                                              └── 输出 logits                 │
│                                                                             │
│  llama_sampler_chain_init() / llama_sampler_chain_add()                      │
│      └── 初始化采样链 (如 greedy 采样)                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           阶段3: 推理循环                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  llama_batch_get_one(prompt_tokens)  // 初始 batch                           │
│                                                                             │
│  while (n_pos < n_prompt + n_predict):                                       │
│      │                                                                      │
│      ├── llama_decode(ctx, batch)  ★ 核心推理 ★                             │
│      │       │                                                              │
│      │       ├── sched_reserve()     // 确保计算图和显存足够                  │
│      │       │                                                                      │
│      │       ├── memory_update()     // 更新 KV cache 状态                   │
│      │       │                                                                      │
│      │       ├── model.build_graph() // 重新构建计算图                        │
│      │       │       └── ggml_backend_sched_reset()                         │
│      │       │               └── ggml_backend_sched_alloc_graph()            │
│      │       │                       // 分配计算图节点                        │
│      │       │                                                                      │
│      │       └── graph_compute(res->get_gf(), batch.n_tokens > 1)            │
│      │               │                                                      │
│      │               └── ggml_backend_sched_graph_compute_async()            │
│      │                       └── ggml_backend_sched_compute_splits()         │
│      │                               └── 按 split 分片并行执行                │
│      │                                       └── ggml_backend_graph_compute()│
│      │                                                                      │
│      ├── llama_sampler_sample(smpl, ctx, -1)  // 采样下一个 token            │
│      │       └── 根据采样策略 (greedy/softmax 等) 选择 token                 │
│      │                                                                      │
│      ├── llama_vocab_is_eog()    // 检查是否结束                             │
│      │                                                                      │
│      ├── llama_token_to_piece()  // token → 文本                            │
│      │                                                                      │
│      └── llama_batch_get_one(&new_token_id, 1)  // 准备下一个 batch          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           阶段4: 资源释放                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  llama_sampler_free(smpl)                                                   │
│  llama_free(ctx)          // 释放上下文、调度器、KV cache                     │
│  llama_model_free(model)  // 释放模型权重和 GGML context                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
关键注释
1. 后端加载 (ggml_backend_load_all)：扫描 build/bin/ 或系统库，发现所有支持的后端设备 (CPU/GPU/RPC等)
2. 模型加载 (llama_model_load_from_file)：
   - GGUF 格式解析：一次性读取整个模型文件到内存
   - 延迟分配：tensor 元数据先创建，实际 GPU 显存按需分配
   - load_all_data() 是实际数据搬运的入口
3. 上下文初始化 (llama_init_from_model)：
   - KV cache 分为 recurrent (循环状态) 和 hybrid (混合模式如 Qwen3.5)
   - build_graph() 在初始化时调用一次，构建静态计算图结构
   - 后续推理时 ggml_backend_sched_reset() + ggml_backend_sched_alloc_graph() 复用结构
4. 推理调度 (llama_decode)：
   - Split 机制：ggml_backend_sched_split_graph() 将大模型拆分到多个 GPU
   - graph_compute_async() 支持异步执行，不阻塞主线程
   - Batch 越大，GPU 利用率越高，但内存占用也越大
5. 采样策略：
   - llama_sampler_chain 支持链式采样 (如重复惩罚 + temperature + top-p)
   - 默认使用 greedy 采样 (llama_sampler_init_greedy())