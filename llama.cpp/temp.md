ggml_backend_load_all()

加载backends



llama_model_load_from_file()

加载模型



llama_model_get_vocab()

加载vocab

llama_tokenize()

Tokenizer



llama_model_default_params()



llama_context_default_params()



llama_init_from_model() *****

​	llama_context()



llama_sampler_chain_default_params()





llama_decode()







加载硬件后端

ggml_backend_load_all()





加载模型文件

llama_model_load_from_file()

​	model.load_tensors

解析 GGUF 文件，加载权重到内存，处理 Q4_K_M、IQ4_XS 等量化格式



llama_model_get_vocab()

获取词表，用于后续分词



llama_tokenize()

将 prompt 转为 token 数组，支持 SPM/BPE/WPM 三种分词算法





初始化上下文

llama_init_from_model()





创建推理上下文，初始化 KV cache、计算图、线程池





第三阶段：主推理循环（核心）
3.1 计算图构建 — src/llama-graph.cpp
图谱 line 11275 描述：神经网络在此被组装为 GGML 计算图
关键类：

- llm_graph_context — 架构特定的图构建上下文，持有 layer weights 和中间 tensor 引用
- llm_graph_result — 图构建结果容器
每种 layer 类型通过一系列 ggml_* 调用构建：
build_norm()   // RMSNorm 等归一化层
build_ffn()     // Feed-Forward 网络
build_attn()    // Attention 层（包含 KV cache 读写）
build_inp_embd() // Input embedding
3.2 矩阵运算核心 — ggml_mul_mat
图谱 line 11283：96 种 tensor 操作，核心模式：
ggml_* 创建 tensors（仅构建图，不计算）
       ↓
ggml_build_forward_expand()  最终化图
       ↓
ggml_graph_compute()          执行
3.3 KV Cache 管理 — llama_kv_cache
图谱 line 4486：KV cache 管理，支持滑动窗口注意力
关键操作：
- find_slot() — 为 batch 分配 KV slot
- update() — 更新 KV cache
- cpy_k() / cpy_v() — 复制 K/V tensor
- get_k() / get_v() — 获取 KV tensor
- 
- 第四阶段：
步骤	函数
采样	llama_sampler_sample()
EOS 检测	llama_vocab_is_eog()
转换为文本	llama_token_to_piece()
图谱 line 11324：llama-sampler.cpp 实现了丰富的采样链（greedy、multinomial、temperature、top-k、top-p、penalty 等）
第五阶段：清理
llama_sampler_free(smpl);
llama_free(ctx);           // 释放推理上下文
llama_model_free(model);   // 释放模型权重

---



采样与输出

llama_sampler_sample()

