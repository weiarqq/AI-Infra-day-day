## vLLM







模型分布式部署方案

```python
import torch.multiprocessing as mp

class LLMEngine:

        # 存储分布式进程和事件
        ps = []
        events = []
				# 获取一个新的多进程上下文，使用 "spawn" 方式创建子进程。
        ctx = mp.get_context("spawn")
        # 启动tensor parallel的worker进程（主进程为0号，worker从1开始）
        # tp的多进程管理
        for i in range(1, tensor_parallel_size):
            event = ctx.Event()  # 创建一个进程间同步事件，用于主进程和worker进程之间的通信与同步
            # 创建一个新的worker进程，目标函数是ModelRunner，参数包括配置、进程编号i、同步事件
            process = ctx.Process(target=ModelRunner, args=(config, i, event))  
            process.start()  # 启动该worker进程，让其在后台运行
            self.ps.append(process)  # 将进程对象保存到进程列表，便于后续管理和回收
            self.events.append(event)  # 将事件对象保存到事件列表，便于主进程与各worker通信
        # 主进程的ModelRunner实例
        model_runner = ModelRunner(config, 0, self.events)

  class ModelRunner:
    		# 初始化分布式进程组，使用NCCL后端和TCP通信
    	  dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        torch.set_default_device("cuda")
        load_model(self.model, config.model)
        sampler = Sampler()

        self.allocate_kv_cache()
       
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

                
                
  ctx = mp.get_context("spawn")         
  for
```





### 配置分布式环境

```
import torch.multiprocessing as mp
import torch.distributed as dist

ctx = mp.get_context("spawn") 创建上下文
for 子进程 i (i=1,2,...,并行进程总数-1)   
		事件 = ctx.Event()  创建事件，保证多进程之间同步
		进程 = ctx.Process(ModelRunner, 子进程id, 事件)
				ModelRunner 模型配置
					初始化分布式进程组，使用NCCL后端和TCP通信
					dist.init_process_group("nccl", "tcp://ip:port", 并行进程总数, 当前进程id) 
					torch.cuda.set_device(当前进程id)  设置当前进程的GPU，进程id和GPU id一致
        	torch.set_default_device("cuda")  切换为cuda环境
        	
					load_model 加载模型
					warmup_model 模型预热，主要是为了确定显存峰值时内存占用率
					allocate_kv_cache 分配KV内存，根据系统显存，峰值显存来分配
					
					torch.set_default_device("cpu")   切换到cpu环境
					dist.barrier()  栅栏，等待主进程设置SharedMemory，再子进程设置
          self.shm = SharedMemory(name="nanovllm")  # 其他进程连接共享内存
          self.loop()  # 子进程进入循环等待任务
```



### 模型加载

```
															Qwen3 张量并行整体流程图
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Qwen3ForCausalLM Forward 流程                                   |
│                             以 tp_size=2, num_layers=2 为例                                  ｜
└─────────────────────────────────────────────────────────────────────────────────────────────┘
                              input_ids [batch, seq_len]
                                      │
                                      ▼
                         ┌────────────────────────┐
                         │  VocabParallelEmbedding│
                         │     复制,各rank相同      │
                         └────────────────────────┘
                                      │
                                      ▼
                         hidden_states [batch, seq_len, hidden_size]
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                       ▼
           ┌──────────────────┐                 ┌──────────────────┐
           │     Rank 0       │                 │     Rank 1       │
           │                  │                 │                  │
           │  Layer 1         │                 │  Layer 1         │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │input_ln    │  │                 │  │input_ln    │  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         ▼        │                 │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │qkv_proj    │  │                 │  │qkv_proj    │  │
           │  │ColParallel │  │                 │  │ColParallel │  │
           │  │[h, h/2]    │  │                 │  │[h, h/2]    │  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         ▼        │                 │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │ q,k,v view │  │                 │  │ q,k,v view │  │
           │  │num_heads/2 │  │                 │  │num_heads/2 │  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         ▼        │                 │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │ q_norm     │  │                 │  │ q_norm     │  │
           │  │ k_norm     │  │                 │  │ k_norm     │  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         ▼        │                 │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │ RotaryEmb  │  │                 │  │ RotaryEmb  │  │
           │  │(各rank独立) │  │                 │  │(各rank独立) │  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         ▼        │                 │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │ Attention  │  │                 │  │ Attention  │  │
           │  │ flash_attn │  │                 │  │ flash_attn │  │
           │  │ (部分head)  │  │    ◄─────────►  │  │ (部分head)  │  │
           │  └──────┬─────┘  │   KV Cache      │  └──────┬─────┘  │
           │         ▼        │    交换          │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │ o_proj     │  │                 │  │ o_proj  	  │  │
           │  │RowParallel │  │                 │  │RowParallel	│  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         │        │                 │         │        │
           │    ┌────┴────┐   │                 │    ┌────┴────┐   │
           │    ▼         ▼   │                 │    ▼         ▼   │
           │ ╔═════════════════╧═════════════════════════════════╗ │
           │ ║      🟢 dist.all_reduce (SUM)                     ║ │
           │ ║      合并各 rank 的 attention 输出                	 ║ │
           │ ╚═════════════════╤═════════════════════════════════╝ │
           │                  ▼                                 	 │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │post_attn_ln│  │                 │  │post_attn_ln│  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         ▼        │                 │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │gate_up     │  │                 │  │gate_up     │  │
           │  │ColParallel │  │                 │  │ColParallel │  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         ▼        │                 │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │  SiLU      │  │                 │  │  SiLU      │  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         ▼        │                 │         ▼        │
           │  ┌────────────┐  │                 │  ┌────────────┐  │
           │  │ down_proj  │  │                 │  │ down_proj  │  │
           │  │RowParallel │  │                 │  │RowParallel │  │
           │  └──────┬─────┘  │                 │  └──────┬─────┘  │
           │         │        │                 │         │        │
           │    ┌────┴────┐   │                 │    ┌────┴────┐   │
           │    ▼         ▼   │                 │    ▼         ▼   │
           │ ╔═════════════════╧═════════════════════════════════╗ │
           │ ║      🟢 dist.all_reduce (SUM)                     ║ │
           │ ║        合并各 rank 的 MLP 输出                      ║ │
           │ ╚═════════════════╤═════════════════════════════════╝ │
           │                   ▼                                   │
           │  hidden_states [batch, seq_len, hidden_size]          │
           │                   │                                   │
           └───────────────────┼───────────────────────────────────┘
                              │
                              ▼ (重复 Layer 2...)
                              │
                              ▼
                         ┌────────────┐
                         │ final_norm  │
                         │  (复制)    │
                         └────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  ParallelLMHead    │
                    │ (各 rank 独立计算) │
                    └─────────┬──────────┘
                              │
                              ▼
              ┌───────────────┴───────────────┐
              ▼                               ▼
       ┌─────────────┐               ┌─────────────┐
       │   Rank 0    │               │   Rank 1    │
       │  logits[0]  │               │  logits[1]  │
       │             │               │             │
       │  ┌───────┐  │               │  ┌───────┐  │
       │  │sampler│  │               │  │sampler│  │
       │  │rank 0 │  │               │  │rank 0 │  │
       │  │       │  │               │  │       │  │
       │  └───────┘  │               │  └───────┘  │
       │      │      │               │      │      │
       │      ▼      │               │      ▼      │
       │  token_ids  │               │  (返回None)  │
       └─────────────┘               └─────────────┘
---
```
#### 关键通信点详解
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           All-Reduce 通信时序                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1️⃣ o_proj 之后的 All-Reduce (Attention 输出聚合)                            │
│                                                                                 │
│     Rank 0: o[0] = [1, 2, 3, 4]      Rank 1: o[1] = [5, 6, 7, 8]          │
│           ↓ all_reduce (SUM)                ↓ all_reduce (SUM)               │
│     结果: [1+5, 2+6, 3+7, 4+8] = [6, 8, 10, 12]                           │
│                                                                                 │
│     含义: 各 rank 计算的 half heads 的 O_proj 输出相加 = 完整 hidden_size    │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  2️⃣ down_proj 之后的 All-Reduce (MLP 输出聚合)                               │
│                                                                                 │
│     Rank 0: mlp_out[0] = [1.0, 2.0]    Rank 1: mlp_out[1] = [3.0, 4.0]   │
│           ↓ all_reduce (SUM)                ↓ all_reduce (SUM)              │
│     结果: [1.0+3.0, 2.0+4.0] = [4.0, 6.0]                                 │
│                                                                                 │
│     含义: 各 rank 计算的 half intermediate 的 down_proj 输出相加              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  3️⃣ Gather (采样只在 Rank 0)                                                │
│                                                                                 │
│     Rank 0: logits[0] (完整 vocab)  ──→  Sampler ──→ token_ids             │
│     Rank 1: logits[1] (完整 vocab)  ──→  (丢弃, 不执行 sampler)            │
│                                                                                 │
│     注意: ParallelLMHead 是各 rank 独立计算完整 vocab 的 logits              │
│           但只有 rank 0 执行 sampler 采样                                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
---
```

#### 数据流维度变化

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        维度变化 (假设 hidden=4096, num_heads=32, tp=2)           │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  输入: [batch, seq, 4096]                                                       │
│     │                                                                          │
│     ▼                                                                          │ 
│  qkv_proj (ColumnParallel):                                                    │
│     输入: [batch, seq, 4096]                                                    │
│     输出: [batch, seq, (32+2*2)/2*128] = [batch, seq, 2304]                     │
│     Q: [batch, seq, 16*128]  K: [batch, seq, 2*128]  V: [batch, seq, 2*128]    │
│     (每 rank 持有 16 个 head 的 Q, 2 个 head 的 K/V)                              │
│     │                                                                           │
│     ▼                                                                        │
│  Attention (flash_attn):                                                      │
│     输入: Q[batch, seq, 16, 128], K[batch, seq, 2, 128], V[batch, seq, 2, 128]
│     输出: O[batch, seq, 16, 128]                                              │
│     (每 rank 计算 16 个 head 的 attention)                                    │
│     │                                                                        │
│     ▼                                                                        │
│  o_proj (RowParallel):                                                        │
│     输入: [batch, seq, 16*128] = [batch, seq, 2048]                         │
│     权重: [4096/2, 4096] = [2048, 4096]                                     │
│     输出: [batch, seq, 2048]  ──→ 🟢 all_reduce ──→ [batch, seq, 4096]   │
│     │                                                                        │
│     ▼                                                                        │
│  gate_up_proj (ColumnParallel):                                               │
│     输入: [batch, seq, 4096]                                                 │
│     输出: [batch, seq, 11008*2/2] = [batch, seq, 11008]                     │
│     (每 rank 计算 11008/2 = 5504 维)                                         │
│     │                                                                        │
│     ▼                                                                        │
│  down_proj (RowParallel):                                                     │
│     输入: [batch, seq, 11008/2] = [batch, seq, 5504]                        │
│     权重: [4096/2, 4096] = [2048, 4096]                                     │
│     输出: [batch, seq, 2048]  ──→ 🟢 all_reduce ──→ [batch, seq, 4096]   │
│     │                                                                        │
│     ▼                                                                        │
│  输出: [batch, seq, 4096]                                                    │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
---

```

####  总结
| 位置                                                         | 通信操作           | 数据           | 目的                             |
| ------------------------------------------------------------ | ------------------ | -------------- | -------------------------------- |
| o_proj 后                                                    | all_reduce (SUM)   | attention 输出 | 合并各 rank 的 half heads        |
| down_proj 后                                                 | all_reduce (SUM)   | MLP 输出       | 合并各 rank 的 half intermediate |
| LM Head 后                                                   | 无 (各 rank 独立)  | logits         | 各 rank 计算完整 vocab           |
| Sampler                                                      | gather (仅 rank 0) | token_ids      | 只有 rank 0 执行采样             |
- 每层 Transformer: 2 次 all_reduce
- N 层模型: 2N 次 all_reduce 