
# vLLM 完整架构分析

## 1. 整体架构概览
---



```
┌─────────────────────────────────────────────────────────────────┐
│                        Entry Points                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐ │
│  │  LLM (离线推理)  │  │ AsyncLLMEngine │  │  OpenAI API Server │ │
│  │                 │  │  (在线服务)      │  │  (FastAPI)        │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘  │
└───────────┼─────────────────────┼────────────────────┼───────────┘
            │                     │                    │
            ▼                     ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Engine Layer                               │
│  ┌─────────────────────────┐    ┌────────────────────────────┐ │
│  │      V0 Engine          │    │        V1 Engine             │ │
│  │  ┌───────────────────┐  │    │  ┌──────────────────────┐    │ │
│  │  │   LLMEngine       │  │    │  │    LLMEngine (V1)    │    │ │
│  │  │  (同步/异步引擎)    │  │    │  └──────────┬───────────┘    │ │
│  │  └───────────────────┘  │    │             ▼                │ │
│  └─────────────────────────┘    │  ┌────────────────────────┐  │ │
│                                 │  │     EngineCore         │  │ │
│                                 │  │  ┌──────────────────┐  │  │ │
│                                 │  │  │  Processor       │  │  │ │
│                                 │  │  │  Scheduler       │  │  │ │
│                                 │  │  │  ModelExecutor   │  │  │ │
│                                 │  │  │  KVCacheManager  │  │  │ │
│                                 │  │  │  Detokenizer     │  │  │ │
│                                 │  │  │  OutputProcessor │  │  │ │
│                                 │  │  └──────────────────┘  │  │ │
│                                 │  └────────────────────────┘  │ │
│                                 └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 2. 核心模块详细架构

### 2.1 入口层 (vllm/entrypoints/)

```
┌────────────────────────────────────────────────────────┐
│                   entrypoints/                          │
│  ┌──────────────────────────────────────────────────┐ │
│  │  llm.py                                           │ │
│  │    └── LLM (主入口类，同步/异步推理)               │ │
│  │         └── LLM.__init__()                        │ │
│  │              └── AsyncLLMEngine._from_engine()    │ │
│  ├──────────────────────────────────────────────────┤ │
│  │  api_server.py                                    │ │
│  │    └── create_app() → FastAPI                    │ │
│  ├──────────────────────────────────────────────────┤ │
│  │  serving_chat.py                                  │ │
│  │    └── OpenAIServingChat                         │ │
│  │         ├── serve_chat_completion()              │ │
│  │         └── chat_templatejinja_env               │ │
│  ├──────────────────────────────────────────────────┤ │
│  │  serving_completion.py                            │ │
│  │    └── OpenAIServingCompletion                   │ │
│  ├──────────────────────────────────────────────────┤ │
│  │  serving_embedding.py                             │ │
│  │    └── OpenAIServingEmbedding                    │ │
│  ├──────────────────────────────────────────────────┤ │
│  │  serving_pooling.py                              │ │
│  │    └── OpenAIServingPooling                      │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### 2.2 V0 引擎层 (vllm/engine/)

```
┌─────────────────────────────────────────────────────────────────┐
│                        vllm/engine/                              │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  llm_engine.py                                             │ │
│  │    ┌─────────────────┐    ┌─────────────────────────────┐   │ │
│  │    │ LLMEngine      │    │ SchedulerOutputState        │   │ │
│  │    │ (同步引擎核心)  │    │ (调度输出状态，Multi-Step)  │   │ │
│  │    │                 │    └─────────────────────────────┘   │ │
│  │    │ - add_request() │    ┌─────────────────────────────┐   │ │
│  │    │ - step()        │    │ SchedulerContext            │   │ │
│  │    │ - _run_workers()│    │ (调度上下文)                │   │ │
│  │    └────────┬────────┘    └─────────────────────────────┘   │ │
│  │             │                                               │ │
│  │             ▼                                               │ │
│  │    ┌─────────────────────────────────────────┐              │ │
│  │    │           LLMEngine Components:         │              │ │
│  │    │  ┌────────────────┐ ┌────────────────┐  │              │ │
│  │    │  │InputPreprocessor│ │OutputProcessor│  │              │ │
│  │    │  │ (输入预处理)    │ │ (输出处理)     │  │              │ │
│  │    │  └────────────────┘ └────────────────┘  │              │ │
│  │    │  ┌────────────────┐ ┌────────────────┐  │              │ │
│  │    │  │   Scheduler   │ │ ModelExecutor │  │              │ │
│  │    │  │   (调度器)     │ │ (模型执行器)  │  │              │ │
│  │    │  └────────────────┘ └────────────────┘  │              │ │
│  │    └─────────────────────────────────────────┘              │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  async_llm_engine.py                                       │ │
│  │    ┌─────────────────┐    ┌─────────────────────────────┐   │ │
│  │    │ AsyncLLMEngine │    │  _AsyncLLMEngine           │   │ │
│  │    │ (异步引擎)      │    │  (内部实现)                 │   │ │
│  │    └─────────────────┘    └─────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 V1 引擎层 (vllm/v1/engine/)

```
┌─────────────────────────────────────────────────────────────────┐
│                       vllm/v1/engine/                           │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  llm_engine.py                                            │  │
│  │    └── LLMEngine (V1 向后兼容入口)         	               │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  core.py                                                  │  │
│  │    └── EngineCore (V1 核心，包含主循环)                      │ │
│  │         ├── step() → SchedulerOutput                       │ │
│  │         └── execute_model()                                │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │  core_client.py                                            │ │
│  │    └── EngineCoreClient (EngineCore 客户端封装)           │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │  processor.py                                              │ │
│  │    └── Processor (输入处理)                                │ │
│  │         ├── process_request()                              │ │
│  │         └── _convert_inputs()                              │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │  output_processor.py                                       │ │
│  │    └── OutputProcessor (输出处理)                          │ │
│  │         └── process_outputs()                              │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │  detokenizer.py                                           │ │
│  │    └── Detokenizer (解码器)                                │ │
│  │         └── step()                                         │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 调度器层

**V0 调度器 (vllm/core/scheduler.py):**

```
┌────────────────────────────────────────────────────────┐
│              vllm/core/scheduler.py                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Scheduler (V0 调度器)                            │ │
│  │    ├── schedule() → SchedulerOutputs              │ │
│  │    │    ├── running_outputs (运行中)               │ │
│  │    │    ├── prefill_outputs (预填充)              │ │
│  │    │    └── swapped_outputs (换出)                 │ │
│  │    ├── _schedule_prefills()                       │ │
│  │    ├── _schedule_running()                        │ │
│  │    └── _schedule_swapped()                        │ │
│  │                                                    │ │
│  │  Data Classes:                                    │ │
│  │    ├── SchedulerOutputs                           │ │
│  │    ├── SchedulerRunningOutputs                    │ │
│  │    ├── SchedulerPrefillOutputs                    │ │
│  │    ├── ScheduledSequenceGroup                     │ │
│  │    ├── SchedulingBudget                           │ │
│  │    └── PreemptionMode (枚举)                      │ │
│  └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

**V1 调度器 (vllm/v1/core/scheduler.py):**

```
┌────────────────────────────────────────────────────────┐
│           vllm/v1/core/scheduler.py                     │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Scheduler (V1 调度器)                            │ │
│  │    ├── schedule() → SchedulerOutput              │ │
│  │    ├── add_request()                              │ │
│  │    ├── abort_request()                            │ │
│  │    └── can_allocate()                             │ │
│  │                                                    │ │
│  │  vllm/v1/core/scheduler_output.py                 │ │
│  │    ├── SchedulerOutput                            │ │
│  │    ├── CachedRequestData                          │ │
│  │    └── NewRequestData                             │ │
│  └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### 2.5 块管理/缓存层

**V0 块管理 (vllm/core/block/):**

```
┌────────────────────────────────────────────────────────┐
│              vllm/core/block/                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  interfaces.py                                     │ │
│  │    └── BlockSpaceManager (接口)                    │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  block_manager.py                                  │ │
│  │    └── SelfAttnBlockSpaceManager                  │ │
│  │         ├── allocate()                            │ │
│  │         └── can_allocate()                        │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  block_allocator.py                               │ │
│  │    ├── BlockAllocator (基类)                      │ │
│  │    ├── PrefixCachingBlockAllocator               │ │
│  │    └── NaiveBlockAllocator                        │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  block_table.py                                   │ │
│  │    └── BlockTable                                 │ │
│  └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

**V1 块管理 (vllm/v1/core/):**

```
┌────────────────────────────────────────────────────────┐
│              vllm/v1/core/                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │  kv_cache_manager.py                               │ │
│  │    └── KVCacheManager (KV缓存管理器)               │ │
│  │         ├── allocate()                            │ │
│  │         ├── free()                                │ │
│  │         └── can_allocate()                         │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  block_pool.py                                     │ │
│  │    └── BlockPool (块池)                            │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  kv_cache_utils.py                                │ │
│  │    ├── KVCacheBlock                               │ │
│  │    └── FreeKVCacheBlockQueue                      │ │
│  └────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

### 2.6 Worker/执行器层

**V0 执行器 (vllm/executor/):**

```
┌────────────────────────────────────────────────────────┐
│                 vllm/executor/                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  executor_base.py                                  │ │
│  │    └── ExecutorBase (基类)                        │ │
│  │         ├── execute_model()                       │ │
│  │         └── collective_rendezvous()               │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  uniproc_executor.py                              │ │
│  │    └── UniProcExecutor (单进程)                   │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  mp_distributed_executor.py                       │ │
│  │    └── MPDistributedExecutor (多进程)              │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  ray_distributed_executor.py                      │ │
│  │    └── RayDistributedExecutor (Ray)               │ │
│  └────────────────────────────────────────────────────┘ │
```

**V1 执行器 (vllm/v1/executor/):**

```
┌────────────────────────────────────────────────────────┐
│              vllm/v1/executor/                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  abstract.py                                       │ │
│  │    └── Executor (V1 执行器基类)                   │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  uniproc_executor.py                              │ │
│  │    └── UniProcExecutor                            │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  multiproc_executor.py                            │ │
│  │    └── MultiprocExecutor                         │ │
│  └────────────────────────────────────────────────────┘ │
```

**Worker 层 (vllm/worker/):**

```
┌────────────────────────────────────────────────────────┐
│                   vllm/worker/                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  worker_base.py                                    │ │
│  │    └── WorkerBase (基类)                          │ │
│  │         │                                          │ │
│  │         ▼                                          │ │
│  │    LocalOrDistributedWorkerBase                    │ │
│  │         │                                          │ │
│  │    ┌────┴────┬────────┬──────┬──────┬─────────┐   │ │
│  │    ▼         ▼        ▼      ▼      ▼         ▼     │ │
│  │  Worker  CPUWorker TPUWorker HPUWorker XPUWorker   │ │
│  │  (GPU)                                           │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  model_runner_base.py                             │ │
│  │    └── ModelRunnerBase (模型运行基类)            │ │
│  │         │                                          │ │
│  │         ▼                                          │ │
│  │    GPUModelRunnerBase                              │ │
│  │         │                                          │ │
│  │         ▼                                          │ │
│  │    ModelRunner (V0 GPU)                           │ │
│  │         ├── execute_model()                       │ │
│  │         └── profile_model()                       │ │
│  └────────────────────────────────────────────────────┘ │
```

**V1 Worker (vllm/v1/worker/):**

```
┌────────────────────────────────────────────────────────┐
│                vllm/v1/worker/                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │  gpu_worker.py                                     │ │
│  │    └── Worker (V1 GPU Worker)                     │ │
│  │         └── execute_model()                        │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  gpu_model_runner.py                              │ │
│  │    └── GPUModelRunner (V1 GPU 模型运行器)        │ │
│  │         ├── execute_model()                       │ │
│  │         ├── load_model()                          │ │
│  │         └── make_mm_graph()                       │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  gpu_input_batch.py                               │ │
│  │    ├── InputBatch (输入批次)                      │ │
│  │    └── CachedRequestState (缓存请求状态)          │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  tpu_model_runner.py                              │ │
│  │    └── TPUModelRunner                             │ │
│  └────────────────────────────────────────────────────┘ │
```

### 2.7 模型执行层 (vllm/model_executor/)

```
┌─────────────────────────────────────────────────────────────────┐
│                    vllm/model_executor/                         │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  model_loader.py                                           │ │
│  │    ├── ModelRegistry (模型注册表)                         │ │
│  │    ├── BaseModelLoader (基类)                              │ │
│  │    └── get_model() (加载函数)                              │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │  models/ (模型实现)                                        │ │
│  │    ├── llama.py (LLaMA)                                   │ │
│  │    ├── qwen.py (Qwen)                                     │ │
│  │    ├── mistral.py (Mistral)                               │ │
│  │    └── ... (其他模型)                                     │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │  layers/ (层实现)                                          │ │
│  │    ├── sampler.py (Sampler - 采样层)                      │ │
│  │    ├── logits_process.py (LogitsProcessor)               │ │
│  │    ├── rotary_embedding.py (RotaryEmbedding)            │ │
│  │    ├── quantization/ (量化层)                             │ │
│  │    │    ├── awq.py                                       │ │
│  │    │    ├── gptq.py                                      │ │
│  │    │    └── fp8.py                                       │ │
│  │    └── ... (其他层)                                       │ │
│  ├───────────────────────────────────────────────────────────┤ │
│  │  pooling.py                                                │ │
│  │    └── PoolerOutput                                       │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 2.8 分布式模块 (vllm/distributed/)

```
┌─────────────────────────────────────────────────────────┐
│                vllm/distributed/                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │  parallel_state.py                                 │ │
│  │    ├── GroupCoordinator (进程组协调器)               │ │
│  │    ├── TensorParallelGroup (TP)                    │ │
│  │    ├── PipelineParallelGroup (PP)                  │ │
│  │    ├── DataParallelGroup (DP)                      │ │
│  │    ├── get_tp_group()                              │ │
│  │    ├── get_pp_group()                              │ │
│  │    └── get_dp_group()                              │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  device_communicators/                             │ │
│  │    ├── communicator_base.py                        │ │
│  │    │    └── DeviceCommunicatorBase                 │ │
│  │    ├── cuda_communicator.py                        │ │
│  │    │    └── CUDACommunicator                       │ │
│  │    ├── cpu_communicator.py                         │ │
│  │    ├── tpu_communicator.py                         │ │
│  │    └── hpu_communicator.py                         │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  custom_all_reduce.py                              │ │
│  │    ├── CustomAllReduce                             │ │
│  │    └── CustomAllReduceConstants                    │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 2.9 输入/输出处理

**输入处理 (vllm/inputs/):**

```
┌─────────────────────────────────────────────────────────┐
│                    vllm/inputs/                    	    │
│  ┌────────────────────────────────────────────────────┐ │
│  │  preprocess.py                                     │ │
│  │    └── InputPreprocessor (输入预处理器)           		│ │
│  │         ├── preprocess()                          	│ │
│  │         └── _preprocess_vision()                 	│ │
│  ├────────────────────────────────────────────────────┤ │
│  │  registry.py                                       │ │
│  │    └── InputRegistry (输入注册表)                		 │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  data.py                                           │ │
│  │    ├── ProcessorInputs                            	│ │
│  │    └── SingletonInputsAdapter                     	│ │
│  └────────────────────────────────────────────────────┘ │
```

**输出数据结构 (vllm/outputs.py):**

```
┌────────────────────────────────────────────────────────┐
│                    vllm/outputs.py                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │  RequestOutput (请求输出)                          │ │
│  │    ├── request_id                                 │ │
│  │    ├── outputs: List[CompletionOutput]            │ │
│  │    └── prompt_token_ids                           │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  CompletionOutput (完成输出)                      │ │
│  │    ├── index                                      │ │
│  │    ├── text                                       │ │
│  │    ├── token_ids                                  │ │
│  │    └── logprobs                                   │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  EmbeddingRequestOutput                            │ │
│  │  PoolingRequestOutput                             │ │
│  └────────────────────────────────────────────────────┘ │
```

**V1 输出 (vllm/v1/outputs.py):**

```
┌────────────────────────────────────────────────────────┐
│                vllm/v1/outputs.py                       │
│  ┌────────────────────────────────────────────────────┐ │
│  │  ModelRunnerOutput (模型运行器输出)                │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  SamplerOutput (采样器输出)                        │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  LogprobsTensors (Logprobs 张量)                   │ │
│  └────────────────────────────────────────────────────┘ │
```

### 2.10 核心数据结构

**序列管理 (vllm/sequence.py):**

```
┌────────────────────────────────────────────────────────┐
│                   vllm/sequence.py                      │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Sequence (单个序列)                                │ │
│  │    ├── seq_id                                      │ │
│  │    ├── data: SequenceData                         │ │
│  │    └── block_table                                │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  SequenceGroup (序列组)                           │ │
│  │    ├── request_id                                 │ │
│  │    ├── sequences: List[Sequence]                  │ │
│  │    └── sampling_params                           │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  SequenceData (序列数据)                          │ │
│  │    ├── token_ids                                  │ │
│  │    └── prompt_token_ids                           │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  SequenceGroupMetadata                            │ │
│  │  ExecuteModelRequest (执行模型请求)               │ │
│  │  SamplerOutput (采样器输出)                       │ │
│  └────────────────────────────────────────────────────┘ │
```

**采样参数 (vllm/sampling_params.py):**

```
┌────────────────────────────────────────────────────────┐
│               vllm/sampling_params.py                  │
│  ┌────────────────────────────────────────────────────┐ │
│  │  SamplingParams (采样参数)                        │ │
│  │    ├── temperature                                │ │
│  │    ├── top_p                                      │ │
│  │    ├── max_tokens                                 │ │
│  │    ├── n (候选数量)                               │ │
│  │    └── ...                                        │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  SamplingType (采样类型枚举)                      │ │
│  │    ├── GREEDY                                     │ │
│  │    ├── RANDOM                                     │ │
│  │    └── beam search                                │ │
│  └────────────────────────────────────────────────────┘ │
```

### 2.11 配置模块 (vllm/config.py)

```
┌────────────────────────────────────────────────────────┐
│                    vllm/config.py                       │
│  ┌────────────────────────────────────────────────────┐ │
│  │  ModelConfig (模型配置)                           │ │
│  │    ├── hf_config                                  │ │
│  │    ├── dtype                                      │ │
│  │    └── sliding_window                             │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  CacheConfig (缓存配置)                           │ │
│  │    ├── block_size                                 │ │
│  │    ├── gpu_memory_utilization                     │ │
│  │    └── kv_cache_dtype                             │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  ParallelConfig (并行配置)                        │ │
│  │    ├── tensor_parallel_size                       │ │
│  │    ├── pipeline_parallel_size                     │ │
│  │    └── data_parallel_size                         │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  SchedulerConfig (调度器配置)                     │ │
│  │    ├── max_num_seqs                               │ │
│  │    └── max_model_len                              │ │
│  ├────────────────────────────────────────────────────┤ │
│  │  DeviceConfig (设备配置)                          │ │
│  │  LoRAConfig (LoRA 配置)                           │ │
│  │  SpeculativeConfig (投机解码配置)                 │ │
│  │  VllmConfig (统一配置包装)                        │ │
│  └────────────────────────────────────────────────────┘ │
```

### 2.12 其他重要模块

**Attention 后端 (vllm/attention/):**

```
vllm/attention/
    ├── __init__.py
    ├── backends/
    │    ├── flash_attn.py (FlashAttention)
    │    ├── flash_attn_2.py
    │    ├── flash_attn_3.py
    │    ├── xformers.py
    │    └── ... (其他后端)
    └── layer.py
```

**多模态 (vllm/multimodal/):**

```
vllm/multimodal/
    ├── registry.py (多模态注册表)
    ├── base.py (基类)
    ├── image.py (图像处理)
    └── video.py (视频处理)
```

**LoRA (vllm/lora/):**

```
vllm/lora/
    ├── lora.py (LoRA 实现)
    ├── layers.py (LoRA 层)
    └── worker.py (LoRA Worker)
```

**投机解码 (vllm/spec_decode/):**

```
vllm/spec_decode/
    ├── spec_decode_worker.py
    ├── metrics.py
    └── draft_model_executor.py
```

---

## 3. 完整数据流图

```
用户请求
    │
    ▼
┌─────────────────┐
│  LLM / API      │ ────────────────────────► OpenAI API Response
│  Entry Point    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  InputPreprocessor      │ ──► 文本/多模态预处理
│  (V1: Processor)        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Scheduler              │ ◄───► KVCacheManager/BlockSpaceManager
│  (请求调度 & 资源分配)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  ModelExecutor          │ ◄───► SamplingMetadata
│  (V1: GPUModelRunner)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Worker(s)              │ ◄───► GPU/TPU
│  (模型实际执行)         │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  OutputProcessor        │ ◄───► Detokenizer (V1)
│  (输出处理 & 解码)      │
└────────┬────────────────┘
         │
         ▼
    RequestOutput ◄───────────────────────► 流式输出
```

---

## 4. V0 vs V1 架构对比

| 组件     | V0                  | V1                                |
| -------- | ------------------- | --------------------------------- |
| 引擎入口 | `LLMEngine`         | `LLMEngine (v1)`                  |
| 输入处理 | `InputPreprocessor` | `Processor`                       |
| 调度器   | `Scheduler (core)`  | `Scheduler (v1/core)`             |
| 缓存管理 | `BlockSpaceManager` | `KVCacheManager`                  |
| 模型运行 | `ModelRunner`       | `GPUModelRunner`                  |
| 输出处理 | 内嵌在 Engine       | `OutputProcessor` + `Detokenizer` |
| 执行器   | `ExecutorBase`      | `Executor (v1)`                   |

---

## 5. 核心类索引表

### 引擎层

| 类名               | 作用              | 文件位置                         |
| ------------------ | ----------------- | -------------------------------- |
| **LLM**            | 离线推理入口类    | `entrypoints/llm.py:52`          |
| **LLMEngine**      | V0 同步引擎核心类 | `engine/llm_engine.py:123`       |
| **AsyncLLMEngine** | V0 异步引擎       | `engine/async_llm_engine.py:574` |
| **LLMEngine (V1)** | V1 引擎入口       | `v1/engine/llm_engine.py:34`     |
| **EngineCore**     | V1 引擎核心       | `v1/engine/core.py:42`           |

### 调度层

| 类名                  | 作用          | 文件位置                         |
| --------------------- | ------------- | -------------------------------- |
| **Scheduler (V0)**    | V0 调度器     | `core/scheduler.py:425`          |
| **Scheduler (V1)**    | V1 调度器     | `v1/core/scheduler.py:28`        |
| **BlockSpaceManager** | V0 块空间管理 | `core/block/interfaces.py`       |
| **KVCacheManager**    | V1 KV缓存管理 | `v1/core/kv_cache_manager.py:18` |

### Worker/执行层

| 类名                    | 作用             | 文件位置                           |
| ----------------------- | ---------------- | ---------------------------------- |
| **Worker**              | V0 GPU Worker    | `worker/worker.py`                 |
| **ModelRunner**         | V0 模型运行器    | `worker/model_runner.py`           |
| **GPUModelRunner (V1)** | V1 GPU模型运行器 | `v1/worker/gpu_model_runner.py:54` |
| **ExecutorBase**        | V0 执行器基类    | `executor/executor_base.py:27`     |

### 数据结构

| 类名               | 作用     | 文件位置             |
| ------------------ | -------- | -------------------- |
| **Sequence**       | 单个序列 | `sequence.py`        |
| **SequenceGroup**  | 序列组   | `sequence.py`        |
| **RequestOutput**  | 请求输出 | `outputs.py`         |
| **SamplingParams** | 采样参数 | `sampling_params.py` |

---

# 模型分布式部署方案

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





## 配置分布式环境

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



## 模型加载

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
### 关键通信点详解
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

### 数据流维度变化

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

##  总结
| 位置                                                         | 通信操作           | 数据           | 目的                             |
| ------------------------------------------------------------ | ------------------ | -------------- | -------------------------------- |
| o_proj 后                                                    | all_reduce (SUM)   | attention 输出 | 合并各 rank 的 half heads        |
| down_proj 后                                                 | all_reduce (SUM)   | MLP 输出       | 合并各 rank 的 half intermediate |
| LM Head 后                                                   | 无 (各 rank 独立)  | logits         | 各 rank 计算完整 vocab           |
| Sampler                                                      | gather (仅 rank 0) | token_ids      | 只有 rank 0 执行采样             |
- 每层 Transformer: 2 次 all_reduce
- N 层模型: 2N 次 all_reduce 



