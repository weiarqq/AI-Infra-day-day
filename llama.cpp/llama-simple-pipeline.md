ggml_backend_load_all()



llama_model_default_params()



llama_model_load_from_file()

​	llama_model_load_from_file_impl()

​                 new llama_model()

​		 ggml_backend_dev_count()  // 统计可用后端 并遍历

​		 	ggml_backend_registry()

​			 ggml_backend_dev_t dev = ggml_backend_dev_get(i);  // 除 CPU外，记录可用的 GPU和igpu rpc等

​		若存在RPC/GPU/IGPU后端则：

​			ggml_backend_dev_props props;

​			ggml_backend_dev_get_props(dev.dev, &props);

​                llama_model_load

​			llama_model_loader ml() //llama_model_loader::llama_model_loader

​				ggml_context* ctx;

​		                gguf_init_params

​                                gguf_init_from_file

​				get_key() 获取模型架构

​                                LLM_KV(llm_arch_from_string(arch_name));	 // 模型架构枚举类型
​			        new llama_file // 创建模型文件读取 句柄				


​				for (ggml_tensor * cur = ggml_get_first_tensor(ctx); cur; cur = ggml_get_next_tensor(ctx, cur))

​					读取 权重名称 参数数量 参数占用字节数

​					创建tensor->llama_tensor_weight

​		        model.load_arch(ml);

​                        model.load_hparams(ml); // 超参数和模型配置信息 即config.json内信息

​                        model.load_vocab(ml);

​                        model.load_stats(ml);

​                        model.load_tensors(ml)

​				    make_cpu_buft_list // build a list of buffer types for the CPU and GPU devices

​				    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

​				    ggml_backend_dev_memory

​                                    llama_model::impl::layer_dev   // 划分 layer 根据devices，ngl

​				    create_tensor lambda // llama-model.cpp#line3103 *****

​						llama_model_loader::create_tensor // create_tensor 只创建 ggml_tensor 元数据结构（形状、类型、buffer 类型、context 关联），并不分配权重数据的内存或加载权重

​							ctx_for_buft lambda

​								ggml_tensor_overhead()*max_n_tensors;

​								ggml_init_params

​								ggml_context * ctx = ggml_init(params);

​							buft_for_tensor lambda

​								llm_tensor tn_tensor

​								llm_tensor_info = llm_tensor_info_for(tn_tensor)

​								ggml_op op;

​								buft_list_t * buft_list; 输入输出

​								ggml_backend_buffer_type_t buft = nullptr;

​								auto * buft_dev = ggml_backend_buft_get_device(buft);

​								buft = ggml_backend_dev_buffer_type(cpu_dev);

​				    create_tensor_gate_up_exps lambda // llama-model.cpp#line3116

​						创建 gate_up_exps 类型tensor

​				    create_tensor_qkv lambda // llama-model.cpp#line3125

​						创建 attention qkv 类型tensor

​				    switch (arch) 选择模型对应架构，创建模型权重tensor

​			    ml.done_getting_tensors();

​			    ml.init_mappings

​			    使用cpu buff兜底

​			    ml.get_mapping_range

​			    ggml_get_max_tensor_size

​		            ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (char *) addr + first, last - first, max_size);  // 向主机申请内存 存放权重

​			    load_all_data *****加载权重到内存中

​					upload_backend lambda

​						ggml_backend_buffer_get_type

​						ggml_backend_buft_get_device

​						ggml_backend_dev_props

​						ggml_backend_dev_get_props

​						ggml_backend_dev_host_buffer_type

​						ggml_backend_buft_alloc_buffer

​						ggml_backend_buffer_get_base

​						ggml_backend_event_new

​						ggml_backend_dev_init

​					 for (struct ggml_tensor * cur = ggml_get_first_tensor(ctx); cur != NULL; cur = ggml_get_next_tensor(ctx, cur)) 

​						        const auto * weight = get_weight(ggml_get_name(cur));

​							progress_callback

​							ggml_nbytes(cur);

​							use_mmap

​								ggml_backend_tensor_alloc

​					   ggml_backend_event_synchronize(event);

​					   ggml_backend_event_free

​					   ggml_backend_buffer_free

​					  ggml_backend_free





llama_model_get_vocab

llama_tokenize				

llama_context_default_params

llama_context * ctx = llama_init_from_model(model, ctx_params);

​						检查模型配置是否符合条件

​						new llama_context *****

​							模型部署的配置和rope的配置

​							获取当前可支持的后端/ GPU/ ACCEL/CPU

​								ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());

​								ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr; 注册后端

​								ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");

​								llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

​								llama_context::output_reserve

​										ggml_backend_cpu_buffer_type

​										ggml_backend_dev_host_buffer_type

​										ggml_backend_buft_alloc_buffer

​										ggml_backend_buffer_clear

​										ggml_backend_buffer_get_base

​								model.create_memory(params_mem, cparams) // kv cache conv_state

​									llm_arch_is_recurrent	

​										llama_memory_recurrent

​									llm_arch_is_hybrid    // qwen3.5

​										llama_memory_hybrid// llama_memory_hybrid::llama_memory_hybrid

​											llama_kv_cache

​												ggml_backend_alloc_ctx_tensors_from_buft_impl

​													alloc_tensor_range

​											llama_memory_recurrent

​								 初始化后端

​								 	ggml_backend_get_default_buffer_type

​							         	ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

​								sched_reserve(); ***** // llama_context::sched_reserve()

​									synchronize

​									graph_max_nodes

​									llm_graph_result

​									ggml_backend_sched_new

​										(ggml_backend_sched *) calloc

​									llama_memory_context_ptr

​									cparams.auto_fa

​										graph_reserve

​											ggml_backend_sched_reset

​											gf_res_prev->reset();

​											gparams = graph_params

​											res->reset();

​											model.build_graph(gparams); *****

​												swith(arch)

​													LLM_ARCH_LLAMA_XXX

​														llm_build_qwen35moe::llm_build_qwen35moe

​															build_xxx

​																ggml_ops_xxx

​																ggml_set_input

​																ggml_build_forward_select

​																	ggml_build_forward_impl

​																ggml_build_forward_expand

​																	ggml_build_forward_impl

​																		ggml_visit_parents_graph

​																			ggml_hash_find

​																			ggml_bitset_get

​															build_inp_mem_hybrid *****

​												llm->build_pooling

​												llm->build_sampling()

​												llm->build_dense_out

​												llm->res->set_outputs();

​												llm->res->get_gf();

​											ggml_backend_sched_split_graph *****

​												ggml_init(params);

​												pass 1: assign backends to ops with pre-allocated inputs

​													ggml_backend_sched_backend_id_from_cur

​														ggml_backend_sched_backend_from_buffer

​															ggml_backend_supports_buft

​															ggml_backend_supports_op

​												pass 2: expand current backend assignments

​												

​												pass 3: upgrade nodes to higher prio backends with compatible buffer types

​												

​												pass 4: assign backends to remaining src from dst and view_src

​												pass 5: split graph, find tensors that need to be copied

​													

llama_sampler_chain_default_params

llama_sampler * smpl = llama_sampler_chain_init(sparams);

llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

for tokens:

​	llama_token_to_piece

llama_batch_get_one

llama_model_has_encoder 判断是否为encoder模型



while True

​	llama_decode() *****

​		balloc->init

​		sched_reserve

​		memory_update

​		memory->init_batch

​		output_reserve

​		mctx->get_ubatch

​		process_ubatch

​			gf_res_prev.get();

​			res->get_gf();

​			graph_params(res, ubatch, mctx, gtype);

​			res->reset();

​			重新构建计算图

​				ggml_backend_sched_reset(sched.get());

​				ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

​				gf = model.build_graph(gparams);

​				ggml_backend_sched_alloc_graph

​			graph_compute(res->get_gf(), ubatch.n_tokens > 1); *****

​				ggml_threadpool_t

​				ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));

​				ggml_backend_reg_get_proc_address

​				ggml_backend_sched_graph_compute_async

​					ggml_backend_sched_compute_splits *****

​						for split in splits:

​							ggml_backend_sched_split

​						ggml_backend_graph_compute_async

​			ggml_backend_sched_get_tensor_backend

​			ggml_backend_tensor_get_async

​	llama_sampler_sample

​	llama_vocab_is_eog

​	llama_token_to_piece

​	llama_batch_get_one
