
### nvprof vs ncu

| nvprof --metrics | ncu --metrics (>= SM 7.0) | 指标解释 |
|------------------|----------------------------|----------|
| achieved_occupancy | sm__warps_active.avg.pct_of_peak_sustained_active | 每个活动周期的平均活动 warp 与 SM 支持的最大 warp 数之比 |
| atomic_transactions | l1tex__t_set_accesses_pipe_lsu_mem_global_op_atom.sum + l1tex__t_set_accesses_pipe_lsu_mem_global_op_red.sum | 全局内存原子和减少事务 |
| atomic_transactions_per_request | (l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum) / (l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum + l1tex__t_requests_pipe_lsu_mem_global_op_red.sum) | 为每个原子和归约指令执行的全局内存原子和归约事务的平均数量 |
| branch_efficiency | smsp__sass_average_branch_targets_threads_uniform.pct | 非发散分支与总分支的比率 |
| cf_executed | smsp__inst_executed_pipe_cbu.sum + smsp__inst_executed_pipe_adu.sum | 执行的控制流指令数 |
| cf_fu_utilization | n/a | 执行控制流指令的 SM 的利用率级别，范围为 0 到 10 |
| cf_issued | n/a | 发出的控制流指令数 |
| double_precision_fu_utilization | smsp__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active | 执行双精度浮点指令的 SM 的利用率级别 |
| dram_read_bytes | dram__bytes_read.sum | -（list中无对应解释，仅为辅助指标） |
| dram_read_throughput | dram__bytes_read.sum.per_second | 设备内存读取吞吐量 |
| dram_read_transactions | dram__sectors_read.sum | 设备内存读取事务 |
| dram_utilization | dram__throughput.avg.pct_of_peak_sustained_elapsed | 设备内存利用率相对于理论峰值利用率的级别，范围为 0 到 10 |
| dram_write_bytes | dram__bytes_write.sum | -（list中无对应解释，仅为辅助指标） |
| dram_write_throughput | dram__bytes_write.sum.per_second | 设备内存写入吞吐量 |
| dram_write_transactions | dram__sectors_write.sum | 设备内存写入事务 |
| eligible_warps_per_cycle | smsp__warps_eligible.sum.per_cycle_active | 每个活动周期有资格发布的平均 warp 数 |
| flop_count_dp | smsp__sass_thread_inst_executed_op_dadd_pred_on.sum + smsp__sass_thread_inst_executed_op_dmul_pred_on.sum + smsp__sass_thread_inst_executed_op_dfma_pred_on.sum * 2 | 非谓词线程执行的双精度浮点运算数（加法、乘法和乘法累加）。每个乘法累加运算对计数贡献 2 |
| flop_count_dp_add | smsp__sass_thread_inst_executed_op_dadd_pred_on.sum | 非断言线程执行的双精度浮点加法运算次数 |
| flop_count_dp_fma | smsp__sass_thread_inst_executed_op_dfma_pred_on.sum | 非谓词线程执行的双精度浮点乘累加运算次数，每个乘法累加运算使计数加一 |
| flop_count_dp_mul | smsp__sass_thread_inst_executed_op_dmul_pred_on.sum | 非谓词线程执行的双精度浮点乘法运算次数 |
| flop_count_hp | smsp__sass_thread_inst_executed_op_hadd_pred_on.sum + smsp__sass_thread_inst_executed_op_hmul_pred_on.sum + smsp__sass_thread_inst_executed_op_hfma_pred_on.sum * 2 | 非谓词线程执行的半精度浮点运算数（加法、乘法和乘法累加），每个乘法累加运算使计数加二 |
| flop_count_hp_add | smsp__sass_thread_inst_executed_op_hadd_pred_on.sum | 非断言线程执行的半精度浮点加法运算的次数 |
| flop_count_hp_fma | smsp__sass_thread_inst_executed_op_hfma_pred_on.sum | 非谓词线程执行的半精度浮点乘累加运算次数。每个乘法累加运算使计数加一 |
| flop_count_hp_mul | smsp__sass_thread_inst_executed_op_hmul_pred_on.sum | 非谓词线程执行的半精度浮点乘法运算次数 |
| flop_count_sp | smsp__sass_thread_inst_executed_op_fadd_pred_on.sum + smsp__sass_thread_inst_executed_op_fmul_pred_on.sum + smsp__sass_thread_inst_executed_op_ffma_pred_on.sum * 2 | 非谓词线程执行的单精度浮点运算数（加法、乘法和乘法累加），每个乘法累加运算使计数加二（不包括特殊操作） |
| flop_count_sp_add | smsp__sass_thread_inst_executed_op_fadd_pred_on.sum | 非断言线程执行的单精度浮点加法运算次数 |
| flop_count_sp_fma | smsp__sass_thread_inst_executed_op_ffma_pred_on.sum | 非谓词线程执行的单精度浮点乘累加运算次数。每个乘法累加运算使计数加一 |
| flop_count_sp_mul | smsp__sass_thread_inst_executed_op_fmul_pred_on.sum | 非谓词线程执行的单精度浮点乘法运算次数 |
| flop_count_sp_special | n/a | 非谓词线程执行的单精度浮点特殊操作数 |
| flop_dp_efficiency | smsp__sass_thread_inst_executed_ops_dadd_dmul_dfma_pred_on.avg.pct_of_peak_sustained_elapsed | 实现的双精度浮点运算与理论峰值的比值 |
| flop_hp_efficiency | smsp__sass_thread_inst_executed_ops_hadd_hmul_hfma_pred_on.avg.pct_of_peak_sustained_elapsed | 实现的半精度浮点运算与理论峰值的比值 |
| flop_sp_efficiency | smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed | 实现的单精度浮点运算与理论峰值的比值 |
| gld_efficiency | smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct | 请求的全局内存负载吞吐量与所需的全局内存负载吞吐量的比率 |
| gld_requested_throughput | n/a | 请求的全局内存负载吞吐量 |
| gld_throughput | l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second | 全局内存负载吞吐量 |
| gld_transactions | l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum | 为每个全局内存加载执行的全局内存加载事务的平均数 |
| gld_transactions_per_request | l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio | 为每个全局内存加载执行的全局内存加载事务的平均数 |
| global_atomic_requests | l1tex__t_requests_pipe_lsu_mem_global_op_atom.sum | -（list中无对应解释，仅为辅助指标） |
| global_hit_rate | (l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_red_lookup_hit.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_atom_lookup_hit.sum) / (l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_red.sum + l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum) | 统一 L1/tex 缓存中全局加载的命中率 |
| global_load_requests | l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum | -（list中无对应解释，仅为辅助指标） |
| global_reduction_requests | l1tex__t_requests_pipe_lsu_mem_global_op_red.sum | -（list中无对应解释，仅为辅助指标） |
| global_store_requests | l1tex__t_requests_pipe_lsu_mem_global_op_st.sum | -（list中无对应解释，仅为辅助指标） |
| gst_efficiency | smsp__sass_average_data_bytes_per_sector_mem_global_op_st.pct | 请求的全局内存存储吞吐量与所需的全局内存存储吞吐量的比率 |
| gst_requested_throughput | n/a | 请求的全局内存存储吞吐量 |
| gst_throughput | l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second | 全局内存存储吞吐量 |
| gst_transactions | l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum | 为每个全局内存存储执行的平均全局内存存储事务数 |
| gst_transactions_per_request | l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio | 为每个全局内存存储执行的平均全局内存存储事务数 |
| half_precision_fu_utilization | smsp__inst_executed_pipe_fp16.avg.pct_of_peak_sustained_active | 执行 16 位浮点指令和整数指令的 SM 的利用率级别，范围为0到10 |
| inst_bit_convert | smsp__sass_thread_inst_executed_op_conversion_pred_on.sum | 非谓词线程执行的位转换指令数 |
| inst_compute_ld_st | smsp__sass_thread_inst_executed_op_memory_pred_on.sum | 非谓词线程执行的计算加载/存储指令数 |
| inst_control | smsp__sass_thread_inst_executed_op_control_pred_on.sum | 非谓词线程（跳转、分支等）执行的控制流指令数 |
| inst_executed | smsp__inst_executed.sum | 执行的指令数 |
| inst_executed_global_atomics | smsp__sass_inst_executed_op_global_atom.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_global_loads | smsp__inst_executed_op_global_ld.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_global_reductions | smsp__inst_executed_op_global_red.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_global_stores | smsp__inst_executed_op_global_st.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_local_loads | smsp__inst_executed_op_local_ld.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_local_stores | smsp__inst_executed_op_local_st.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_shared_atomics | smsp__inst_executed_op_shared_atom.sum + smsp__inst_executed_op_shared_atom_dot_alu.sum + smsp__inst_executed_op_shared_atom_dot_cas.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_shared_loads | smsp__inst_executed_op_shared_ld.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_shared_stores | smsp__inst_executed_op_shared_st.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_surface_atomics | smsp__inst_executed_op_surface_atom.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_surface_loads | smsp__inst_executed_op_surface_ld.sum + smsp__inst_executed_op_shared_atom_dot_alu.sum + smsp__inst_executed_op_shared_atom_dot_cas.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_surface_reductions | smsp__inst_executed_op_surface_red.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_surface_stores | smsp__inst_executed_op_surface_st.sum | -（list中无对应解释，仅为辅助指标） |
| inst_executed_tex_ops | smsp__inst_executed_op_texture.sum | -（list中无对应解释，仅为辅助指标） |
| inst_fp_16 | smsp__sass_thread_inst_executed_op_fp16_pred_on.sum | 非谓词线程（算术、比较等）执行的半精度浮点指令数 |
| inst_fp_32 | smsp__sass_thread_inst_executed_op_fp32_pred_on.sum | 非谓词线程（算术、比较等）执行的单精度浮点指令数 |
| inst_fp_64 | smsp__sass_thread_inst_executed_op_fp64_pred_on.sum | 非谓词线程（算术、比较等）执行的双精度浮点指令数 |
| inst_integer | smsp__sass_thread_inst_executed_op_integer_pred_on.sum | 非谓词线程执行的整数指令数 |
| inst_inter_thread_communication | smsp__sass_thread_inst_executed_op_inter_thread_communication_pred_on.sum | 非谓词线程执行的线程间通信指令数 |
| inst_issued | smsp__inst_issued.sum | 发出的指令数 |
| inst_misc | smsp__sass_thread_inst_executed_op_misc_pred_on.sum | 非谓词线程执行的杂项指令数 |
| inst_per_warp | smsp__average_inst_executed_per_warp.ratio | 每个 warp 执行的平均指令数 |
| inst_replay_overhead | n/a | 每条指令执行的平均重放次数 |
| ipc | smsp__inst_executed.avg.per_cycle_active | 每个周期执行的指令 |
| issue_slot_utilization | smsp__issue_active.avg.pct_of_peak_sustained_active | 发出至少一条指令的发布槽的百分比，在所有周期中取平均值 |
| issue_slots | smsp__inst_issued.sum | 使用的问题槽数 |
| issued_ipc | smsp__inst_issued.avg.per_cycle_active | 每个周期发出的指令 |
| l1_sm_lg_utilization | l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active | -（list中无对应解释，仅为辅助指标） |
| l2_atomic_throughput | 2 * ( lts__t_sectors_op_atom.sum.per_second + lts__t_sectors_op_red.sum.per_second ) | 在 L2 缓存中接收到的原子和减少请求的内存读取吞吐量 |
| l2_atomic_transactions | 2 * ( lts__t_sectors_op_atom.sum + lts__t_sectors_op_red.sum ) | 在 L2 缓存中接收到的内存读取事务，用于原子请求和缩减请求 |
| l2_global_atomic_store_bytes | lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_atom.sum | -（list中无对应解释，仅为辅助指标） |
| l2_global_load_bytes | lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld.sum | -（list中无对应解释，仅为辅助指标） |
| l2_local_global_store_bytes | lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_st.sum + lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st.sum | -（list中无对应解释，仅为辅助指标） |
| l2_local_load_bytes | lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_ld.sum | -（list中无对应解释，仅为辅助指标） |
| l2_read_throughput | lts__t_sectors_op_read.sum.per_second + lts__t_sectors_op_atom.sum.per_second + lts__t_sectors_op_red.sum.per_second | 在 L2 缓存中接收到的所有内存读取吞吐量 |
| l2_read_transactions | lts__t_sectors_op_read.sum + lts__t_sectors_op_atom.sum + lts__t_sectors_op_red.sum | 所有读取请求在 L2 缓存中接收到的内存读取事务 |
| l2_surface_load_bytes | lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_surface_op_ld.sum | -（list中无对应解释，仅为辅助指标） |
| l2_surface_store_bytes | lts__t_bytes_equiv_l1sectormiss_pipe_tex_mem_surface_op_st.sum | -（list中无对应解释，仅为辅助指标） |
| l2_tex_hit_rate | lts__t_sector_hit_rate.pct | -（list中无对应解释，仅为辅助指标） |
| l2_tex_read_hit_rate | lts__t_sector_op_read_hit_rate.pct | 来自纹理缓存的所有读取请求在 L2 缓存中的命中率 |
| l2_tex_read_throughput | lts__t_sectors_srcunit_tex_op_read.sum.per_second | 在 L2 缓存中接收到的来自纹理缓存的内存读取吞吐量 |
| l2_tex_read_transactions | lts__t_sectors_srcunit_tex_op_read.sum | 在 L2 缓存中接收到的内存读取事务，用于来自纹理缓存的读取请求 |
| l2_tex_write_hit_rate | lts__t_sector_op_write_hit_rate.pct | 来自纹理缓存的所有写入请求在 L2 缓存中的命中率 |
| l2_tex_write_throughput | lts__t_sectors_srcunit_tex_op_write.sum.per_second | 在 L2 缓存中接收到的来自纹理缓存的内存写入吞吐量 |
| l2_tex_write_transactions | lts__t_sectors_srcunit_tex_op_write.sum | 在 L2 缓存中接收到的内存写入事务，用于来自纹理缓存的写入请求 |
| l2_utilization | lts__t_sectors.avg.pct_of_peak_sustained_elapsed | L2 缓存利用率相对于理论峰值利用率的级别，范围为 0 到 10 |
| l2_write_throughput | lts__t_sectors_op_write.sum.per_second + lts__t_sectors_op_atom.sum.per_second + lts__t_sectors_op_red.sum.per_second | 在 L2 缓存中接收到的所有内存写入吞吐量 |
| l2_write_transactions | lts__t_sectors_op_write.sum + lts__t_sectors_op_atom.sum + lts__t_sectors_op_red.sum | 所有写入请求在 L2 缓存中接收到的内存写入事务 |
| ldst_executed | n/a | 执行的本地、全局、共享和纹理内存加载和存储指令的数量 |
| ldst_fu_utilization | smsp__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active | 执行共享加载、共享存储和恒定加载指令的 SM 的利用率级别 |
| ldst_issued | n/a | 发出的本地、全局、共享和纹理内存加载和存储指令的数量 |
| local_hit_rate | n/a | 本地加载和存储的命中率 |
| local_load_requests | l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum | -（list中无对应解释，仅为辅助指标） |
| local_load_throughput | l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum.per_second | 本地内存加载吞吐量 |
| local_load_transactions | l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum | 本地内存加载事务数 |
| local_load_transactions_per_request | l1tex__average_t_sectors_per_request_pipe_lsu_mem_local_op_ld.ratio | 每次本地内存加载执行的本地内存加载事务平均数 |
| local_memory_overhead | n/a | 本地内存流量占 L1 和 L2 缓存之间总内存流量之比 |
| local_store_requests | l1tex__t_requests_pipe_lsu_mem_local_op_st.sum | -（list中无对应解释，仅为辅助指标） |
| local_store_throughput | l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum.per_second | 本地内存存储吞吐量 |
| local_store_transactions | l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum | 本地内存存储事务数 |
| local_store_transactions_per_request | l1tex__average_t_sectors_per_request_pipe_lsu_mem_local_op_st.ratio | 为每个本地内存存储执行的本地内存存储交易的平均数量 |
| nvlink_data_receive_efficiency | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_data_transmission_efficiency | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_overhead_data_received | (nvlrx__bytes_data_protocol.sum / nvlrx__bytes.sum) * 100 | -（list中无对应解释，仅为辅助指标） |
| nvlink_overhead_data_transmitted | (nvltx__bytes_data_protocol.sum / nvltx__bytes.sum) * 100 | -（list中无对应解释，仅为辅助指标） |
| nvlink_receive_throughput | nvlrx__bytes.sum.per_second | -（list中无对应解释，仅为辅助指标） |
| nvlink_total_data_received | nvlrx__bytes.sum | -（list中无对应解释，仅为辅助指标） |
| nvlink_total_data_transmitted | nvltx__bytes.sum | -（list中无对应解释，仅为辅助指标） |
| nvlink_total_nratom_data_transmitted | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_total_ratom_data_transmitted | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_total_response_data_received | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_total_write_data_transmitted | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_transmit_throughput | nvltx__bytes.sum.per_second | -（list中无对应解释，仅为辅助指标） |
| nvlink_user_data_received | nvlrx__bytes_data_user.sum | -（list中无对应解释，仅为辅助指标） |
| nvlink_user_data_transmitted | nvltx__bytes_data_user.sum | -（list中无对应解释，仅为辅助指标） |
| nvlink_user_nratom_data_transmitted | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_user_ratom_data_transmitted | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_user_response_data_received | n/a | -（list中无对应解释，仅为辅助指标） |
| nvlink_user_write_data_transmitted | n/a | -（list中无对应解释，仅为辅助指标） |
| pcie_total_data_received | pcie__read_bytes.sum | -（list中无对应解释，仅为辅助指标） |
| pcie_total_data_transmitted | pcie__write_bytes.sum | -（list中无对应解释，仅为辅助指标） |
| shared_efficiency | smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct | 请求的共享内存吞吐量与所需共享内存吞吐量的比率 |
| shared_load_throughput | l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second | 共享内存负载吞吐量 |
| shared_load_transactions | l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum | 共享内存加载事务数 |
| shared_load_transactions_per_request | n/a | 每次共享内存加载时执行的平均共享内存加载事务数 |
| shared_store_throughput | l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.per_second | 共享内存存储吞吐量 |
| shared_store_transactions | l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum | 共享内存存储事务数 |
| shared_store_transactions_per_request | n/a | 每次共享内存加载时执行的平均共享内存写入事务数 |
| shared_utilization | l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_elapsed | 共享内存相对于理论峰值利用率的利用率级别 |
| single_precision_fu_utilization | smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active | 执行单精度浮点指令和整数指令的 SM 的利用率级别 |
| sm_efficiency | smsp__cycles_active.avg.pct_of_peak_sustained_elapsed | 至少一个 warp 在特定 SM 上处于活动状态的时间百分比 |
| sm_tex_utilization | l1tex__texin_sm2tex_req_cycles_active.avg.pct_of_peak_sustained_active | -（list中无对应解释，仅为辅助指标） |
| special_fu_utilization | smsp__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active | 执行 sin、cos、ex2、popc、flo 和类似指令的 SM 的利用率级别，范围为 0 到 10 |
| stall_constant_memory_dependency | smsp__warp_issue_stalled_imc_miss_per_warp_active.pct | 由于立即常量高速缓存未命中而发生的停顿百分比 |
| stall_exec_dependency | smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct + smsp__warp_issue_stalled_wait_per_warp_active.pct | 由于指令所需的输入尚不可用而发生的停顿百分比 |
| stall_inst_fetch | smsp__warp_issue_stalled_no_instruction_per_warp_active.pct | 由于尚未获取下一条汇编指令而发生的停顿百分比 |
| stall_memory_dependency | smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct | 由于所需资源不可用或未完全利用而无法执行内存操作，或者由于给定类型的太多请求未完成而导致的停顿百分比 |
| stall_memory_throttle | smsp__warp_issue_stalled_drain_per_warp_active.pct + smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct | 由于内存节流而发生的停顿百分比 |
| stall_not_selected | smsp__warp_issue_stalled_not_selected_per_warp_active.pct | 由于未选择 warp 而发生的停顿百分比 |
| stall_other | smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct + smsp__warp_issue_stalled_misc_per_warp_active.pct | 由于各种原因发生的停顿百分比 |
| stall_pipe_busy | smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct + smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct | 由于计算管道繁忙而无法执行计算操作而发生的停顿百分比 |
| stall_sleeping | smsp__warp_issue_stalled_sleeping_per_warp_active.pct | -（list中无对应解释，仅为辅助指标） |
| stall_sync | smsp__warp_issue_stalled_barrier_per_warp_active.pct + smsp__warp_issue_stalled_membar_per_warp_active.pct | 由于 warp 在 __syncthreads() 调用时被阻塞而发生的停顿百分比 |
| stall_texture | smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct | 由于纹理子系统被充分利用或有太多未完成的请求而发生的停顿百分比 |
| surface_atomic_requests | l1tex__t_requests_pipe_tex_mem_surface_op_atom.sum | -（list中无对应解释，仅为辅助指标） |
| surface_load_requests | l1tex__t_requests_pipe_tex_mem_surface_op_ld.sum | -（list中无对应解释，仅为辅助指标） |
| surface_reduction_requests | l1tex__t_requests_pipe_tex_mem_surface_op_red.sum | -（list中无对应解释，仅为辅助指标） |
| surface_store_requests | l1tex__t_requests_pipe_tex_mem_surface_op_st.sum | -（list中无对应解释，仅为辅助指标） |
| sysmem_read_bytes | lts__t_sectors_aperture_sysmem_op_read * 32 | -（list中无对应解释，仅为辅助指标） |
| sysmem_read_throughput | lts__t_sectors_aperture_sysmem_op_read.sum.per_second | 系统内存读取吞吐量 |
| sysmem_read_transactions | lts__t_sectors_aperture_sysmem_op_read.sum | 系统内存读取事务数 |
| sysmem_read_utilization | n/a | 系统内存的读取利用率相对于理论峰值利用率的级别，范围为 0 到 10 |
| sysmem_utilization | n/a | 系统内存利用率相对于理论峰值利用率的级别 |
| sysmem_write_bytes | lts__t_sectors_aperture_sysmem_op_write * 32 | -（list中无对应解释，仅为辅助指标） |
| sysmem_write_throughput | lts__t_sectors_aperture_sysmem_op_write.sum.per_second | 系统内存写入吞吐量 |
| sysmem_write_transactions | lts__t_sectors_aperture_sysmem_op_write.sum | 系统内存写入事务数 |
| sysmem_write_utilization | n/a | 系统内存的写入利用率相对于理论峰值利用率的级别，范围为 0 到 10 |
| tensor_precision_fu_utilization | sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active | -（list中无对应解释，仅为辅助指标） |
| tensor_precision_int_utilization | sm__pipe_tensor_op_imma_cycles_active.avg.pct_of_peak_sustained_active (SM 7.2+) | -（list中无对应解释，仅为辅助指标） |
| tex_cache_hit_rate | l1tex__t_sector_hit_rate.pct | 统一缓存命中率 |
| tex_cache_throughput | n/a | 统一缓存吞吐量 |
| tex_cache_transactions | l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active + l1tex__tex_writeback_active.avg.pct_of_peak_sustained_active | 统一缓存读取事务 |
| tex_fu_utilization | smsp__inst_executed_pipe_tex.avg.pct_of_peak_sustained_active | 执行全局、局部和纹理内存指令的 SM 的利用率级别，范围为 0 到 10 |
| tex_sm_tex_utilization | l1tex__f_tex2sm_cycles_active.avg.pct_of_peak_sustained_elapsed | -（list中无对应解释，仅为辅助指标） |
| tex_sm_utilization | sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed | -（list中无对应解释，仅为辅助指标） |
| tex_utilization | n/a | 统一缓存利用率相对于理论峰值利用率的级别 |
| texture_load_requests | l1tex__t_requests_pipe_tex_mem_texture.sum | -（list中无对应解释，仅为辅助指标） |
| warp_execution_efficiency | smsp__thread_inst_executed_per_inst_executed.ratio | 每个 warp 的平均活动线程数与 SM 支持的每个 warp 的最大线程数之比 |
| warp_nonpred_execution_efficiency | smsp__thread_inst_executed_per_inst_executed.pct | 执行非谓词指令的每个 warp 的平均活动线程数与 SM 支持的每个 warp 的最大线程数之比 |

### 一、表格中 `n/a` 项整理（nvprof指标无对应ncu指标）

| nvprof --metrics | 指标解释 | 备注（ncu中无直接对应指标） |
|------------------|----------|------------------------------|
| cf_fu_utilization | 执行控制流指令的 SM 的利用率级别，范围为 0 到 10 | ncu未提供直接映射指标，可通过 `smsp__inst_executed_pipe_cbu.avg.pct_of_peak_sustained_active` 近似评估控制流单元利用率 |
| cf_issued | 发出的控制流指令数 | ncu未单独统计“发出的控制流指令”，仅统计“执行的控制流指令”（`cf_executed`） |
| flop_count_sp_special | 非谓词线程执行的单精度浮点特殊操作数 | ncu未单独拆分“特殊单精度浮点操作”，需结合 `smsp__sass_thread_inst_executed_op_fp32_special_pred_on.sum` 自定义统计（部分架构支持） |
| gld_requested_throughput | 请求的全局内存负载吞吐量 | ncu仅提供实际吞吐量（`gld_throughput`），需通过 `gld_efficiency * 实际吞吐量` 间接计算请求吞吐量 |
| gst_requested_throughput | 请求的全局内存存储吞吐量 | 同上述逻辑，需通过 `gst_efficiency * 实际吞吐量` 间接计算 |
| inst_replay_overhead | 每条指令执行的平均重放次数 | ncu无直接指标，可通过 `(inst_issued / inst_executed) - 1` 近似计算指令重放率 |
| ldst_executed | 执行的本地、全局、共享和纹理内存加载和存储指令的数量 | ncu拆分统计各类型内存指令（如 `inst_executed_global_loads`/`inst_executed_shared_stores` 等），无汇总指标 |
| ldst_issued | 发出的本地、全局、共享和纹理内存加载和存储指令的数量 | ncu未统计“发出的内存指令”，仅统计“执行的内存指令” |
| local_hit_rate | 本地加载和存储的命中率 | ncu无直接指标，可通过 `l1tex__t_sectors_pipe_lsu_mem_local_op_ld_lookup_hit.sum / l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum` 自定义计算本地加载命中率（存储同理） |
| local_memory_overhead | 本地内存流量占 L1 和 L2 缓存之间总内存流量之比 | ncu需手动计算：`(local_load_throughput + local_store_throughput) / (l2_read_throughput + l2_write_throughput)` |
| shared_load_transactions_per_request | 每次共享内存加载时执行的平均共享内存加载事务数 | ncu无直接指标，可通过 `shared_load_transactions / 共享内存加载请求数` 计算（请求数需从 `l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum` 获取） |
| shared_store_transactions_per_request | 每次共享内存加载时执行的平均共享内存写入事务数 | 同上述逻辑，通过 `shared_store_transactions / l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum` 计算 |
| sysmem_read_utilization | 系统内存的读取利用率相对于理论峰值利用率的级别，范围为 0 到 10 | ncu无直接指标，可通过 `sysmem_read_throughput / 系统内存理论峰值带宽` 计算利用率 |
| sysmem_utilization | 系统内存利用率相对于理论峰值利用率的级别 | ncu拆分统计读/写利用率，无汇总指标 |
| sysmem_write_utilization | 系统内存的写入利用率相对于理论峰值利用率的级别，范围为 0 到 10 | 同 `sysmem_read_utilization`，通过 `sysmem_write_throughput / 理论峰值带宽` 计算 |
| tex_cache_throughput | 统一缓存吞吐量 | ncu拆分统计纹理缓存读/写吞吐量（如 `l2_tex_read_throughput`），无汇总指标 |
| tex_utilization | 统一缓存利用率相对于理论峰值利用率的级别 | ncu可通过 `l1tex__t_sectors.avg.pct_of_peak_sustained_elapsed` 近似评估纹理缓存整体利用率 |
| nvlink_data_receive_efficiency | NVLink 数据接收效率 | ncu无直接指标，需结合 `nvlink_receive_throughput` 和 NVLink 理论带宽计算 |
| nvlink_data_transmission_efficiency | NVLink 数据传输效率 | 同上述逻辑，结合 `nvlink_transmit_throughput` 计算 |
| nvlink_total_nratom_data_transmitted | NVLink 非原子数据传输总量 | ncu未拆分原子/非原子数据传输 |
| nvlink_total_ratom_data_transmitted | NVLink 原子数据传输总量 | 同上 |
| nvlink_total_response_data_received | NVLink 响应数据接收总量 | ncu无细分响应数据统计 |
| nvlink_total_write_data_transmitted | NVLink 写数据传输总量 | ncu仅统计总传输数据（`nvlink_total_data_transmitted`），无读写拆分 |
| nvlink_user_nratom_data_transmitted | NVLink 用户态非原子数据传输总量 | ncu未细分用户态数据类型 |
| nvlink_user_ratom_data_transmitted | NVLink 用户态原子数据传输总量 | 同上 |
| nvlink_user_response_data_received | NVLink 用户态响应数据接收总量 | ncu无细分 |
| nvlink_user_write_data_transmitted | NVLink 用户态写数据传输总量 | 同上 |

### 二、高频关注指标分类整理（按性能分析维度）

#### 1. 内存性能相关指标

| 类别 | nvprof指标 | ncu对应指标 | 核心作用 |
|------|------------|-------------|----------|
| 全局内存 | gld_throughput | l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second | 评估全局内存读取吞吐量 |
| 全局内存 | gst_throughput | l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second | 评估全局内存写入吞吐量 |
| 全局内存 | gld_efficiency | smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct | 衡量全局内存读取的合并效率 |
| 共享内存 | shared_load_throughput | l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.per_second | 评估共享内存读取吞吐量 |
| 共享内存 | shared_efficiency | smsp__sass_average_data_bytes_per_wavefront_mem_shared.pct | 衡量共享内存访问的合并效率 |
| 本地内存 | local_load_throughput | l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum.per_second | 评估本地内存读取吞吐量 |
| L2缓存 | l2_read_throughput | lts__t_sectors_op_read.sum.per_second + ...（原子/归约） | 评估L2缓存读取吞吐量 |
| 显存 | dram_utilization | dram__throughput.avg.pct_of_peak_sustained_elapsed | 评估显存带宽利用率 |

#### 2. 计算性能相关指标

| 类别 | nvprof指标 | ncu对应指标 | 核心作用 |
|------|------------|-------------|----------|
| 浮点运算 | flop_count_sp | smsp__sass_thread_inst_executed_op_fadd/fmul/ffma_pred_on.sum（加权） | 统计单精度浮点运算总量 |
| 浮点运算 | flop_sp_efficiency | smsp__sass_thread_inst_executed_ops_fadd_fmul_ffma_pred_on.avg.pct_of_peak_sustained_elapsed | 评估单精度浮点单元利用率 |
| 双精度运算 | flop_count_dp | smsp__sass_thread_inst_executed_op_dadd/dmul/dfma_pred_on.sum（加权） | 统计双精度浮点运算总量 |
| 计算单元 | single_precision_fu_utilization | smsp__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active | 评估单精度浮点单元负载 |

#### 3. 线程与Warp调度相关指标

| 类别 | nvprof指标 | ncu对应指标 | 核心作用 |
|------|------------|-------------|----------|
| Warp利用率 | achieved_occupancy | sm__warps_active.avg.pct_of_peak_sustained_active | 评估SM上活动Warp的占比 |
| Warp效率 | warp_execution_efficiency | smsp__thread_inst_executed_per_inst_executed.ratio | 衡量Warp内线程发散程度 |
| 指令吞吐量 | ipc | smsp__inst_executed.avg.per_cycle_active | 评估每周期执行的指令数（IPC） |

#### 4. 停顿分析相关指标

| 类别 | nvprof指标 | ncu对应指标 | 核心作用 |
|------|------------|-------------|----------|
| 内存停顿 | stall_memory_dependency | smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct | 统计因内存依赖导致的停顿占比 |
| 同步停顿 | stall_sync | smsp__warp_issue_stalled_barrier/membar_per_warp_active.pct | 统计因__syncthreads导致的停顿占比 |
| 指令获取停顿 | stall_inst_fetch | smsp__warp_issue_stalled_no_instruction_per_warp_active.pct | 统计因指令获取失败导致的停顿占比 |

是否需要我进一步整理**ncu自定义指标的计算脚本**（比如通过ncu输出的csv文件计算local_hit_rate），或者补充**各指标的性能优化阈值参考**？









### 硬件指标



```python
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
properties = driver.active.utils.get_device_properties(DEVICE.index)
# SM 流式处理器数量
NUM_SM = properties["multiprocessor_count"]
# 每个SM上寄存器数量 每个寄存器有32位 即 4字节=32/8
NUM_REGS = properties["max_num_regs"]
# 每个 SM 可用的最大共享内存容量
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
# 获取当前 GPU 的微架构信息（例如 sm_80 代表 Ampere 架构，sm_90 代表 Hopper 架构）。
target = triton.runtime.driver.active.get_current_target()
```





| **属性名称**           | **数值**   | **添加单位后的表达**          | **含义说明**                           |
| ---------------------- | ---------- | ----------------------------- | -------------------------------------- |
| `max_shared_mem`       | 101,376    | **101,376 Bytes (约 99 KB)**  | 每个 SM 可用的最大共享内存容量         |
| `max_num_regs`         | 65,536     | **65,536 Registers**          | 每个 SM 可用的 32-bit 寄存器总数       |
| `multiprocessor_count` | 128        | **128 SMs**                   | 流式多处理器的数量（决定并行规模）     |
| `warpSize`             | 32         | **32 Threads**                | 硬件执行的最小单元（线程束）大小       |
| `sm_clock_rate`        | 2,535,000  | **2,535,000 kHz (2.535 GHz)** | GPU 核心（SM）的最大加速频率           |
| `mem_clock_rate`       | 10,501,000 | **10,501,000 kHz (10.5 GHz)** | 显存的数据传输频率（有效频率通常更高） |
| `mem_bus_width`        | 384        | **384 Bits**                  | 显存位宽（决定了数据交换的“车道数”）   |

#### 理论显存带宽计算

利用你提供的这些数据，我们可以计算出这块显卡最核心的性能指标——理论峰值带宽。

公式为：



$$\text{Bandwidth} = \frac{\text{mem\_clock\_rate} \times 2 \times (\text{mem\_bus\_width} / 8)}{10^6}$$



(注：对于 GDDR6X，计算方式略有不同，但 4090 的理论带宽通常在 1,008 GB/s 左右。)

#### 

#### 寄存器容量

1. 单个寄存器的容量：32 Bits

​	在 NVIDIA GPU（如你的 RTX 4090）中，每个寄存器的大小固定为 **32 位（4 字节）**。

2. 每个线程能用多少？

​	虽然硬件上有几万个寄存器，但它们是分给成百上千个线程用的。

- **默认限制**：在 CUDA/Triton 中，单个线程通常最多只能使用 **255 个** 寄存器。
- **这意味着**：一个线程在不访问显存的情况下，手中最多能握住大约 **1 KB** ($255 \times 4 \text{ Bytes}$) 的临时数据。

#### 线程数限制

对于你的 **RTX 4090 (Ada Lovelace 架构)**，每个 SM 的最大线程数如下：

1. 核心限制数值

- **每个 SM 最大线程数**：**2048 个线程**
- **每个 SM 最大线程束 (Warp) 数**：**64 个 Warps** ($2048 / 32 = 64$)
- **线程块 (Thread Block) 限制**：单个线程块最多包含 **1024 个线程**。这意味着要填满一个 SM 的 2048 个名额，你至少需要启动 2 个 1024 线程的块，或者 4 个 512 线程的块。









### 合并访存

在 CUDA 编程中，**合并访存（Memory Coalescing）** 是提高全局内存（Global Memory）带宽利用率的最核心优化手段。

简单来说，合并访存是指：当一个 Warp（一组 32 个线程）同时访问一段**连续**的内存地址时，硬件会将这些多次访问请求合并为一个或极少数个内存事务（Transaction）来完成。

------

1. 核心原理：为什么需要合并？

GPU 的全局内存并非像 CPU 缓存那样可以随意读取单个字节。内存系统通常以 **32 字节、64 字节或 128 字节** 为粒度的“事务”进行传输。

- **合并成功**：如果 Warp 内 32 个线程请求的地址连在一起，硬件只需发出 1 次 128 字节的事务就能填满所有线程的需求，带宽利用率接近 100%。
- **合并失败（不连续访问）**：如果 32 个线程访问的地址散落在内存各处，硬件可能需要发出 32 次独立的内存事务。即便每个线程只想要 4 字节，硬件每次也必须传输至少 32 字节，导致大量带宽被浪费。

------

2. 满足合并访存的条件

要实现高效的合并访存，通常需要满足以下两个主要条件：

A. 访问地址的连续性 (Contiguity)

Warp 中相邻线程（如 `thread 0` 和 `thread 1`）访问的地址也应该是相邻的。

- **推荐：** `A[threadIdx.x]` —— 线程 $i$ 访问地址 $p+i$。
- **规避：** `A[threadIdx.x * 2]` —— 产生了空隙（Strided Access），会导致带宽减半。

B. 对齐 (Alignment)

内存事务通常起始于 32/64/128 字节的边界。虽然现代显卡（计算能力 3.0 及以上）对非对齐访问的惩罚已经大大减小，但确保基地址对齐依然是最佳实践。

------

3. 常见场景分析

场景一：结构体数组 (AoS) vs 数组结构体 (SoA)

这是导致访存不合并的最常见原因。

- **AoS (Array of Structures)**:

  C++

  ```
  struct Particle { float x, y, z; };
  Particle p[N]; 
  // 每个线程读取 p[i].x，但 p[i].x 和 p[i+1].x 之间隔了 y 和 z
  ```

  **后果**：由于跨步访问（Stride），访存无法完全合并。

- **SoA (Structure of Arrays)**:

  C++

  ```
  struct Particles { float x[N], y[N], z[N]; };
  // 线程 i 访问 x[i]，线程 i+1 访问 x[i+1]
  ```

  **后果**：地址完美连续，**实现 100% 合并访存**。

场景二：矩阵转置

在处理矩阵时，行优先存储意味着横向读取是合并的，但纵向读取（按列）会导致巨大的跨步。

- **解决方案**：通常使用 **Shared Memory（共享内存）**。先将数据以合并方式读入共享内存，在共享内存中进行转置，最后再以合并方式写回全局内存。

------

4. 性能影响对比

| **访问方式**             | **内存事务数量**  | **带宽利用率** |
| ------------------------ | ----------------- | -------------- |
| **完美合并 (Coalesced)** | 1 个 (128B)       | ~100%          |
| **跨步访问 (Stride=2)**  | 2 个 (128B x 2)   | ~50%           |
| **随机访问 (Random)**    | 多个 (最多 32 个) | < 10%          |

------

5. 如何检测？

你可以使用 NVIDIA 的性能分析工具来查看合并情况：

- **NVIDIA Nsight Compute**：查看 `Memory Workload Analysis` 部分。
- **指标**：关注 `Global Load L2 Transactions`。如果这个值远高于请求的数据量，说明存在严重的访存不合并。

**总结建议**：在写 Kernel 时，始终保证 `threadIdx.x` 负责操作内存的最内层维度（最连续的部分）。

您是否正在调试某个具体的算子，发现其内存带宽受限？我可以帮您分析具体的代码片段。







### L2缓存利用

在 GPU 架构中，L2 缓存是连接全局内存（VRAM）和 Streaming Multiprocessors (SM) 的重要桥梁。与 L1/Shared Memory 不同，L2 缓存是**所有 SM 共享的**。

要更好地利用 L2 缓存，核心思路是：**减少对显存（DRAM）的直接访问，提高 L2 的命中率（Hit Rate）和数据复用（Reuse）。**

---

#### 1. 内存访问的局部性（Locality）优化

L2 缓存的效率极大程度上取决于数据访问的**时间局部性**（近期访问过）和**空间局部性**（物理地址邻近）。

* **数据对齐与合并访存**：虽然合并访存主要是为了减少内存事务，但它也能确保一次 L2 缓存行（Cache Line，通常为 32 字节或 128 字节）的加载能被多个线程充分利用，避免重复加载。
* **Tile-based 算法设计**：将大矩阵或大数据集拆分为能够装入 L2 的分块（Tiles）。如果一个 Tile 的数据能驻留在 L2 中并被多次读写，就能大幅降低 DRAM 带宽压力。

---

#### 2. 使用 L2 访问持久化（L2 Persistence）

从 NVIDIA Ampere 架构（Compute Capability 8.0+）开始，CUDA 引入了显式控制 L2 缓存的特性。

* **L2 Set-Aside**：你可以预留一部分 L2 缓存（例如 50%），专门用于“持久化”存储。
* **Access Policy Window**：通过 `cudaStreamAttrValue` 告诉 GPU，某一段内存地址在访问时具有更高的优先级留在 L2 中，不被其他普通访问轻易剔除（Evict）。
* **适用场景**：在深度学习中，某些权重（Weights）或频繁读取的查找表（Lookup Tables）非常适合设为持久化。



---

#### 3. 优化访存模式（Access Patterns）

* **减少原子操作的冲突**：原子操作（Atomic Operations）通常在 L2 层面处理。如果大量线程竞争同一个地址，L2 的吞吐量会成为瓶颈。尽量通过 Shuffle 指令或 Shared Memory 先进行局部规约。
* **避免读取后立即丢弃**：如果某些数据只使用一次且规模巨大（流式数据），过多的读取会“污染” L2 缓存，把原本有复用价值的数据挤出去。在某些底层开发中，可以使用 **Non-coherent / Streaming Loads** 指令（如 `LDG.E` 的特殊缓存策略）来跳过缓存或降低其优先级。

---

#### 4. 算子融合（Operator Fusion）

这是目前主流推理框架（如 TensorRT, Triton）提升 L2 利用率最高效的方法。

* **原理**：将多个算子（如 `Conv + Bias + ReLU`）合并成一个 Kernel。
* **收益**：中间结果（如 Conv 的输出）直接留在寄存器或 L2 缓存中，作为下一个逻辑算子的输入，而**不需要写回显存再读回来**。这直接节省了大量的显存带宽。

---

#### 5. 调整控制参数（针对高级用户）

在一些复杂的应用中，可以通过 CUDA API 手动干预：

* **`cudaDeviceSetLimit`**：调整缓存配置。
* **利用 `cuMemAddressRange_t**`：在多 GPU 环境下，通过 NVLink 优化 L2 缓存的一致性和访问路径。

---

#### 6. 监控与性能分析

如果不确定 L2 是否被高效利用，可以使用 **NVIDIA Nsight Compute**：

* 查看 **`Memory Workload Analysis`**。
* 关注 **`L2 Hit Rate`**（命中率）：如果命中率极低且显存带宽受限，说明存在严重的缓存失效。
* 查看 **`L2 Compression Success Rate`**：现代 GPU 支持 L2 级别的数据压缩，保持数据的压缩状态可以变相增加 L2 的有效容量。

---

##### 总结

利用 L2 的关键在于：**复用、对齐、融合**。

**您目前的 Kernel 遇到了瓶颈吗？** 如果可以提供算子的逻辑类型（例如是访存密集型的还是计算密集型的），我可以为您提供具体的代码层面的优化建议，比如如何使用 `cudaStreamContext` 设置 L2 窗口。
