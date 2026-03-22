#include <cublas_v2.h>
#include <cuda.h>
#include <cute/tensor.hpp>
#include <float.h>
#include <stdlib.h>

// BlockSwizzle: means apply thread block swizzle across N dim
template <
    typename T,
    int BM,
    int BN,
    int BK,
    int kStage,
    typename TiledMMA,
    typename G2SCopyA,
    typename G2SCopyB,
    typename SmemLayoutA,
    typename SmemLayoutB,
    typename SmemLayoutC,
    typename S2RCopyAtomA,
    typename S2RCopyAtomB,
    typename R2SCopyAtomC,
    typename S2GCopyAtomC,
    typename S2GCopyC,
    const bool BlockSwizzle>
__global__ void hgemm_mma_stages_block_swizzle_tn_cute_kernel(
    T* Aptr, T* Bptr, T* Dptr, int m, int n, int k)
{
    using namespace cute;
    // Initilize shared memory
    extern __shared__ T shm_data[];

    T* Ashm = shm_data;
    // cute::cosize(SmemLayoutA {})计算 A在共享内存的偏移
    T* Bshm = shm_data + cute::cosize(SmemLayoutA {});

    // Initilize thread block
    int idx = threadIdx.x;
    // BlockSwizzle 0/1 control use block swizzle or not.
    int ix = ((int)BlockSwizzle) * blockIdx.z * gridDim.x + blockIdx.x;
    int iy = blockIdx.y;

    if (iy * BM >= m || ix * BN >= n)
        return;

    // use Tensor notation to represent device pointer + dimension
    Tensor A = make_tensor(make_gmem_ptr(Aptr), make_shape(m, k), make_stride(k, Int<1> {}));
    Tensor B = make_tensor(make_gmem_ptr(Bptr), make_shape(n, k), make_stride(k, Int<1> {}));
    Tensor D = make_tensor(make_gmem_ptr(Dptr), make_shape(m, n), make_stride(n, Int<1> {}));

    // slice the tensor to small one which is used for current thread block.
    // tile 维度(BM, BK), 数据根据 tile 划分为(M/BM. K/BK), iy代表第几个tile 一个block处理一个 Tile，所以可以对应上
    Tensor gA = local_tile(A, make_tile(Int<BM> {}, Int<BK> {}), make_coord(iy, _)); // (BM, BK, num_tile_k)
    Tensor gB = local_tile(B, make_tile(Int<BN> {}, Int<BK> {}), make_coord(ix, _)); // (BN, BK, num_tile_k)
    Tensor gD = local_tile(D, make_tile(Int<BM> {}, Int<BN> {}), make_coord(iy, ix)); // (BM, BN)

    // shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA {}); // (BM, BK, kStage)
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB {}); // (BN, BK, kStage)

    // dispatch TileA/TileB/TileC mma tensor into thread fragment via partition
    TiledMMA tiled_mma;
    // 声明threadIdx.x负责的数据写入到gD的位置
    // 在全局内存（或共享内存）的 $BM \times BN$ 大块中，标出当前线程最终要写回的那些点。
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    auto tCgD = thr_mma.partition_C(gD); // (MMA,MMA_M, MMA_N)

    // 这几行代码执行后，编译器会真的在当前线程的 寄存器堆（Register File） 里分配空间。
    // gA(_, _, 0) 代表计算一个tile有多大
    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0)); // (MMA, MMA_M, MMA_K)
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0)); // (MMA, MMA_N, MMA_K)
    auto tCrD = thr_mma.partition_fragment_C(gD); // (MMA, MMA_M, MMA_N)
    clear(tCrD);

    // from global memory to shared memory
    // 从全局内存中搬运数据到共享内存，partition_S 是源内存， partition_D是目标内存
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA); // (CPY, CPY_M, CPY_K, num_tile_k)
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA); // (CPY, CPY_M, CPY_K, kStage)
    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB); // (CPY, CPY_N, CPY_K, num_tile_k)
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB); // (CPY, CPY_N, CPY_K, kStage)

    // from shared memory to register, use tiled_mma to generate tiled_copy
    // 从共享内存搬运数据到寄存器
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA {}, tiled_mma);
    auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA = s2r_thr_copy_a.partition_S(sA); // (CPY, CPY_M, CPY_K, kStage)
    // retile_D 重新解释寄存器的布局（Layout），使其适配搬运指令
    // retile_D作用：它把原本按照“计算需求”排列的寄存器索引（M, K 维度），重新映射成“搬运需求”的索引（Copy 维度）。
    auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA); // (CPY, CPY_M, CPY_K)

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB {}, tiled_mma);
    auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB = s2r_thr_copy_b.partition_S(sB); // (CPY, CPY_N, CPY_K, kStage)
    auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB); // (CPY, CPY_N, CPY_K)

    /* PREFETCH */
    // submit kStage - 1 tile
    // gmem -> shm
    int itile_to_read = 0;
    int ismem_read = 0;
    int ismem_write = 0;

#pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        // 从全局内存中搬运数据到共享内存
        cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage),
            tAsA_copy(_, _, _, istage));
        cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage),
            tBsB_copy(_, _, _, istage));
        // 每执行一次，硬件内部的任务计数器就会增加。它标记了一批传输任务的终点。
        cp_async_fence();

        ++itile_to_read;
        ++ismem_write;
    }

    // wait one submitted gmem->smem done
    // 它的意思是“请等待，直到剩下的待处理批次只剩下 N个。 配合 cp_async_fence, cp_async_fence记录有几个异步任务， cp_async_wait设置必须等到完成到还剩N个任务再往下执行
    cp_async_wait<kStage - 2>();
    __syncthreads();

    int ik = 0;
    // smem -> reg
    // tAsA: (CPY, CPY_M, CPY_K, kStage) tCrA_view: (CPY, CPY_M, CPY_K)
    cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
    cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

    // loop over k: i. load tile, ii. mma
    int ntile = k / BK;
#pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA); // (MMA, MMA_M, MMA_K)

#pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
            int ik_next = (ik + 1) % nk;

            if (ik == nk - 1) {
                cp_async_wait<kStage - 2>();
                __syncthreads();

                ismem_read = (ismem_read + 1) % kStage;
            }

            // shm -> reg s[itile][ik + 1] -> r[ik + 1]
            // tAsA: (CPY, CPY_M, CPY_K, kStage), tCrA_view: (CPY, CPY_M, CPY_K)
            cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read),
                tCrA_view(_, _, ik_next));
            // tBsB: (CPY, CPY_M, CPY_K, kStage), tCrB_view: (CPY, CPY_M, CPY_K)
            cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read),
                tCrB_view(_, _, ik_next));

            if (ik == 0) {
                if (itile_to_read < ntile) {
                    cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read),
                        tAsA_copy(_, _, _, ismem_write));
                    cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read),
                        tBsB_copy(_, _, _, ismem_write));
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }

                cp_async_fence();
            }

            cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
        } // for ik
    }

    // use less shared memory as a scratchpad tile to use large wide instuction
    // Dreg -> shm -> reg -> global
    auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC {});

    auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC {}, tiled_mma);
    auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
    auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD); // (CPY, CPY_M, CPY_N)
    auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC); // (CPY, _1, _1, pipe)

    S2GCopyC s2g_tiled_copy_c;
    auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
    auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC); // (CPY, _1, _1, pipe)
    auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD); // (CPY, CPY_M, CPY_N)

    auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g); // (CPY_, CPY_MN)
    auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s); // (CPY_, CPY_MN)

    int step = size<3>(tCsC_r2s); // pipe
#pragma unroll
    for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
// reg -> shm
#pragma unroll
        for (int j = 0; j < step; ++j) {
            // we add a temp tensor to cope with accumulator and output data type
            // difference
            auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
            cute::copy(tCrC_r2sx(_, i + j), t);

            cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
        }
        __syncthreads();

#pragma unroll
        // shm -> global
        for (int j = 0; j < step; ++j) {
            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
        }
        __syncthreads();
    } // end for
}

// For torch binding, need dynamic block swizzle stride
template <typename T, const int Stages = 2, const bool BlockSwizzle = false>
void launch_hgemm_mma_stages_block_swizzle_tn_cute(T* a,
    T* b,
    T* c,
    int M,
    int N,
    int K,
    int swizzle_stride)
{
    // block swizzle_stride: 1024/2048/..., etc.
    using namespace cute;

    auto BM = Int<128> {};
    auto BN = Int<256> {};
    auto BK = Int<32> {};
    auto KStage = Int<Stages> {}; // default 2
    auto kSmemLayoutCBatch = Int<4> {}; // namely, stages.

    // Define the smem layouts, Swizzle<3, 3, 3> and
    // Swizzle<2, 3, 3> will get the same results.
    // reference: https://zhuanlan.zhihu.com/p/671419093
    // Swizzle<3, 3, 3> 做了什么： 它通过位运算（通常是 XOR 异或）对地址进行变换。
    // 第一个 3(B)：表示参与变换的位宽（Base）。
    // 第二个 3(M)：表示变换的周期（Mask）。
    // 第三个 3(S)：表示左移的位数（Shift）。 简单来说，它把原本在内存里“排成直线”的数据，变成了“交叉分布”。这样当你按列读的时候，原本会撞车的线程被散开到了不同的 Bank 上。

    // 在 CuTe 中，composition(A, B) 的作用是**“把函数 B 的输出作为函数 A 的输入”**。
    // decltype(...) 的意思就是：“编译器，你自己算一下后面这一坨表达式推导出来的类型是什么，然后把这个类型给 SmemLayoutAtom 变量。”
    using SmemLayoutAtom
        = decltype(composition(
            Swizzle<3, 3, 3> {},
            make_layout(make_shape(Int<8> {}, Int<BK> {}),
            make_stride(Int<BK> {}, Int<1> {}))));
    using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom {},
        make_shape(Int<BM> {}, Int<BK> {}, Int<KStage> {})));
    using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom {},
        make_shape(Int<BN> {}, Int<BK> {}, Int<KStage> {}))); // (m,n) -> smem_idx
    // mma
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    static constexpr int kMmaEURepeatM = 2; // MMA repeat 2 times across M
    static constexpr int kMmaEURepeatN = 2; // MMA repeat 2 times across N
    static constexpr int kMmaEURepeatK = 1; // MMA no repeat across K

    using mma_atom_shape = mma_traits::Shape_MNK; // M,N,K 16,8,16
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape {}); // 1*2*16=32
    static constexpr int kMmaPN = 2 * kMmaEURepeatN * get<1>(mma_atom_shape {}); // 2*2*8 =32
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape {}); // 1*1*16=16
    // TiledMMA, more threads, MMAThrLayout(2,2,1), 4 MMA = 4 warps = 32x4 threads.
    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM> {}, Int<kMmaEURepeatN> {}, Int<kMmaEURepeatK> {})));
    // TiledMMA, more values, Permutations(32,32,16)
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom {}, MMA_EU_RepeatT {}, MMA_P_T {}));

    // copy from global memory to shared memory
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom {},
        make_layout(make_shape(Int<32> {}, Int<4> {}), // Thr layout 32x4 k-major
            make_stride(Int<4> {}, Int<1> {})),
        make_layout(make_shape(Int<1> {}, Int<8> {})))); // Val layout 1x8
    using G2SCopyB = G2SCopyA;
    using s2r_copy_op = SM75_U32x4_LDSM_N;
    using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
    using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;

    using SmemLayoutAtomC = decltype(composition(
        Swizzle<3, 3, 3> {},
        make_layout(make_shape(Int<kMmaPM> {}, Int<kMmaPN> {}), // 32*32
            make_stride(Int<kMmaPN> {}, Int<1> {}))));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC {},
        make_shape(Int<kMmaPM> {}, Int<kMmaPN> {}, Int<kSmemLayoutCBatch> {})));

    static_assert(
        size<0>(SmemLayoutA {}) * size<1>(SmemLayoutA {}) >= size(SmemLayoutC {}),
        "C shared memory request is large than A's one pipe");
    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;

    using S2GCopyAtomC = Copy_Atom<UniversalCopy<cute::uint128_t>, T>;
    using S2GCopyC = decltype(make_tiled_copy(
        S2GCopyAtomC {},
        make_layout(make_shape(Int<32> {}, Int<4> {}),
            make_stride(Int<4> {}, Int<1> {})),
        make_layout(make_shape(Int<1> {}, Int<8> {}))));

    int BX = (N + BN - 1) / BN;
    int BY = (M + BM - 1) / BM;
    // NOTE: Apply thread block swizzle across N dim.
    int BZ = BlockSwizzle ? (N + (swizzle_stride)-1) / (swizzle_stride) : 1;
    BX = BlockSwizzle ? (BX + BZ - 1) / BZ : BX;

    dim3 block(size(MMA {}));
    dim3 grid(BX, BY, BZ);

    // C_shm is shared with A_shm and B_shm
    // we don't allocate new smem for C_shm.
    // (128 * 32 * 2) * 2 + (256 * 32 * 2) * 2 = 49152 bytes, stages=2
    static constexpr int shm_size_AB = cute::cosize(SmemLayoutA {}) + cute::cosize(SmemLayoutB {});
    static constexpr int shm_size_C = cute::cosize(SmemLayoutC {});
    static constexpr int kShmSize = cute::max(shm_size_AB, shm_size_C) * sizeof(T);

    int shm_size = kShmSize;

    cudaFuncSetAttribute(
        hgemm_mma_stages_block_swizzle_tn_cute_kernel<
            T,
            BM, BN, BK,
            KStage,
            MMA,
            G2SCopyA,
            G2SCopyB,
            SmemLayoutA,
            SmemLayoutB,
            SmemLayoutC,
            S2RCopyAtomA,
            S2RCopyAtomB,
            R2SCopyAtomC,
            S2GCopyAtomC,
            S2GCopyC,
            BlockSwizzle>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shm_size);

    hgemm_mma_stages_block_swizzle_tn_cute_kernel<
        T,
        BM, BN, BK,
        KStage,
        MMA,
        G2SCopyA,
        G2SCopyB,
        SmemLayoutA,
        SmemLayoutB,
        SmemLayoutC,
        S2RCopyAtomA,
        S2RCopyAtomB,
        R2SCopyAtomC,
        S2GCopyAtomC,
        S2GCopyC,
        BlockSwizzle><<<grid, block, shm_size>>>(a, b, c, M, N, K);
}

// build cpp binary
#ifndef NO_CUTE_HGEMM_BIN

#include "utils.h"

int main()
{
    using T = cute::half_t;
    using namespace cute;
    const int test_num = 64;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];

    for (int i = 0; i < test_num; i++) {
        M_list[i] = (i + 1) * 256;
        N_list[i] = (i + 1) * 256;
        K_list[i] = (i + 1) * 256;
    }

    const int thread_block_swizzle_stride = 2048; // thread block swizzle stride
    printf("ALGO = CuTe HGEMM, TN, STAGES=2, SMEM SWIZZLE=<3, 3, 3>, BLOCK SWIZZLE=2048\n");
    int check_num = test_num > 5 ? 5 : 1;
    for (int j = 0; j < check_num; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];
        float max_error = gemm_error_check_tn_swizzle<T>(
            launch_hgemm_mma_stages_block_swizzle_tn_cute<T, 2, true>,
            M, N, K, thread_block_swizzle_stride);
        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Max Error = %f\n", max_error);
    }

#ifndef CUTE_HGEMM_DEBUG
    const int outer_repeat = 10, inner_repeat = 1;
    for (int j = 0; j < test_num; j++) {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++) {
            double this_sec = perf_gemm_swizzle<T>(
                launch_hgemm_mma_stages_block_swizzle_tn_cute<T, 2, true>,
                M, N, K, thread_block_swizzle_stride, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        // 1 TFLOPS = 10^12 FLOPS
        // ref: https://imgtec.eetrend.com/blog/2021/100062210.html.
        double avg_sec = total_sec / outer_repeat;
        double avg_Tflops = ((double)M) * N * K * 2 * 1e-12 / avg_sec;

        printf("M N K = %6d %6d %6d, ", M, N, K);
        printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
        printf("AVG Performance = %10.4lf Tflops\n", avg_Tflops);
    }
#endif

    return 0;
}