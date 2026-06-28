### SYCL XMX GEMM Q41



```c++
// Copyright (C) 2024 - 2026 Intel Corporation
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you ("License"). Unless the License provides otherwise,
// you may not use, modify, copy, publish, distribute, disclose or transmit
// this software or the related documents without Intel's prior written
// permission.

// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly stated
// in the License.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <map>

#include <windows.h>

using namespace sycl;
using fp16 = sycl::half;
#define DEVICE_MEM_ALIGNMENT 4096;


using namespace std;
using namespace sycl::ext::intel::esimd;

#define GROUP_SIZE 128

extern "C" bool runGemmXmx16_Q41Weights_L3(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned batch, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt);

ESIMD_INLINE void shuffle_input_quater(float *inputs, fp16 *outputs, uint32_t token_len, uint32_t input_len, nd_item<2>& ndi)
{
    const uint32_t blockWidth = 4096;
    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    uint32_t localRange = ndi.get_local_range(0);
    int localLinearId = ndi.get_local_linear_id();
		// h: input_len 方向的 4096-wide block
		// v: token 方向的 16-token block
		// localLinearId: 当前 group 内的第几个 work-item，范围 0..31
  
    uint32_t alignedTokenLen = (token_len + 255) / 256 * 256;

    uint32_t curBlockWidth = input_len - h * blockWidth; 
    // 当前 group 需要处理的 channel位置，每个group 4096
    // 每个group 32个 work item, 4096/32=128=64*2 每次处理 64个channel, 处理两次
    curBlockWidth = curBlockWidth > blockWidth ? blockWidth : curBlockWidth; 
    // 若 hidden_size 不能被4096整除，则 使用剩余的，而不是对其到 4096
		
  	// 每个 work-item 用两个 SIMD 寄存器：
    // 1024 因为每个 work-item 一次处理：16 tokens x 64 input channels = 1024 elements
    simd<float, 1024> inputData = 0;
    simd<fp16, 1024> outputData = 0;
    
    uint32_t v_blk_idx = v >> 4;
    uint32_t v_inn_idx = v & 0x0F;
  
    // 因为一个大 token block 是 256 tokens。
    // 而 shuffle kernel 的 v 每次只处理 16 tokens。
    // 所以：
		// 			v_blk_idx = v / 16
		// 			v_inn_idx = v % 16
		// 含义：
		// 			v_blk_idx: 当前属于第几个 256-token 大块； 
  	//							  v_range = token_len/16 
  	//							  v_blk_idx = v_range/16 = token_len/16/16 = token_len /256
		// 			v_inn_idx: 当前是这个 256-token 大块里的第几个 16-token 小块
    // 								v_range = token_len/16
    // 								v_inn_idx: 256 token分为 16个 16-tokens， 第n个 16tokens
  
    // 找到对应 token开始位置 v*16，即第v个 16 token block 每个token维度 input_len ,即  v * 16 * input_len
    // h * blockWidth 当前 work group 处理的对应channel 开始位置，即 h* blockWidth
    // localLinearId * 64 表示当前 work-item 负责的 64 个 input channel。
    uint32_t inputOffset = v * 16 * input_len + h * blockWidth + localLinearId * 64;
    
    // v_blk_idx * 256 * input_len: 第几个 256-token 大块。
    // h * 256 * blockWidth: 当前 input channel 的 4096 block，在一个 256-token 大块内的位置。
    // [token0:0-4096, token1:0-4096, ..., token255:0-4096][token0: 4096-8192, ..., token255: 4096-8192]
    // localLinearId * 256 * 64: 当前 work-item 的 64-channel 子块，在 256-token block 里的位置。
    // v_inn_idx * 16 * 16: 当前 16-token 小块在该重排布局里的位置。
		// 这个布局不是普通 [token][input]，而是为了后续 block_load<fp16, 128> 和 dpas 连续读取。
    uint32_t outputOffset = v_blk_idx * 256 * input_len + h * 256 * blockWidth +  localLinearId * 256 * 64 + v_inn_idx * 16 * 16;
    uint32_t readOffset = inputOffset;
    uint32_t writeOffset = outputOffset;

    // 确认当前 work-item 负责的 64-channel 起点没有超过 input_len。
    if (h * blockWidth + localLinearId * 64 < input_len) // 
    {
#pragma unroll
        // 读取 16 个 token
        for (int j = 0; j < 16; j++)
        {
            if (v * 16 + j < token_len)
            {	
              	// block_load<float, 64> 每次连续 64 个 input channel。
                inputData.select<64, 1>(j * 64) = block_load<float, 64>(inputs + readOffset);
                // readOffset += input_len 表示跳到下一个 token 的同一 input channel 起点。
                // 所以 inputOffset 没有考虑 token block中的第几个token，每个 work item 会负责16个token
                readOffset += input_len;
                // 所以 inputData 里面的布局是：
                // 		token0 的 64 个 input
                // 		token1 的 64 个 input
                // 		...
                // 		token15 的 64 个 input
                // 		即：
                // 		inputData[j][0..63] 		
            }
        }
        // 重排 + float 转 fp16 + 除以 4
#pragma unroll
        for (int j = 0; j < 16 && v*16+j < token_len; j++)
        {
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                outputData.select<16, 1>(k * 256 + j * 16) = inputData.select<16, 1>(j * 64 + k * 16) / 4.0; // to handle granite overflow issue
            }
          	// 每个 token 的 64 个 input channel 被拆成 4 份：
            // k = 0: input channel 0..15
            // k = 1: input channel 16..31
            // k = 2: input channel 32..47
            // k = 3: input channel 48..63
            // 读取：
            // inputData.select<16, 1>(j * 64 + k * 16)
            // 表示：
            // 第 j 个 token 的第 k 组 16 个 input
            // 写入：
            // outputData.select<16, 1>(k * 256 + j * 16)
            // 也就是把布局从：
            // [token][channel]
            // 变成：
            // [channel group][token]
            // 更具体地：
            // 原始:
            // token0: c0..c63
            // token1: c0..c63
            // ...
            // token15: c0..c63

            // 重排后:
            // k=0: token0 c0..c15, token1 c0..c15, ..., token15 c0..c15
            // k=1: token0 c16..c31, ..., token15 c16..c31
            // k=2: token0 c32..c47, ...
            // k=3: token0 c48..c63, ...
            // 这里 / 4.0 是为了规避 Granite 上的溢出问题。后面 GEMM 里会把 scales 和 zps 乘以 4 抵消掉。
        }

#pragma unroll
        // 写出 4 个 256 元素块
        for (int j = 0; j < 4; j++)
        {
            block_store<fp16, 256>(outputs + writeOffset, outputData.select<256, 1>(j * 256));
            writeOffset += 16 * 256;
          	// outputData 总共 1024 个元素，被分成 4 个 256 元素块。
            // 每个 256 元素块对应：
            // 16 tokens x 16 input channels
            // 写完一个块后：
            // writeOffset += 16 * 256;
            // 这里的 stride 是重排布局的一部分，让后续 GEMM 可以按需要连续读取。
        }
    }
		// 下面这一大段和上面几乎一样，只是处理另一个 64-channel 子块。
    inputData = 0.0;
    outputData = 0.0;
    inputOffset = v * 16 * input_len + h * blockWidth + (localLinearId + localRange)* 64;
    outputOffset = v_blk_idx * 256 * input_len + h * 256 * blockWidth +  (localRange + localLinearId) * 256 * 64 + v_inn_idx * 16 * 16;
    readOffset = inputOffset;
    writeOffset = outputOffset;
    if (h * blockWidth + (localRange + localLinearId) * 64 < input_len)
    {
#pragma unroll
        for (int j = 0; j < 16; j++)
        {
            if (v * 16 + j < token_len)
            {
                inputData.select<64, 1>(j * 64) = block_load<float, 64>(inputs + readOffset);
                readOffset += input_len;
            }
        }

#pragma unroll
        for (int j = 0; j < 16 && v*16+j < token_len; j++)
        {
#pragma unroll
            for (int k = 0; k < 4; k++)
            {
                outputData.select<16, 1>(k * 256 + j * 16) = inputData.select<16, 1>(j * 64 + k * 16) / 4.0; // to handle granite overflow issue
            }
        }

#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            block_store<fp16, 256>(outputs + writeOffset, outputData.select<256, 1>(j * 256));
            writeOffset += 16 * 256;
        }
    }

}

ESIMD_INLINE void gemm_q41weights_xmx16(fp16 *inputs, uint8_t *weights, fp16 *scales, fp16 *zps, float *outputs, uint32_t token_len, uint32_t input_len, uint32_t output_len, float* scratch, nd_item<2>& ndi)
{
    //slm_init(256 * 128 * sizeof(fp16) * 2);
    constexpr uint32_t slmOffsetScale = 0;
    constexpr uint32_t slmOffsetZps = 256 * 128 * sizeof(fp16);
    const uint32_t localRange = 32;

    float* debug = outputs + token_len * output_len;

    int h = ndi.get_group(0);
    int v = ndi.get_group(1);
    int localLinearId = ndi.get_local_linear_id();
    // 每个 work-item 负责 64 tokens x 32 output
    // hh 表示 哪个 32 output块 vv表示哪个 64 tokens 快
    int hh = localLinearId & 0x07; // 32%8 hh 表示当前线程在输出通道方向上的 32-channel block 编号，范围是 0..7。
    int vv = localLinearId >> 3;   // 32/8 vv 把 32 个线程分成 4 组，用于选择 256 token 里的哪 64 个。

    if (v * 256 + vv * 64 >= token_len)
    {
        return;
    }

    uint32_t globalOffsetInput = v * 256 * input_len; // 当前 token tile 的输入起点。
    uint32_t globalOffsetWeight = h * 256 * input_len; // 当前 output tile 对应权重起点。
  	// 可以把权重矩阵理解成二维：
		// weights[output_len][input_len] 所以 * input_len
    uint32_t globalOffsetOutput = 256 * v * output_len + 256 * h; // 当前输出 tile 起点。
		
    // 输入已经被预先 shuffle 成适合 XMX 的 layout。这里每个 vv 负责 64 token，对应 vv * 64 * 16 的偏移。
    // 16 代表shuffle_input 里的 内层维度 [token0-256][channel0-16]
    uint32_t localOffsetInput = globalOffsetInput + vv * 64 * 16; // 一组 64个token，每个token有16个channels
    // uint32_t localOffsetWeight = globalOffsetWeight/2 + hh * 32 * 16;
  	// 权重矩阵的输出通道坐标。当前 group 是 h * 256，当前线程负责其中 hh * 32。
    uint32_t localOffsetWeightY = h * 256 + hh * 32;
    // 权重矩阵输入通道方向坐标，从 0 开始，每轮增加 4 个 uint32_t 单位，对应 32 个 4-bit 权重块。
    uint32_t localOffsetWeightX = 0;
    
    uint32_t loopStep = 32;
    uint32_t loopNum = input_len / loopStep; // loopNum 是 K 维循环次数。

    uint32_t scaleOffset = (h * 256 + hh * 32) * input_len / 32; // 是当前输出通道块对应的 scale/zp 起点。
		// 因为 Q4_1 通常每 32 个输入元素一组 scale/zp，所以除以 32。
    // 它对应的是：
		// 			scales[output_channel][input_group]
		// 			zps[output_channel][input_group]
		// 其中 input_group 是按每 32 个 input channel 一组量化参数。
    // 需要了解 shuffle_Q41Weights_group32_L3 
  
  
    simd<fp16, 32> scales_buf; //保存 32 个 scale 和 32 个 zero-point。当前线程负责 32 个输出通道，所以一次读 32 个。
    simd<fp16, 32> zps_buf;
		
  	// AData0_tik 到 AData7_tik 对应 8 个 token 小块。每个是 128 个 fp16，用于 dpas。
    // AData0_tik:
    // 		8 个 token
    // 		每个 token 16 个 input-channel 元素
    // 		总共 8 × 16 = 128 fp16
    simd<fp16, 128> AData0_tik;
    simd<fp16, 128> AData1_tik;
    simd<fp16, 128> AData2_tik;
    simd<fp16, 128> AData3_tik;
    simd<fp16, 128> AData4_tik;
    simd<fp16, 128> AData5_tik;
    simd<fp16, 128> AData6_tik;
    simd<fp16, 128> AData7_tik;

  	// BData 是一次从 packed Q4 权重里加载的原始 byte 数据。
		// BData0/BData1 把 512 bytes 拆成两半，对应两个 16 输出通道块。
		// BData0_tik/BData1_tik 是低 4-bit nibble 解包并反量化后的权重。
    simd<uint8_t, 512> BData;
    simd<uint8_t, 256> BData0;
    simd<uint8_t, 256> BData1;

    simd<fp16, 256> BData0_tik;
    simd<fp16, 256> BData1_tik;
  
  
		// 双缓冲用的下一批 A/B 数据。
	  // tik 和 tok 是 ping-pong buffer：
    // tik: 当前低 nibble 或当前批
    // tok: 下一批或高 nibble
    simd<fp16, 128> AData0_tok;
    simd<fp16, 128> AData1_tok;
    simd<fp16, 128> AData2_tok;
    simd<fp16, 128> AData3_tok;
    simd<fp16, 128> AData4_tok;
    simd<fp16, 128> AData5_tok;
    simd<fp16, 128> AData6_tok;
    simd<fp16, 128> AData7_tok;

    simd<fp16, 256> BData0_tok;
    simd<fp16, 256> BData1_tok;
  
  
// 累加器 C。
// 共有 16 个累加器：
// 8 个 token block × 2 个 output half-block
// 命名规律：
// CDataXY
// X = 0..7：第几个 token 子块。
// Y = 0/1：输出通道的前 16 或后 16。
// 每个累加器是 simd<float, 128>，最终 store 成 16 x 8 的输出 tile。
    simd<float, 128> CData00 = 0.0;
    simd<float, 128> CData10 = 0.0;
    simd<float, 128> CData20 = 0.0;
    simd<float, 128> CData30 = 0.0;
    simd<float, 128> CData40 = 0.0;
    simd<float, 128> CData50 = 0.0;
    simd<float, 128> CData60 = 0.0;
    simd<float, 128> CData70 = 0.0;

    simd<float, 128> CData01 = 0.0;
    simd<float, 128> CData11 = 0.0;
    simd<float, 128> CData21 = 0.0;
    simd<float, 128> CData31 = 0.0;
    simd<float, 128> CData41 = 0.0;
    simd<float, 128> CData51 = 0.0;
    simd<float, 128> CData61 = 0.0;
    simd<float, 128> CData71 = 0.0;

// 预加载第一批输入 A。
// 一次加载 8 组，每组 128 个 fp16。
// 这些对应当前线程负责的 64 token 区域。 128*8/16 = 64
// t0[c0-c15] t1[c0-c15]..t255[c0-c15] 
// 128/16 = 8 所以 ADataN_tik 一次有8个token,一共8个 ADataN_tik 共64个token
    AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
    AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
    AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
    AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
    AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
    AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
    AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
    AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);
  
// 从 packed Q4 权重矩阵中用 2D load 读取一块权重。
// load_2d<uint32_t, 4, 32> 表示读取一个 4 × 32 的 uint32_t block。
// 因为每个 uint32_t 是 4 bytes，所以总共：
// 4 * 32 * 4 = 512 bytes
// 这些 bytes 里面每个 byte 有两个 4-bit 权重。
// bit_cast_view<T>() 的意思是：不改变底层 bits，只换一种类型视角来看同一段 SIMD 数据。
  
// typedef struct {
//     union {
//         struct {
//             sycl::half d;  // delta
//             sycl::half m;  // min
//         };

//         uint32_t dm;
//     };

//     uint8_t qs[32 / 2];  // nibbles / quants for 32 elements
// } block_q4_1_32;

// uint8_t qs[16] => uint32_t qs[4] 所以 一次读取 32行，一行一个block_q4_1 
    BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);
  
// 加载当前 32 个输出通道对应的 scale 和 zero-point。
// SLM 版本被注释掉了，现在直接从 global memory load。
    scales_buf = block_load<fp16, 32>(scales + scaleOffset);
    zps_buf = block_load<fp16, 32>(zps + scaleOffset);

// 把 scale 和 zp 乘以 4。
// 原因对应前面的 shuffle_input_quater：
// input / 4.0
// 输入为了避免 Granite overflow 被除以 4，这里权重反量化参数乘以 4 来补偿整体乘积。
    scales_buf *= 4;
    zps_buf *= 4;

// 把 BData 中的 packed 数据重排到 BData0/BData1。
// 这里用了 bit_cast_view<uint16_t>()，按 16-bit 粒度重新解释数据。
// 每轮 j 处理一段。
// BData0 取前半部分，BData1 取后半部分。
// 这一步主要是把 load_2d 得到的 layout 转成 dpas 期望的 B operand layout。
#pragma unroll
    for (int j = 0; j < 4; j++)
    {
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
        BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
        BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
    }

// 准备下一轮 K 维循环。
// localOffsetInput += 4096，对应输入 layout 中前进一个 K block。
// localOffsetWeightX += 4，权重 2D load 的 X 方向前进。
// scaleOffset += 32，scale/zp 前进到下一组 K block。
    localOffsetInput += 16 * 256;
    localOffsetWeightX += 4;
    scaleOffset += 32; // 32 是因为 output channel 方向 移动32步，input channel还是1

// 提取每个 byte 的低 4-bit 权重。
// 低 nibble：
// byte & 0x0f
// 得到的还是整数值 0..15，随后会反量化。
    BData0_tik = BData0 & 0x0f;
    BData1_tik = BData1 & 0x0f;

  
// 构造广播后的 scale/zp。
// BData0_tik 每 32 个元素需要对应 16 个 scale 重复两次。
// 所以这里把 scales_buf[0..15] 扩展成 32 个元素：
// s = [s0, s0, s1, s1, ..., s15, s15]
// z 同理。
    simd<fp16, 32> s;
    simd<fp16, 32> z;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
    z.select<16, 2>(1) = z.select<16, 2>(0);

// 对 BData0_tik 的低 nibble 权重做反量化：
// dequant = q * scale + zp
// 每次处理 32 个元素，共 8 次，总计 256 个元素。
// 低 nibble 权重 指的是：一个 uint8_t 字节里低 4 bit 存的那个 4-bit 权重值。
// nibble 就是半个 byte：
//   1 byte = 8 bits
//   1 nibble = 4 bits
// 所以一个 uint8_t 可以放两个 4-bit 数：
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s  + z;
    }

// 对 BData1_tik 做同样的反量化。
// 区别是使用 scales_buf[16..31] 和 zps_buf[16..31]，对应后 16 个输出通道。
    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s  + z;
    }

// 主 K 维循环。
// 因为每个 4-bit byte 有两个权重：
// 低 nibble
// 高 nibble
// 代码采用流水方式：
// 先处理当前 low nibble，再处理 high nibble，同时预取下一轮数据。
    for (int i = 0; i < loopNum - 1; i++)
    {	
      	// 加载下一批输入 A 到 tok buffer。
        AData0_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);
			
      	// 提取每个 byte 的高 4-bit 权重：
        BData0_tok = BData0 >> 4;
        BData1_tok = BData1 >> 4; // 这就是另一个 Q4 权重值。
				
        // 对高 nibble 的 BData0_tok/BData1_tok 做反量化。 
        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tok.select<32, 1>(j * 32) = BData0_tok.select<32, 1>(j * 32) * s  + z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
    #pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tok.select<32, 1>(j * 32) = BData1_tok.select<32, 1>(j * 32) * s  + z;
        }

        //scales_buf = slm_block_load<fp16, 32>(slmOffsetScale + scaleOffset);
        // 提前加载下一轮 K block 的 scale/zp，并乘以 4 补偿 input 除以 4。
        scales_buf = block_load<fp16, 32>(scales + scaleOffset);
        zps_buf = block_load<fp16, 32>(zps + scaleOffset);

        scales_buf *= 4;
        zps_buf *= 4;
        
        // 输入偏移前进到下一批。
        localOffsetInput += 16 * 256;
				
      	
      	// 用 XMX dpas 执行矩阵乘累加。
        // 这些行处理 BData0_tik，也就是输出通道前 16 个。
        // 每个 ADataX_tik 对应不同 token 子块。
        // 数学上是：
        // C += B_low_half * A
        CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tik, AData0_tik);
        CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tik, AData1_tik);
        CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tik, AData2_tik);
        CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tik, AData3_tik);
        CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tik, AData4_tik);
        CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tik, AData5_tik);
        CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tik, AData6_tik);
        CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tik, AData7_tik);
				
      	// 同样做 dpas，但处理 BData1_tik，也就是输出通道后 16 个。
        CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tik, AData0_tik);
        CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tik, AData1_tik);
        CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tik, AData2_tik);
        CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tik, AData3_tik);
        CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tik, AData4_tik);
        CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tik, AData5_tik);
        CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tik, AData6_tik);
        CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tik, AData7_tik);

				// 再预取下一批 A 到 tik buffer。
				// 这是流水线结构：一边算 tok，一边准备 tik。
        AData0_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
        AData1_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
        AData2_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
        AData3_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
        AData4_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
        AData5_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
        AData6_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
        AData7_tik = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);
        
        // 加载下一批 packed Q4 权重。
        BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);

				// 再次把刚加载的 BData 重排成 BData0/BData1。
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32);
            BData0.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 1);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32);
            BData1.bit_cast_view<uint16_t>().select<16, 1>(j * 2 * 16 + 16) = BData.bit_cast_view<uint16_t>().select<16, 2>(j * 2 * 32 + 32 + 1);
        }
				// 继续推进输入、权重、scale/zp 偏移。
        localOffsetInput += 16 * 256;
        localOffsetWeightX += 4;
        scaleOffset += 32;
				
        // 提取下一批权重的低 nibble，并反量化。
        BData0_tik = BData0 & 0x0f;
        BData1_tik = BData1 & 0x0f;
        s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
        z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData0_tik.select<32, 1>(j * 32) = BData0_tik.select<32, 1>(j * 32) * s  + z;
        }

        s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
        s.select<16, 2>(1) = s.select<16, 2>(0);
        z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
        z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
        for (int j = 0; j < 8; j++)
        {
            BData1_tik.select<32, 1>(j * 32) = BData1_tik.select<32, 1>(j * 32) * s  + z;
        }
				
      
        // 计算上一批的高 nibble 权重 BData0_tok。
        CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tok, AData0_tok);
        CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tok, AData1_tok);
        CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tok, AData2_tok);
        CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tok, AData3_tok);
        CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tok, AData4_tok);
        CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tok, AData5_tok);
        CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tok, AData6_tok);
        CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tok, AData7_tok);

        CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tok, AData0_tok);
        CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tok, AData1_tok);
        CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tok, AData2_tok);
        CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tok, AData3_tok);
        CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tok, AData4_tok);
        CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tok, AData5_tok);
        CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tok, AData6_tok);
        CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tok, AData7_tok);


    }
		// 主循环结束后，还需要处理最后一批高 nibble。
		// 这里加载最后的 A 到 tok。
    AData0_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 0);
    AData1_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 1);
    AData2_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 2);
    AData3_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 3);
    AData4_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 4);
    AData5_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 5);
    AData6_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 6);
    AData7_tok = block_load<fp16, 128>(inputs + localOffsetInput + 128 * 7);
		
  	// 提取最后一批 packed 权重的高 nibble。
    BData0_tok = BData0 >> 4;
    BData1_tok = BData1 >> 4;

    s.select<16, 2>(0) = scales_buf.select<16, 1>(0);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(0);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData0_tok.select<32, 1>(j * 32) = BData0_tok.select<32, 1>(j * 32) * s  + z;
    }

    s.select<16, 2>(0) = scales_buf.select<16, 1>(16);
    s.select<16, 2>(1) = s.select<16, 2>(0);
    z.select<16, 2>(0) = zps_buf.select<16, 1>(16);
    z.select<16, 2>(1) = z.select<16, 2>(0);
#pragma unroll
    for (int j = 0; j < 8; j++)
    {
        BData1_tok.select<32, 1>(j * 32) = BData1_tok.select<32, 1>(j * 32) * s  + z;
    }
    // 推进输入偏移。这里对后续计算其实基本没有用了，属于流水代码残留。 
    localOffsetInput += 16 * 256;
		
    // 处理最后一批高 nibble 权重。
		// 到这里，整个 K 维度全部累加完成。
    CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tik, AData0_tik);
    CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tik, AData1_tik);
    CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tik, AData2_tik);
    CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tik, AData3_tik);
    CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tik, AData4_tik);
    CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tik, AData5_tik);
    CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tik, AData6_tik);
    CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tik, AData7_tik);

    CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tik, AData0_tik);
    CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tik, AData1_tik);
    CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tik, AData2_tik);
    CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tik, AData3_tik);
    CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tik, AData4_tik);
    CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tik, AData5_tik);
    CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tik, AData6_tik);
    CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tik, AData7_tik);

    CData00 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData00, BData0_tok, AData0_tok);
    CData10 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData10, BData0_tok, AData1_tok);
    CData20 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData20, BData0_tok, AData2_tok);
    CData30 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData30, BData0_tok, AData3_tok);
    CData40 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData40, BData0_tok, AData4_tok);
    CData50 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData50, BData0_tok, AData5_tok);
    CData60 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData60, BData0_tok, AData6_tok);
    CData70 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData70, BData0_tok, AData7_tok);

    CData01 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData01, BData1_tok, AData0_tok);
    CData11 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData11, BData1_tok, AData1_tok);
    CData21 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData21, BData1_tok, AData2_tok);
    CData31 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData31, BData1_tok, AData3_tok);
    CData41 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData41, BData1_tok, AData4_tok);
    CData51 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData51, BData1_tok, AData5_tok);
    CData61 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData61, BData1_tok, AData6_tok);
    CData71 = xmx::dpas<8, 8, float, float, fp16, fp16>(CData71, BData1_tok, AData7_tok);

		// 计算当前线程输出 tile 的起始坐标。
		// X 是输出通道方向：
		// h * 256 + hh * 32
		// Y 是 token 方向：
		// v * 256 + vv * 64
    uint32_t localOutputOffsetX = h * 256 + hh * 32;
    uint32_t localOutputOffsetY = v * 256 + vv * 64;
		
  	// 默认直接写到最终输出矩阵。surfaceHeight 是 2D store 使用的矩阵高度。
    float *write_buffer = outputs;
    uint32_t surfaceHeight = token_len;
    
  	// 处理尾部 token。
		// 如果当前线程负责的 64 token 超过真实 token_len，不能直接用 store_2d 写到 outputs，否则可能越界。
		// 所以先写到 scratch，然后外层 runGemmXmx16_Q41Weights_L3 在第 569 行再 memcpy 有效部分回 outputs。
    if (v * 256 + vv * 64 + 64 > token_len)
    {
        write_buffer = scratch;
        localOutputOffsetY = 0;
        surfaceHeight = 64;
    }
		
  	// 把 16 个累加器写回输出矩阵。
    // 每个 store_2d<float, 16, 8> 写一个：
    // 16 输出通道 × 8 token
    // 每个 token 子块写两次：
    // CDataX0 -> 前 16 个输出通道
    // CDataX1 -> 后 16 个输出通道
    // 每写完一组 token 子块：
    // localOutputOffsetY += 8;
    // 所以 CData00/01 写 token 0..7，CData10/11 写 token 8..15，一直到 CData70/71 写 token 56..63。
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData00);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData01);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData10);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData11);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData20);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData21);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData30);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData31);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData40);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData41);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData50);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData51);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData60);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData61);
    localOutputOffsetY += 8;
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX, localOutputOffsetY, CData70);
    store_2d<float, 16, 8>(write_buffer, output_len * sizeof(float) - 1, surfaceHeight, output_len * sizeof(float) - 1, localOutputOffsetX + 16, localOutputOffsetY, CData71);


}

// token_len: 输入token 长度
// input_len: 输入的维度 hidden_size
// output_len: 输出的维度


bool runGemmXmx16_Q41Weights_L3(sycl::queue* q, uint8_t* inputs, uint8_t* weights, uint8_t* scales, uint8_t* zps, uint8_t* outputs, unsigned token_len, unsigned input_len, unsigned output_len, unsigned input_precision, unsigned output_precision,  uint8_t* shuffleTt)
{
    const uint32_t blockWidth = 4096;
    uint32_t threadShuffleH = (input_len + blockWidth - 1)/blockWidth; // 每个 work-group处理4096个通道， x方向
    uint32_t alignedTokenLen = (token_len + 255)/256 * 256; // token_len对其到256 
    uint32_t threadShuffleV = alignedTokenLen / 16; // 每个work-group 处理 16个token y方向
    uint32_t localThreadShuffle = 32;

    uint8_t* outputScratch = shuffleTt + alignedTokenLen * input_len * sizeof(fp16);

    sycl::range<2> GlobalRangeShuffle(localThreadShuffle * threadShuffleH, threadShuffleV);
    sycl::range<2> LocalRangeShuffle(localThreadShuffle, 1);
    sycl::nd_range<2> RangeShuffle(GlobalRangeShuffle, LocalRangeShuffle);
  
  	// shuffle 的 range 是按：
		// 16 tokens x 4096 input 来切。

    try {
        sycl::event e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(RangeShuffle, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        shuffle_input_quater((float *)inputs, (fp16 *)shuffleTt, token_len, input_len, ndi);
                    });
                });
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    uint32_t threadA = (token_len + 255)/256;
    uint32_t threadB = (output_len + 255)/256;
    uint32_t localThreadNum = 32;

    sycl::range<2> GlobalRange(localThreadNum * threadB, threadA);
    sycl::range<2> LocalRange(localThreadNum, 1);
    sycl::nd_range<2> Range(GlobalRange, LocalRange);
		// gemm 的 range 是按：
		// 256 tokens x 256 output来切
    // 每个 group 32 个 work-item，每个 work-item 算： 64 tokens x 32 output

    try {
        sycl::event e = q->submit([&](handler& cgh) {
                    cgh.parallel_for(Range, [=](nd_item<2> ndi) SYCL_ESIMD_KERNEL{
                        gemm_q41weights_xmx16((fp16 *)shuffleTt, (uint8_t *)weights, (fp16 *)scales, (fp16 *)zps, (float *)outputs, token_len, input_len, output_len, (float *)outputScratch, ndi);
                    });
                });
    } catch (sycl::exception const &e) {
        std::cout << "SYCL exception caught: " << e.what() << '\n';
        return false;
    }

    uint32_t tailStart = token_len/64*64;
    uint32_t tailLen = token_len - tailStart;
    if (tailLen > 0)
    {
        q->memcpy((float *)outputs + tailStart * output_len, outputScratch, tailLen * output_len * sizeof(float));
    }
}
```







shuffle_input_quater(...) 先把输入从普通布局重排到 XMX 友好的布局，并且从 float 转成 fp16。
gemm_q41weights_xmx16(...) 执行真正的 GEMM。

最后处理 token tail：
uint32_t tailStart = token_len / 64 * 64;
uint32_t tailLen = token_len - tailStart;

if (tailLen > 0) {
    q->memcpy(outputs + tailStart * output_len,
              outputScratch,
              tailLen * output_len * sizeof(float));
}
因为 kernel 内部按 64 个 token 为一个小块写结果，最后不足 64 token 时先写到 scratch，再 copy 回真正输出。

#### 输入输出映射

用完整一维公式表示 outputs
对于原始元素：
inputs\[token][channel]
令：
t = token;
c = channel;
那么在 outputs 中的位置是：
token_256_block    = t / 256;
token_in_256       = t % 256;
token_16_block     = token_in_256 / 16;
token_inside_16    = token_in_256 % 16;

channel_4096_block = c / 4096;
channel_in_4096    = c % 4096;
channel_64_block   = channel_in_4096 / 64;
channel_in_64      = channel_in_4096 % 64;
channel_16_group   = channel_in_64 / 16;
channel_inside_16  = channel_in_64 % 16;
最终：
output_index =
    token_256_block    * 256 * 8192
  + channel_4096_block * 256 * 4096
  + channel_64_block   * 256 * 64
  + channel_16_group   * 16 * 256
  + token_16_block     * 16 * 16
  + token_inside_16    * 16
  + channel_inside_16;
这个公式就是 shuffle_input_quater 输出布局的完整一维表达。

```c++
for token_256_block in 0..3:
  for channel_4096_block in 0..1:
    for channel_64_block in 0..63:
      for channel_16_group in 0..3:
        for token_16_block in 0..15:
          for token_inside_16 in 0..15:
            for channel_inside_16 in 0..15:
              store input[token][channel] / 4
```







```c++
BData.bit_cast_view<uint32_t>() = load_2d<uint32_t, 4, 32, 1, true, false>((uint32_t *)weights, input_len/2 - 1, output_len -1, input_len/2 - 1, localOffsetWeightX, localOffsetWeightY);  详细解释一下 这行的具体参数 以及实现的功能

```
这行是在从 weights 里按 2D tile 方式加载一块 Q4 weight 数据，并放进 BData：
BData.bit_cast_view<uint32_t>() =
    load_2d<uint32_t, 4, 32, 1, true, false>(
        (uint32_t *)weights,
        input_len/2 - 1,
        output_len - 1,
        input_len/2 - 1,
        localOffsetWeightX,
        localOffsetWeightY);
可以拆成两部分看。
左边
BData.bit_cast_view<uint32_t>()
BData 定义是：
simd<uint8_t, 512> BData;
也就是 512 bytes。
但 load_2d<uint32_t, 4, 32, 1, ...> 返回的是：
4 * 32 * 1 = 128 个 uint32_t
总大小：
128 * 4 bytes = 512 bytes
所以这里用 bit_cast_view<uint32_t>() 把 BData 临时看成 simd<uint32_t, 128>，正好接住这次 2D load 的结果。
模板参数
load_2d<uint32_t, 4, 32, 1, true, false>
含义大致是：
T = uint32_t
BlockWidth = 4
BlockHeight = 32
NBlocks = 1
Transpose = true
Transform = false
也就是从二维内存中加载：
32 行，每行 4 个 uint32_t
原始数据量：
32 * 4 * 4 bytes = 512 bytes
这对应 Q4 weight 的一块：
32 output channels x 32 input-K values
因为 Q4 是 4bit，一个 byte 存两个 weight，所以：
32 K values = 16 bytes = 4 uint32_t
因此每个 output channel 一行，需要读 4 个 uint32_t。
运行时参数
(uint32_t *)weights
把 weight buffer 当成 uint32_t* 读。不是改变数据，只是为了按 32-bit 粒度 load。
input_len/2 - 1
二维 surface 的宽度减 1。因为 Q4 weight 每个 K 只占 4bit，所以一行的 byte 数是：
input_len / 2
output_len - 1
二维 surface 的高度减 1。weight 矩阵高度是 output channel 数。
input_len/2 - 1
二维 surface 的 pitch 减 1，也就是每一行之间的跨度。这里 weight 是紧密排布的，所以 pitch 等于 width：
每个 output channel 占 input_len / 2 bytes
localOffsetWeightX
当前 K 方向的偏移。初始化是：
uint32_t localOffsetWeightX = 0;
每轮 K loop 后：
localOffsetWeightX += 4;
因为每轮处理 32 个 Q4 K：
32 q4 values = 16 bytes = 4 uint32_t
所以每次向右移动 4 个 uint32_t。
localOffsetWeightY
当前 output channel 方向的起始行：
uint32_t localOffsetWeightY = h * 256 + hh * 32;
含义是：

- h * 256：当前 work-group 负责的 256 个 output channel block
- hh * 32：当前 work-item 负责其中的 32 个 output channel
所以这一行加载的是：
weights[localOffsetWeightY : localOffsetWeightY + 31]
       [localOffsetWeightX : localOffsetWeightX + 3 uint32]
也就是：
32 个 output channel
每个 channel 读取 32 个 Q4 K values
true, false 的作用
true
表示做 transpose 式的 2D load。也就是说，内存中是按：
output channel 行
K packed 列
存的，但加载到 SIMD register 后，会按更适合后续 dpas/shuffle 的顺序排列。
false
表示不做额外 transform，比如不做 VNNI transform。这里只有普通 2D load + transpose。
这行最终实现的功能
它从 packed Q4 weight 矩阵中取出一个 tile：
32 output channels x 32 K values
因为 Q4 packed，所以实际读：
32 行 x 16 bytes = 512 bytes
读完后放进：
simd<uint8_t, 512> BData
后面再拆成低 4bit 和高 4bit：
BData0_tik = BData0 & 0x0f; // low nibble
BData0_tok = BData0 >> 4;   // high nibble
然后乘 scale/zp，转成 fp16，送进：
xmx::dpas(...)
一句话：  
这行就是 从 Q4 packed weight 矩阵中按 2D tile 方式高效加载当前 work-item 需要的 32x32 weight block，并重排成后续 XMX dpas 更容易消费的 SIMD 布局。



	load_2d<
	T,
	BlockWidth,
	BlockHeight,
	NBlocks,
	Transposed,
	Transformed>
	(
	base,
	surface_width_minus_1,
	surface_height_minus_1,
	surface_pitch_minus_1,
	x,
	y
	)


>对应这里：
>load_2d<uint32_t, 4, 32, 1, true, false>(
>(uint32_t *)weights,
>input_len/2 - 1,
>output_len - 1,
>input_len/2 - 1,
>localOffsetWeightX,
>localOffsetWeightY);
>模板参数
>uint32_t
>表示这次 load 的元素类型。返回值逻辑上是 simd<uint32_t, 4 * 32 * 1>，也就是 128 个 uint32_t。
>4
>BlockWidth，每一行读取 4 个 uint32_t。
>注意这里是以 T 为单位，不是 4 bytes。因为 T = uint32_t，所以实际每行读取：
>4 * sizeof(uint32_t) = 16 bytes
>对 Q4 weight 来说，16 bytes = 32 个 4bit weight。
>32
>BlockHeight，读取 32 行。
>这里每一行对应一个 output channel，所以是一次读取 32 个 output channel。
>1
>NBlocks，读取几个这样的 2D block。这里是 1，所以只取一个 4 x 32 的 block。
>true
>Transposed，表示 load 回来的寄存器布局做转置式排列。
>内存里是二维：
>row = output channel
>col = packed K
>正常 load 的自然顺序类似：
>row0 col0..3
>row1 col0..3
>...
>row31 col0..3
>Transposed = true 后，寄存器里会变成更适合后续 SIMD/XMX 使用的顺序。它不改变全局内存，只改变 load 到 simd 里的元素排列。
>false
>Transformed，表示不启用额外的数据 transform。常见 transform 是给 VNNI/DPAS 准备的特殊格式变换。这里已经自己在后面用 select、& 0x0f、>> 4 做 Q4 unpack/reorder，所以这里不让硬件额外 transform。
>运行时参数
>(uint32_t *)weights
>base pointer。2D surface 的起始地址。
>虽然原始 weights 是 uint8_t*，这里 cast 成 uint32_t*，是因为这次 load 以 uint32_t 为元素单位读取。
>input_len/2 - 1
>这是 surface_width_minus_1。
>它告诉 load_2d：这个 2D surface 每一行的有效宽度是多少。
>因为 Q4 weight 是 4bit，一个 byte 存两个 K，所以一行实际 byte 数是：
>input_len / 2
>API 要求传 width - 1，所以是：
>input_len / 2 - 1
>它具体干嘛用？
>主要有三件事：
1. 定义一行的合法访问范围。
2. 帮硬件判断 x + block_width 是否越过 surface 右边界。
3. 和 surface_pitch 一起构造 2D block load 的地址描述符。
它不是简单地参与 C++ 指针加法，而是作为硬件 2D load 的 descriptor 参数。
output_len - 1
这是 surface_height_minus_1。
它告诉硬件这个 surface 有多少行。这里一行是一个 output channel，所以高度是 output_len。
同样 API 要求传 height - 1。
input_len/2 - 1
这是 surface_pitch_minus_1。
pitch 是相邻两行之间的跨度。这里 weight 是紧密排列的：
weights[output_channel][packed_k]
每个 output channel 一行，每行 input_len / 2 bytes，所以 pitch 也是：
input_len / 2
API 要求 pitch - 1，所以传：
input_len/2 - 1
它具体干嘛用？
硬件读取第 r 行时，会类似这样算地址：
row_base = base + (y + r) * pitch
addr     = row_base + x
所以 surface_pitch_minus_1 决定了从一行跳到下一行要跨多少 bytes。
localOffsetWeightX
这是 2D load 的 X 坐标，也就是从每一行的哪个位置开始读。
这里每次读取 4 个 uint32_t：
4 uint32_t = 16 bytes = 32 个 Q4 weight
所以 loop 里每轮：
localOffsetWeightX += 4;
表示 K 方向向后移动 32 个 Q4 元素。
localOffsetWeightY
这是 2D load 的 Y 坐标，也就是从第几个 output channel 行开始读。
localOffsetWeightY = h * 256 + hh * 32;
含义是当前 work-item 读取当前 output block 内的 32 行。
这行整体做的事
它让硬件从 weights 这个二维 surface 中读取：
起点: (x = localOffsetWeightX, y = localOffsetWeightY)

范围:
32 行
每行 4 个 uint32_t
也就是：
32 output channels x 16 bytes
= 32 output channels x 32 个 Q4 weight
= 512 bytes
然后把这 512 bytes 放进 BData：
simd<uint8_t, 512> BData;
左边用：
BData.bit_cast_view<uint32_t>()
只是为了用 simd<uint32_t, 128> 的视角接住 load_2d 返回值。底层还是同一组 512 bytes。