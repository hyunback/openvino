// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/int4_utils.cl"

KERNEL(fc)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const int M = FILTER_OFM_NUM;
    const int K = FILTER_IFM_NUM;
    const int lid = get_sub_group_local_id();
    const int group_id = get_group_id(0);
    const int m_base = (group_id * 16);

    if (m_base >= M) return;

    float16 local_acc = 0.0f;

    // --- Decompression Parameters Loading ---
#if COMPRESSED_WEIGHTS
    float16 ds = 1.0f;
    float16 dzp = 0.0f;

#if DECOMPRESSION_SCALE_TERM
    for (int i = 0; i < 16; ++i) {
        if (m_base + i < M) {
            ((float*)&ds)[i] = (float)decompression_scale[m_base + i];
        }
    }
    // #pragma unroll   // Debugging code
    // for (int i = 0; i < 16; ++i) {
    //     float current_scale = ds[i];
    //     if (lid == 0) {
    //         output[m_base + i] = (OUTPUT_TYPE)current_scale; // only 0~15 shows 1, others 0(weird)
    //         // output[m_base + i] = (OUTPUT_TYPE)decompression_scale[m_base + i];  // all shows 1, okay!
    //         // output[m_base + i] = (OUTPUT_TYPE)m_base;   // correctly shows 0, 16, 32, ...
    //     }
    // }
    // return;
#endif  // DECOMPRESSION_SCALE_TERM

#if 0   // Debugging code for scale loading
    float ds_local[16];
    #if DECOMPRESSION_SCALE_TERM
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int m_idx = m_base + i;
            if (m_idx < M) {
                ds_local[i] = (float)decompression_scale[m_idx];
            } else {
                ds_local[i] = 1.0f;
            }
        }
    #else
        #pragma unroll
        for (int i = 0; i < 16; ++i) ds_local[i] = 1.0f;
    #endif
#endif

#if DECOMPRESSION_ZP_TERM
    #if DECOMPRESSION_ZP_SCALAR
        dzp = (float)DECOMPRESSION_ZP_VALUE;
    #else
        dzp = convert_float16(vload16(0, decompression_zp + m_base));
    #endif
#endif  // DECOMPRESSION_ZP_TERM
#endif  // COMPRESSED_WEIGHTS

#if 1
    // --- Main Loop ---
    for (int k = 0; k < K; k += 16) {
        int current_k = k + lid;

        if (current_k < K) {
            float in_val = (float)input[current_k];
            if (in_val == 0.0f) continue;

            long weight_offset = (long)current_k * (long)FILTER_IFM_PITCH + (long)m_base;
            float16 decompressed_weight;
#if COMPRESSED_WEIGHTS
            // INT8 Case: Load as char16 -> Convert to float16 -> Decompress
            float16 raw_weight = convert_float16(vload16(0, weights + weight_offset));
            decompressed_weight = (raw_weight - dzp) * ds;
#else
            // FP16 Case: Load as half16 -> Convert to float16 directly
            decompressed_weight = convert_float16(vload16(0, weights + weight_offset));
#endif
            local_acc += in_val * decompressed_weight;

            // // Trial1. All inputs are 1.f, All weights are fixed to 1.0f (Ignoring Scale/ZP)
            // float in_val = 1.0f;
            // float in_val = (float)input[current_k];
            // float16 decompressed_weight = (float16)1.0f;
            // local_acc += in_val * decompressed_weight;

            // // Trial2.
            // float in_val = 1.0f;
            // // Actual weight load
            // long weight_offset = (long)current_k * (long)FILTER_IFM_PITCH + (long)m_base;
            // float16 raw_weight = convert_float16(vload16(0, weights + weight_offset));
            // local_acc += in_val * raw_weight;

            // float in_val = 1.0f;
            // float16 raw_weight_vec;
            // for (int i = 0; i < 16; ++i) {
            //     int target_m = m_base + i;
            //     long final_offset = (long)current_k * (long)FILTER_IFM_PITCH + (long)target_m;\
            //     ((float*)&raw_weight_vec)[i] = (float)weights[final_offset];
            // }
            // local_acc += in_val * raw_weight_vec;

            // // ACC Okay, vload16 with int8 has the issue since 1 byte not aligned...
            // float in_val = (float)input[current_k];
            // // if (in_val == 0.0f) continue;
            // long weight_offset = (long)current_k * (long)FILTER_IFM_PITCH + (long)m_base;
            // float16 raw_weight_vec;
            // #pragma unroll
            // for (int i = 0; i < 16; ++i) {
            //     ((float*)&raw_weight_vec)[i] = (float)weights[weight_offset + i];
            // }
            // local_acc += in_val * raw_weight_vec;


            // // Trial3. Block Read + Unpacking -> ACC failed
            // float in_val = (float)input[k];
            // // if (in_val == 0.0f) continue;
            // long weight_offset = (long)k * (long)FILTER_IFM_PITCH + (long)m_base;
            // float16 raw_weight_vec;
            // uint4 packed_w = intel_sub_group_block_read4((__global uint*)(weights + weight_offset));
            // #pragma unroll
            // for (int i = 0; i < 4; i++) {
            //     uint val = ((uint*)&packed_w)[i];
            //     ((float*)&raw_weight_vec)[i*4 + 0] = (float)(char)(val & 0xFF);
            //     ((float*)&raw_weight_vec)[i*4 + 1] = (float)(char)((val >> 8) & 0xFF);
            //     ((float*)&raw_weight_vec)[i*4 + 2] = (float)(char)((val >> 16) & 0xFF);
            //     ((float*)&raw_weight_vec)[i*4 + 3] = (float)(char)((val >> 24) & 0xFF);
            // }
            // local_acc += in_val * raw_weight_vec;


            // float in_val = (float)input[k];
            // if (in_val == 0.0f) continue;
            // long weight_offset = (long)k * (long)FILTER_IFM_PITCH + (long)m_base;
            // float16 raw_weight_vec;
            // int4 packed_w = vload4(0, (__global int*)(weights + weight_offset));
            // #pragma unroll
            // for (int i = 0; i < 4; i++) {
            //     int val = ((int*)&packed_w)[i];
            //     ((float*)&raw_weight_vec)[i*4 + 0] = (float)(char)(val & 0xFF);
            //     ((float*)&raw_weight_vec)[i*4 + 1] = (float)(char)((val >> 8) & 0xFF);
            //     ((float*)&raw_weight_vec)[i*4 + 2] = (float)(char)((val >> 16) & 0xFF);
            //     ((float*)&raw_weight_vec)[i*4 + 3] = (float)(char)((val >> 24) & 0xFF);
            // }

            // local_acc += in_val * raw_weight_vec;
        }
    }
#endif

#if 0
    // --- Final Reduction and Output --- Original kernel code!
    float* acc_ptr = (float*)&local_acc;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float sum = sub_group_reduce_add(acc_ptr[i]);
        if (lid == 0 && (m_base + i) < M) {
            float final_res = sum;
#if BIAS_TERM
            final_res += (float)biases[m_base + i];
#endif
            output[m_base + i] = (OUTPUT_TYPE)final_res;
        }
    }
#else
    // Support HAS_FUSED_OPS
    float* acc_ptr = (float*)&local_acc;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float sum = sub_group_reduce_add(acc_ptr[i]);

        if (lid == 0 && (m_base + i) < M) {
            const uint oym = m_base + i; 
            // const uint ofm = 0;
            float dequantized = sum;
    #if HAS_FUSED_OPS
            FUSED_OPS;
            output[m_base + i] = (OUTPUT_TYPE)FUSED_OPS_RESULT;
    #else
            output[m_base + i] = (OUTPUT_TYPE)dequantized;
    #endif
        }
    }
#endif
}
