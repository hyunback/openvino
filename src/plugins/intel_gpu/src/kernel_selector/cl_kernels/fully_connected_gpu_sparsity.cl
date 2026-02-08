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
#if 1
//     // Use float16 to maintain higher precision during accumulation
//     float16 local_acc = 0.0f;

//     // k loop: process 16 inputs per iteration for better memory coalescing
//     for (int k = 0; k < K; k += 16) {
//         int current_k = k + lid;
        
//         // Boundary check for K
//         if (current_k < K) {
//             float in_val = (float)input[current_k];
//             if (in_val == 0.0f) continue;

//             // Weight offset calculation using Pitch for padding awareness
//             long weight_offset = (long)current_k * (long)FILTER_IFM_PITCH + (long)m_base;
            
//             // Vector load and multiply-add with float precision
//             // convert_float16 ensures we do the math in FP32
//             local_acc += in_val * convert_float16(vload16(0, weights + weight_offset));
//         }
//     }

//     float* acc_ptr = (float*)&local_acc;
    
//     #pragma unroll
//     for (int i = 0; i < 16; ++i) {
//         // Parallel reduction within the sub-group
//         float sum = sub_group_reduce_add(acc_ptr[i]);
        
//         if (lid == 0 && (m_base + i) < M) {
//             // Add bias if applicable and apply fused operations
//             float final_res = sum;
// #if BIAS_TERM
//             final_res += (float)biases[m_base + i];
// #endif
//             // Fused ops usually expect float or the output type
//             // Apply them before final casting
//             output[m_base + i] = (OUTPUT_TYPE)final_res;
//         }
//     }


    float acc[16] = {0.0f};

    for (int k = 0; k < K; ++k) {
        float in_val = (float)input[k];
        if (in_val == 0.0f) continue;

        int m_idx = m_base + lid;
        if (m_idx < M) {
            long weight_offset = (long)k * (long)FILTER_IFM_PITCH + (long)m_idx;
            acc[0] += in_val * (float)weights[weight_offset];
        }
    }

    int m_idx = m_base + lid;
    if (m_idx < M) {
        float final_res = acc[0];
    #if BIAS_TERM
        final_res += (float)biases[m_idx];
    #endif
        output[m_idx] = (OUTPUT_TYPE)final_res;
    }

    // if (lid == 0) {
    //     long weight_offset = (long)0 * (long)FILTER_IFM_PITCH + (long)m_base;

    //     #pragma unroll
    //     for (int i = 0; i < 16; ++i) {
    //         if ((m_base + i) < M) {
    //             output[m_base + i] = (OUTPUT_TYPE)weights[weight_offset + i];
    //         }
    //     }
    // }
    // if (lid == 0) {
    //     for (int i = 0; i < 16; ++i) {
    //         int m = m_base + i;
    //         int k = 0;
            
    //         // weight_offset = (0 * M * 8) + (m * 8) + 0;
    //         long weight_offset = (long)m * 8; 

    //         output[m] = (OUTPUT_TYPE)weights[weight_offset];
    //     }
    // }

#else
    float local_acc[16] = {0.0f};

    for (int k = 0; k < K; ++k) {
        float in_val = (float)input[k];
        
        if (in_val == 0.0f) continue;

        long weight_offset = (long)k * (long)M + (long)m_base;

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            local_acc[i] += in_val * (float)weights[weight_offset + i];
        }
    }

    if (lid == 0) {
        for (int i = 0; i < 16; ++i) {
            if ((m_base + i) < M) {
                output[m_base + i] = (OUTPUT_TYPE)local_acc[i];
            }
        }
    }
#endif
}
