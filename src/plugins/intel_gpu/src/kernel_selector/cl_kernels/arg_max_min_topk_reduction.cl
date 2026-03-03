// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Approach A: Two-Pass Reduction TopK
// Algorithm:
//   - One workgroup (WG_SIZE work-items) cooperates per output slice
//   - Data is processed in streaming chunks loaded into SLM
//   - For each chunk: cooperative parallel max-reduction to find best element
//   - Repeat K times, marking found elements as "used"
//   - Result is sorted by value (found in descending/ascending order)
//

#include "include/fetch_utils.cl"

#ifdef BATCH_AXIS
    #define VALUES_NUM INPUT0_BATCH_NUM
    #define AXIS 0
#endif
#ifdef FEATURE_AXIS
    #define VALUES_NUM INPUT0_FEATURE_NUM
    #define AXIS 1
#endif
#ifdef Z_AXIS
    #define VALUES_NUM INPUT0_SIZE_Z
    #define AXIS 2
#endif
#ifdef Y_AXIS
    #define VALUES_NUM INPUT0_SIZE_Y
    #define AXIS 3
#endif
#ifdef X_AXIS
    #define VALUES_NUM INPUT0_SIZE_X
    #define AXIS 4
#endif

#ifdef MAX_OUT
    #define COMPARE_SIGN >
    #define INPUT0_FILL_VAL INPUT0_VAL_MIN
#else
    #define COMPARE_SIGN <
    #define INPUT0_FILL_VAL INPUT0_VAL_MAX
#endif

#ifndef WG_SIZE
    #define WG_SIZE 256
#endif

// Each WI processes this many elements
#define ELEMENTS_PER_WI ((VALUES_NUM + WG_SIZE - 1) / WG_SIZE)

inline void FUNC(get_indices_from_dims)(OPTIONAL_SHAPE_INFO_ARG
                                        const uint output_idx,
                                        uint* indices)
{
#ifdef BATCH_AXIS
    const uint out_first_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_SIZE_Z;
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;
    indices[1] = out_first_dim; indices[2] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef FEATURE_AXIS
    const uint out_first_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_SIZE_Z;
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;
    indices[0] = out_first_dim; indices[2] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef Z_AXIS
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Y * INPUT0_SIZE_X);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_X) % INPUT0_FEATURE_NUM;
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Y;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[3] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef Y_AXIS
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_X);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_X) % INPUT0_FEATURE_NUM;
    const uint out_third_dim = output_idx / INPUT0_SIZE_X % INPUT0_SIZE_Z;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_X;
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[2] = out_third_dim; indices[4] = out_fourth_dim;
#endif
#ifdef X_AXIS
    const uint out_first_dim = output_idx / (INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_Y);
    const uint out_second_dim = output_idx / (INPUT0_SIZE_Z * INPUT0_SIZE_Y) % INPUT0_FEATURE_NUM;
    const uint out_third_dim = output_idx / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint out_fourth_dim = output_idx % INPUT0_SIZE_Y;
    indices[0] = out_first_dim; indices[1] = out_second_dim; indices[2] = out_third_dim; indices[3] = out_fourth_dim;
#endif
}

REQD_SUB_GROUP_SIZE(16)
KERNEL(arg_max_min_topk_reduction)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input
    ,__global OUTPUT_TYPE* output
#ifdef OUTPUT1_TYPE
    ,__global OUTPUT1_TYPE* second_output
#endif
    ,__global INPUT0_TYPE* values_buf     // VALUES_NUM elements for cached input values
    ,__global int* indices_buf            // VALUES_NUM elements for tracking found indices
)
{
    const uint lid = (uint)get_local_id(0);
    const uint output_idx = (uint)get_group_id(0);

    if (OPERATION_NUM > 1 && output_idx >= OPERATION_NUM)
        return;

    uint indices[] = { 0, 0, 0, 0, 0 };
    if (OPERATION_NUM > 1) {
        FUNC_CALL(get_indices_from_dims)(OPTIONAL_SHAPE_INFO_TENSOR output_idx, indices);
    }

    // SLM for reduction
    __local INPUT0_TYPE slm_values[WG_SIZE];
    __local uint slm_indices[WG_SIZE];

    // Phase 1: Load all input values into global buffer (coalesced access)
    for (uint i = lid; i < VALUES_NUM; i += WG_SIZE) {
        indices[AXIS] = i;
        values_buf[output_idx * VALUES_NUM + i] = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])];
        indices_buf[output_idx * VALUES_NUM + i] = 0; // not found yet
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    __global INPUT0_TYPE* my_values = values_buf + output_idx * VALUES_NUM;
    __global int* my_found = indices_buf + output_idx * VALUES_NUM;

    // Phase 2: Iteratively find top-K elements using parallel reduction
    for (uint k = 0; k < TOP_K; ++k) {
        // Each WI finds its local best from its portion
        INPUT0_TYPE local_best_val = INPUT0_FILL_VAL;
        uint local_best_idx = 0;

        for (uint i = lid; i < VALUES_NUM; i += WG_SIZE) {
            if (my_found[i] == 0) {
                INPUT0_TYPE val = my_values[i];
                if (val COMPARE_SIGN local_best_val) {
                    local_best_val = val;
                    local_best_idx = i;
                }
            }
        }

        // Store local best to SLM
        slm_values[lid] = local_best_val;
        slm_indices[lid] = local_best_idx;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Tree reduction in SLM to find global best
        for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                if (slm_values[lid + stride] COMPARE_SIGN slm_values[lid]) {
                    slm_values[lid] = slm_values[lid + stride];
                    slm_indices[lid] = slm_indices[lid + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        // WI 0 has the global best
        if (lid == 0) {
            uint found_idx = slm_indices[0];
            INPUT0_TYPE found_val = slm_values[0];
            my_found[found_idx] = 1; // mark as used

            // Write output: k-th best value/index
            // For SORT_BY_VALUE, results are already in sorted order (best first)
            indices[AXIS] = k;
#ifdef TOP_K_ORDER
            output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT_TYPE(found_val);
#else
            output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT_TYPE(found_idx);
#endif
#ifdef OUTPUT1_TYPE
    #ifdef TOP_K_ORDER
            second_output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT1_TYPE(found_idx);
    #else
            second_output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[0], indices[1], 0, indices[2], indices[3], indices[4])] = TO_OUTPUT1_TYPE(found_val);
    #endif
#endif
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}

#undef COMPARE_SIGN
#undef INPUT0_FILL_VAL
#undef AXIS
#undef VALUES_NUM
#undef ELEMENTS_PER_WI
#undef WG_SIZE
