// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Approach B: Bitonic Sort based TopK
// Algorithm:
//   - One workgroup loads all data into SLM-backed global buffer
//   - Performs bitonic sort on the full array cooperatively
//   - Takes top-K elements from sorted result
//   - For large data sizes, uses chunk-based approach:
//     split into blocks, bitonic sort each block, merge top-K
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

// Block size for bitonic sort (must be power of 2, fits in SLM)
// SLM usage: BLOCK_SIZE * (sizeof(INPUT0_TYPE) + sizeof(uint))
// For f16: BLOCK_SIZE * 6 bytes. With 64KB SLM, max ~10000 elements
// Use 4096 as a good balance
#ifndef BLOCK_SIZE
    #define BLOCK_SIZE 4096
#endif

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

// Swap two iav_type values
inline void FUNC(compare_swap)(__local INPUT0_TYPE* vals, __local uint* idxs,
                                uint i, uint j, bool dir) {
    // dir=true means ascending for MAX_OUT (we want descending overall, so largest first)
    bool should_swap;
#ifdef MAX_OUT
    should_swap = dir ? (vals[i] < vals[j]) : (vals[i] > vals[j]);
#else
    should_swap = dir ? (vals[i] > vals[j]) : (vals[i] < vals[j]);
#endif
    if (should_swap) {
        INPUT0_TYPE tmp_v = vals[i];
        vals[i] = vals[j];
        vals[j] = tmp_v;
        uint tmp_i = idxs[i];
        idxs[i] = idxs[j];
        idxs[j] = tmp_i;
    }
}

KERNEL(arg_max_min_topk_bitonic)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input
    ,__global OUTPUT_TYPE* output
#ifdef OUTPUT1_TYPE
    ,__global OUTPUT1_TYPE* second_output
#endif
    ,__global INPUT0_TYPE* sort_values_buf   // padded to next power-of-2 of VALUES_NUM
    ,__global uint* sort_indices_buf         // same size
)
{
    const uint lid = (uint)get_local_id(0);
    const uint output_idx = (uint)get_group_id(0);

    if (OPERATION_NUM > 1 && output_idx >= OPERATION_NUM)
        return;

    uint base_indices[] = { 0, 0, 0, 0, 0 };
    if (OPERATION_NUM > 1) {
        FUNC_CALL(get_indices_from_dims)(OPTIONAL_SHAPE_INFO_TENSOR output_idx, base_indices);
    }

    // Round up VALUES_NUM to next power of 2 for bitonic sort
    // This is computed as a constant by the JIT: SORT_ARRAY_SIZE
    __global INPUT0_TYPE* my_vals = sort_values_buf + output_idx * SORT_ARRAY_SIZE;
    __global uint* my_idxs = sort_indices_buf + output_idx * SORT_ARRAY_SIZE;

    // SLM for local bitonic sort block
    __local INPUT0_TYPE slm_vals[BLOCK_SIZE];
    __local uint slm_idxs[BLOCK_SIZE];

    // Phase 1: Load data into global buffer, pad with fill values
    for (uint i = lid; i < SORT_ARRAY_SIZE; i += WG_SIZE) {
        if (i < VALUES_NUM) {
            base_indices[AXIS] = i;
            my_vals[i] = input[FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR base_indices[0], base_indices[1], 0, base_indices[2], base_indices[3], base_indices[4])];
            my_idxs[i] = i;
        } else {
            my_vals[i] = INPUT0_FILL_VAL;
            my_idxs[i] = i;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Phase 2: Block-level bitonic sort in SLM
    uint num_blocks = (SORT_ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint elems_per_wi = (BLOCK_SIZE + WG_SIZE - 1) / WG_SIZE;

    for (uint blk = 0; blk < num_blocks; blk++) {
        uint blk_offset = blk * BLOCK_SIZE;
        uint blk_size = min((uint)BLOCK_SIZE, (uint)(SORT_ARRAY_SIZE - blk_offset));

        // Load block into SLM
        for (uint i = lid; i < BLOCK_SIZE; i += WG_SIZE) {
            if (i < blk_size) {
                slm_vals[i] = my_vals[blk_offset + i];
                slm_idxs[i] = my_idxs[blk_offset + i];
            } else {
                slm_vals[i] = INPUT0_FILL_VAL;
                slm_idxs[i] = blk_offset + i;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Bitonic sort in SLM
        for (uint k_stage = 2; k_stage <= BLOCK_SIZE; k_stage <<= 1) {
            for (uint j_step = k_stage >> 1; j_step > 0; j_step >>= 1) {
                for (uint i = lid; i < BLOCK_SIZE / 2; i += WG_SIZE) {
                    uint idx = i;
                    uint block_pos = idx & (j_step - 1);
                    uint pair_base = (idx / j_step) * (j_step * 2);
                    uint left = pair_base + block_pos;
                    uint right = left + j_step;

                    if (right < BLOCK_SIZE) {
                        bool ascending = ((left / k_stage) & 1) == 0;
                        FUNC_CALL(compare_swap)(slm_vals, slm_idxs, left, right, ascending);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        // Write sorted block back to global buffer
        for (uint i = lid; i < blk_size; i += WG_SIZE) {
            my_vals[blk_offset + i] = slm_vals[i];
            my_idxs[blk_offset + i] = slm_idxs[i];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Phase 3: Merge sorted blocks - take top-K using simple comparison
    // Each block is sorted. We do a K-way merge using iterative selection.
    // For small num_blocks (e.g., 6-8), this is efficient.
    // We use SLM to store block pointers (cursors).

    __local uint block_cursors[32]; // max 32 blocks
    __local INPUT0_TYPE best_val[1];
    __local uint best_idx[1];
    __local uint best_block[1];

    if (lid < num_blocks) {
        block_cursors[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // SLM for reduction
    __local INPUT0_TYPE reduce_vals[WG_SIZE];
    __local uint reduce_idxs[WG_SIZE];
    __local uint reduce_blocks[WG_SIZE];

    for (uint k = 0; k < TOP_K; ++k) {
        // Each WI checks one or more blocks to find the current best
        INPUT0_TYPE my_best_val = INPUT0_FILL_VAL;
        uint my_best_idx = 0;
        uint my_best_blk = 0;

        for (uint blk = lid; blk < num_blocks; blk += WG_SIZE) {
            uint cursor = block_cursors[blk];
            if (cursor < min((uint)BLOCK_SIZE, (uint)(SORT_ARRAY_SIZE - blk * BLOCK_SIZE))) {
                INPUT0_TYPE v = my_vals[blk * BLOCK_SIZE + cursor];
                if (v COMPARE_SIGN my_best_val) {
                    my_best_val = v;
                    my_best_idx = my_idxs[blk * BLOCK_SIZE + cursor];
                    my_best_blk = blk;
                }
            }
        }

        // Reduce across work-items to find the globally best
        reduce_vals[lid] = my_best_val;
        reduce_idxs[lid] = my_best_idx;
        reduce_blocks[lid] = my_best_blk;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
            if (lid < stride) {
                if (reduce_vals[lid + stride] COMPARE_SIGN reduce_vals[lid]) {
                    reduce_vals[lid] = reduce_vals[lid + stride];
                    reduce_idxs[lid] = reduce_idxs[lid + stride];
                    reduce_blocks[lid] = reduce_blocks[lid + stride];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            // Advance cursor for the winning block
            block_cursors[reduce_blocks[0]]++;

            // Write k-th result
            base_indices[AXIS] = k;
#ifdef TOP_K_ORDER
            output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR base_indices[0], base_indices[1], 0, base_indices[2], base_indices[3], base_indices[4])] = TO_OUTPUT_TYPE(reduce_vals[0]);
#else
            output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR base_indices[0], base_indices[1], 0, base_indices[2], base_indices[3], base_indices[4])] = TO_OUTPUT_TYPE(reduce_idxs[0]);
#endif
#ifdef OUTPUT1_TYPE
    #ifdef TOP_K_ORDER
            second_output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR base_indices[0], base_indices[1], 0, base_indices[2], base_indices[3], base_indices[4])] = TO_OUTPUT1_TYPE(reduce_idxs[0]);
    #else
            second_output[FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR base_indices[0], base_indices[1], 0, base_indices[2], base_indices[3], base_indices[4])] = TO_OUTPUT1_TYPE(reduce_vals[0]);
    #endif
#endif
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

#undef COMPARE_SIGN
#undef INPUT0_FILL_VAL
#undef AXIS
#undef VALUES_NUM
#undef WG_SIZE
#undef BLOCK_SIZE
