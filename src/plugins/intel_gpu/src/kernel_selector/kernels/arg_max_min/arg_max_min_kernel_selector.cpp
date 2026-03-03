// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_kernel_selector.h"
#include "arg_max_min_kernel_gpu_ref.h"
#include "arg_max_min_kernel_opt.h"
#include "arg_max_min_kernel_axis.h"
#include "arg_max_min_kernel_topk_reduction.h"
#include "arg_max_min_kernel_topk_bitonic.h"
#include "arg_max_min_kernel_topk_subgroup.h"
#include "arg_max_min_kernel_topk_radix.h"

namespace kernel_selector {

arg_max_min_kernel_selector::arg_max_min_kernel_selector() {
    Attach<ArgMaxMinKernelGPURef>();
    // Attach<ArgMaxMinKernelOpt>(); not yet implemented
    Attach<ArgMaxMinKernelAxis>();
    // Experimental TopK optimizations (activated via OV_GPU_TOPK_ALGO=A|B|C|D)
    Attach<ArgMaxMinKernelTopKReduction>();   // A: Two-pass reduction
    Attach<ArgMaxMinKernelTopKBitonic>();     // B: Bitonic sort
    Attach<ArgMaxMinKernelTopKSubgroup>();    // C: Subgroup optimized
    Attach<ArgMaxMinKernelTopKRadix>();       // D: Radix/histogram select
}

KernelsData arg_max_min_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::ARG_MAX_MIN);
}
}  // namespace kernel_selector
