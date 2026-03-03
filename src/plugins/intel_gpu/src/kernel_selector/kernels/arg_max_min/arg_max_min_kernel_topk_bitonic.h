// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "arg_max_min_kernel_base.h"

namespace kernel_selector {

// Approach B: Bitonic Sort based TopK
class ArgMaxMinKernelTopKBitonic : public ArgMaxMinKernelBase {
public:
    ArgMaxMinKernelTopKBitonic() : ArgMaxMinKernelBase("arg_max_min_topk_bitonic") {}
    virtual ~ArgMaxMinKernelTopKBitonic() {}

    JitConstants GetJitConstants(const arg_max_min_params& params) const override;
    DispatchData SetDefault(const arg_max_min_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

private:
    bool Validate(const Params&) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
