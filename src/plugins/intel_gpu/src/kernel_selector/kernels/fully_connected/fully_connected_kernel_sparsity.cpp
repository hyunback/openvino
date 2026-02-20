// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_types.h"
#include "fully_connected_kernel_sparsity.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey FullyConnected_sparsity::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    // k.EnableOutputDataType(Datatype::INT8);
    // k.EnableOutputDataType(Datatype::UINT8);
    // k.EnableInputWeightsType(WeightsType::UINT4);
    // k.EnableInputWeightsType(WeightsType::INT4);
    k.EnableInputWeightsType(WeightsType::UINT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    // k.EnableDynamicShapesSupport();
    k.EnableWeightsCompression();
    return k;
}

FullyConnected_sparsity::DispatchData FullyConnected_sparsity::SetDefault(const fully_connected_params& params,
                                                                          int, int /*kernel_number*/) const {
    auto dispatchData = Parent::SetDefault(params);

    size_t total_m = params.outputs[0].Feature().v * params.outputs[0].Y().v * params.outputs[0].X().v;
    size_t total_b = params.outputs[0].Batch().v;

    dispatchData.gws = { total_m, total_b, 1 };
    dispatchData.lws = { 16, 1, 1 };

    // GPU_DEBUG_INFO << "gws: " << dispatchData.gws[0] << "," << dispatchData.gws[1] << "," << dispatchData.gws[2] << std::endl;
    // GPU_DEBUG_INFO << "lws: " << dispatchData.lws[0] << "," << dispatchData.lws[1] << "," << dispatchData.lws[2] << std::endl;

    return dispatchData;
}

KernelsPriority FullyConnected_sparsity::GetKernelsPriority(const Params& params) const {
    const auto& fc_params = static_cast<const fully_connected_params&>(params);
    // skip llama-3.1-8b-instruct down_proj
    if (fc_params.inputs[0].Y().v < 2048 || fc_params.outputs[0].Y().v < 2048)
        return FORCE_PRIORITY_7;
    if (fc_params.inputs[0].Y().v >= 8192)
        return FORCE_PRIORITY_7;
    if (fc_params.inputs[0].Y().v > fc_params.outputs[0].Y().v)
        return FORCE_PRIORITY_6;    // IFM > OFM bad performance

    return FORCE_PRIORITY_1;
}

JitConstants FullyConnected_sparsity::GetJitConstants(const fully_connected_params& params,
    const FullyConnectedKernelBase::DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);
    Datatype accumulator_dt = GetAccumulatorType(params);
    Datatype activation_dt = GetActivationType(params);
    if (params.outputs[0].GetLayout() == DataLayout::bfyx)
        jit.AddConstant(MakeJitConstant("OUTPUT_3D", true));
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));
    jit.AddConstant(MakeJitConstant("FC_ID", params.layerID));

    auto wt = params.weights.GetDType();
    if (wt == WeightsType::UINT4 || wt == WeightsType::INT4) {
        jit.Merge(make_int4_packed_type_jit_constant("INT4_PACKED_TYPE", wt, 2));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order = { "0", "0", "0", "0" };
        if (params.outputs[0].Feature().v > 1) {
            idx_order[1] = "oym";
        } else if (params.outputs[0].Y().v > 1) {
            idx_order[2] = "oym";
        }

        FusedOpsConfiguration conf = { "", idx_order, "dequantized", activation_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }
    return jit;
}

KernelsData FullyConnected_sparsity::GetKernelsData(const Params& params) const {
    auto& fc_params = static_cast<const fully_connected_params&>(params);
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(
            params,
            fc_params.inputs[0].GetLayout(),
            WeightsLayout::ioyx,
            static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

bool FullyConnected_sparsity::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    // int8 validation
    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx && fc_params.outputs[0].X().v > 1)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    if (fc_params.outputs[0].Batch().v > 1 || fc_params.outputs[0].Feature().v > 1)
        DO_NOT_USE_THIS_KERNEL(params.layerID);

    return true;
}

}  // namespace kernel_selector
