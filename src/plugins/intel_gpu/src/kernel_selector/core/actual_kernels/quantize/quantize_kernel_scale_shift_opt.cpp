// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "quantize_kernel_scale_shift_opt.h"
#include "kernel_selector_utils.h"
#include <string>

static const size_t sub_group_size = 32;
static const size_t feature_size = 32;

namespace kernel_selector {
ParamsKey QuantizeKernelScaleShift::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableQuantizeScaleShiftOpt();
    return k;
}

std::vector<size_t> static GetOptimalLocalWorkGroupSizes_(std::vector<size_t> gws, const EngineInfo& info, std::vector<size_t> order) {
    const size_t lws_max = info.maxWorkGroupSize;
    const size_t optimal_lws_values[] = { 1024, 960, 896, 832, 768, 704, 640, 576,
                                          512, 480, 448, 416, 384, 352, 320, 288,
                                          256, 227, 224, 192, 160, 128, 96, 64, 32, 16, 8, 7, 6, 5, 4, 2, 1 };
    size_t total_lws = 1;
    std::vector<size_t> lws(gws.size());

    if (!order.empty() && gws.size() != order.size()) {
        throw std::runtime_error("order size is different from gws size\n");
    }

    for (size_t i = 0; i < gws.size(); ++i) {
        size_t order_idx = order.empty() ? i : order[i];
        auto rest_lws = lws_max / total_lws;
        size_t lws_idx = 0;
        while (rest_lws < optimal_lws_values[lws_idx]) lws_idx++;

        while (gws[order_idx] % optimal_lws_values[lws_idx]) lws_idx++;

        lws[order_idx] = optimal_lws_values[lws_idx];
        total_lws *= optimal_lws_values[lws_idx];
    }

    return lws;
}

CommonDispatchData QuantizeKernelScaleShift::SetDefault(const quantize_params& params, const optional_params&) const {
    CommonDispatchData dispatchData;

    auto output = params.outputs[0];

    if (output.GetLayout() == DataLayout::b_fs_yx_fsv16 || output.GetLayout() == DataLayout::b_fs_zyx_fsv32) {
        dispatchData.gws[0] = output.Z().v *output.Y().v * output.X().v;
        dispatchData.gws[1] = Align(output.Feature().v, sub_group_size);
        dispatchData.gws[2] = output.Batch().v;

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = sub_group_size;
        dispatchData.lws[2] = 1;
    } else if (output.GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 || output.GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 ||
               output.GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16) {
        dispatchData.gws[0] = output.Y().v * output.X().v;
        dispatchData.gws[1] = Align(output.Feature().v, feature_size);
        dispatchData.gws[2] = Align(output.Batch().v, feature_size);

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = feature_size;
        dispatchData.lws[2] = params.engineInfo.maxWorkGroupSize / feature_size;
    } else if (output.GetLayout() == DataLayout::bs_fs_zyx_bsv32_fsv32) {
        dispatchData.gws[0] = output.Z().v * output.Y().v * output.X().v;
        dispatchData.gws[1] = Align(output.Feature().v, feature_size);
        dispatchData.gws[2] = Align(output.Batch().v, feature_size);

        dispatchData.lws[0] = 1;
        dispatchData.lws[1] = feature_size;
        dispatchData.lws[2] = params.engineInfo.maxWorkGroupSize / feature_size;
    } else {
        dispatchData.gws = GetTensorFriendlyWorkGroups(output);
#if 0
        // dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
#else
        // // test
        // auto out_layout = params.outputs[0].GetLayout();
        // dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, out_layout, out_layout);

        // my test
        // assume gws[xyz, f, b]
        auto output_layout = params.outputs[0].GetLayout();
        if (Tensor::SimpleLayout(output_layout)) {
            dispatchData.lws = GetOptimalLocalWorkGroupSizes_(dispatchData.gws, params.engineInfo, {0, 1, 2});
        } else {
            auto blocked_bsv_fsv_layout = output_layout == DataLayout::bs_fs_yx_bsv16_fsv16 || output_layout == DataLayout::bs_fs_yx_bsv16_fsv4 ||
                                        output_layout == DataLayout::bs_fs_zyx_bsv16_fsv16 || output_layout == DataLayout::bs_fs_zyx_bsv16_fsv32 ||
                                        output_layout == DataLayout::bs_fs_zyx_bsv32_fsv16 || output_layout == DataLayout::bs_fs_zyx_bsv32_fsv32;
            if (blocked_bsv_fsv_layout) {
                dispatchData.lws = GetOptimalLocalWorkGroupSizes_(dispatchData.gws, params.engineInfo, {2, 1, 0});
            } else {
                dispatchData.lws = GetOptimalLocalWorkGroupSizes_(dispatchData.gws, params.engineInfo, {1, 0, 2});
            }
        }
#endif
        for (auto tt : dispatchData.lws) {
            std::cout << tt << " ";
        }
        std::cout << std::endl;
    }

    return dispatchData;
}

JitConstants QuantizeKernelScaleShift::GetJitConstants(const quantize_params& params, const CommonDispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    if (params.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 || params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv32 ||
        params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv16_fsv16 || params.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16) {
        jit.AddConstant(MakeJitConstant("FEATURE_BLOCKED_FORMAT", true));
        jit.AddConstant(MakeJitConstant("GWS_BATCH", 2));
        jit.AddConstant(MakeJitConstant("GWS_FEATURE", 1));
        jit.AddConstant(MakeJitConstant("GWS_YX", 0));
        jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    } else {
        auto tensor_jits = GetTensorFriendlyWorkGroupsJit(params.outputs[0]);
        jit.Merge(tensor_jits);
    }

    auto can_use_output_range = params.per_tensor_output_range && params.out_lo < params.out_hi;
    auto has_output_range_round = !(params.outputs[0].GetDType() == Datatype::INT8 || params.outputs[0].GetDType() == Datatype::UINT8);

    jit.AddConstant(MakeJitConstant("HAS_POST_SCALE", params.has_post_scale));
    jit.AddConstant(MakeJitConstant("HAS_POST_SHIFT", params.has_post_shift));
    jit.AddConstant(MakeJitConstant("HAS_PRE_SHIFT", params.has_pre_shift));
    jit.AddConstant(MakeJitConstant("HAS_CLAMP", params.has_clamp));
    jit.AddConstant(MakeJitConstant("HAS_MIN_CLAMP", params.has_min_clamp));
    jit.AddConstant(MakeJitConstant("HAS_MAX_CLAMP", params.has_max_clamp));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_RANGE", params.per_tensor_input_range));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_OUTPUT_RANGE", params.per_tensor_output_range));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_SCALE", params.per_tensor_input_scale));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_INPUT_SHIFT", params.per_tensor_input_shift));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_OUTPUT_SCALE", params.per_tensor_output_scale));
    jit.AddConstant(MakeJitConstant("PER_TENSOR_OUTPUT_SHIFT", params.per_tensor_output_shift));
    jit.AddConstant(MakeJitConstant("IN_LO_VAL", params.in_lo));
    jit.AddConstant(MakeJitConstant("IN_HI_VAL", params.in_hi));
    jit.AddConstant(MakeJitConstant("OUT_LO_VAL", params.out_lo));
    jit.AddConstant(MakeJitConstant("OUT_HI_VAL", params.out_hi));
    jit.AddConstant(MakeJitConstant("IN_SCALE_VAL", params.in_scale));
    jit.AddConstant(MakeJitConstant("IN_SHIFT_VAL", params.in_shift));
    jit.AddConstant(MakeJitConstant("OUT_SCALE_VAL", params.out_scale));
    jit.AddConstant(MakeJitConstant("OUT_SHIFT_VAL", params.out_shift));
    jit.AddConstant(MakeJitConstant("CAN_USE_OUTPUT_RANGE", can_use_output_range));
    jit.AddConstant(MakeJitConstant("HAS_OUTPUT_RANGE_ROUND", has_output_range_round));

    return jit;
}

bool QuantizeKernelScaleShift::Validate(const Params& p, const optional_params&) const {
    const quantize_params& params = static_cast<const quantize_params&>(p);
    if (params.inputs.size() != 9)
        return false;

    return true;
}

KernelsPriority QuantizeKernelScaleShift::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
