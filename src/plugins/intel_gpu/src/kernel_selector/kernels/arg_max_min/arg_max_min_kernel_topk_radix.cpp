// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "arg_max_min_kernel_topk_radix.h"
#include <kernel_selector_utils.h>
#include <cstdlib>

namespace kernel_selector {

namespace {

static const char* ALGO_ENV = "OV_GPU_TOPK_ALGO";

size_t getOperationNumber(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::FEATURE: return params.outputs[0].Batch().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Z: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Y: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::X: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v;
        default: throw std::invalid_argument("Unsupported axis");
    }
}

std::string getOperationNumberString(const arg_max_min_params& params) {
    const auto& output = params.outputs[0];
    DimensionAccessHelperJit dims(output);
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return toVectorMulString({dims.x(), dims.y(), dims.z(), dims.f()});
        case ArgMaxMinAxis::FEATURE: return toVectorMulString({dims.x(), dims.y(), dims.z(), dims.b()});
        case ArgMaxMinAxis::Z: return toVectorMulString({dims.y(), dims.z(), dims.f(), dims.b()});
        case ArgMaxMinAxis::Y: return toVectorMulString({dims.x(), dims.z(), dims.f(), dims.b()});
        case ArgMaxMinAxis::X: return toVectorMulString({dims.y(), dims.z(), dims.f(), dims.b()});
        default: throw std::invalid_argument("Unsupported axis");
    }
}

size_t getSortSize(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.inputs[0].Batch().v;
        case ArgMaxMinAxis::FEATURE: return params.inputs[0].Feature().v;
        case ArgMaxMinAxis::Z: return params.inputs[0].Z().v;
        case ArgMaxMinAxis::Y: return params.inputs[0].Y().v;
        case ArgMaxMinAxis::X: return params.inputs[0].X().v;
        default: throw std::invalid_argument("Unsupported axis");
    }
}

}  // namespace

ParamsKey ArgMaxMinKernelTopKRadix::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::BATCH);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::X);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Y);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Z);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    return k;
}

bool ArgMaxMinKernelTopKRadix::Validate(const Params& p) const {
    if (!ArgMaxMinKernelBase::Validate(p))
        return false;

    const char* algo = std::getenv(ALGO_ENV);
    if (!algo || std::string(algo) != "D")
        return false;

    const auto& params = static_cast<const arg_max_min_params&>(p);

    // Only f16 input for radix approach (bit manipulation)
    if (params.inputs[0].GetDType() != Datatype::F16)
        return false;

    if (params.argMaxMinSortType != ArgMaxMinSortType::VALUE)
        return false;

    if (getSortSize(params) < 2)
        return false;

    return true;
}

ArgMaxMinKernelBase::DispatchData ArgMaxMinKernelTopKRadix::SetDefault(const arg_max_min_params& params) const {
    DispatchData dispatchData;

    const size_t WG_SIZE = 256;
    size_t ops_size = 1;
    if (!params.has_dynamic_tensors()) {
        ops_size = getOperationNumber(params);
    }

    dispatchData.gws = { ops_size * WG_SIZE, 1, 1 };
    dispatchData.lws = { WG_SIZE, 1, 1 };

    return dispatchData;
}

void ArgMaxMinKernelTopKRadix::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const arg_max_min_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        // Sortable keys buffer: VALUES_NUM * sizeof(uint) per operation
        const size_t sort_size = getSortSize(prim_params);
        const size_t ops_num = getOperationNumber(prim_params);
        kd.internalBuffers.clear();
        kd.internalBuffers.push_back(4 * sort_size * ops_num);
    };
}

JitConstants ArgMaxMinKernelTopKRadix::GetJitConstants(const arg_max_min_params& params) const {
    auto jit = ArgMaxMinKernelBase::GetJitConstants(params);

    const size_t WG_SIZE = 256;
    jit.AddConstant(MakeJitConstant("WG_SIZE", WG_SIZE));

    // PADDED_K: next power of 2 >= TOP_K (for bitonic sort)
    size_t padded_k = 1;
    while (padded_k < params.topK) padded_k <<= 1;
    jit.AddConstant(MakeJitConstant("PADDED_K", padded_k));

    if (params.has_dynamic_tensors()) {
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", getOperationNumberString(params)));
    } else {
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", getOperationNumber(params)));
    }

    if (params.argMaxMinSortType == ArgMaxMinSortType::VALUE)
        jit.AddConstant(MakeJitConstant("SORT_BY_VALUE", 1));

    if (params.values_first)
        jit.AddConstant(MakeJitConstant("TOP_K_ORDER", 1));

    return jit;
}

KernelsData ArgMaxMinKernelTopKRadix::GetKernelsData(const Params& params) const {
    if (!Validate(params))
        return {};

    const auto& orgParams = static_cast<const arg_max_min_params&>(params);
    auto dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<arg_max_min_params>(params);
    GetUpdateDispatchDataFunc(kd);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1,
                     GetFusedPrimitiveInputsCount(params), orgParams.outputs_num,
                     orgParams.is_shape_agnostic);

    // Add internal buffer for sortable keys (VALUES_NUM * sizeof(uint) per operation)
    const size_t sort_size = getSortSize(orgParams);
    const size_t ops_num = getOperationNumber(orgParams);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    kd.internalBuffers.push_back(4 * sort_size * ops_num);
    kd.internalBufferDataType = Datatype::UINT32;

    return {kd};
}

KernelsPriority ArgMaxMinKernelTopKRadix::GetKernelsPriority(const Params&) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
