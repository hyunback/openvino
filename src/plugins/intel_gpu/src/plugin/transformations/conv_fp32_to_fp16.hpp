// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class ConvFp32ToFp16 : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvFp32ToFp16", "0");

    ConvFp32ToFp16();
};

}   // namespace intel_gpu
}   // namespace ov
