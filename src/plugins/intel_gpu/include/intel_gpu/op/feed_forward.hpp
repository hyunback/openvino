// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

/// \brief Operator performing Swish Gated Linear Unit Activation
/// This operation performs gated linear unit activation that combines swish activation function
class FeedForward : public ov::op::Op {
public:
    OPENVINO_OP("FeedForward", "gpu_opset");

    FeedForward() = default;
    /// \brief Constructs a FeedForward operation.
    FeedForward(const Output<Node>& data,
           const Output<Node>& constant1,
           const Output<Node>& constant2,
           const Output<Node>& constant3,
           const Output<Node>& constant4,
           const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

private:
    ov::element::Type m_output_type;
};

std::vector<ov::PartialShape> shape_infer(const FeedForward* op, std::vector<ov::PartialShape> input_shapes);

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov