// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/feed_forward.hpp"
#include "openvino/core/validation_util.hpp"
// #include "variadic_split_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
// #include "openvino/op/variadic_split.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

FeedForward::FeedForward(const Output<Node>& data,
                const Output<Node>& c1,
                const Output<Node>& c2,
                const Output<Node>& c3,
                const Output<Node>& c4,
                const ov::element::Type output_type)
    : Op({data, c1, c2, c3, c4}), m_output_type(output_type) {
    validate_and_infer_types();
}

bool FeedForward::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void FeedForward::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;

    // std::vector<ov::PartialShape> input_shapes = {
    //     get_input_partial_shape(0),
    //     ov::PartialShape(ov::Shape{}),
    //     ov::PartialShape(ov::Shape{2})
    // };
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    auto out_shapes = shape_infer(this, input_shapes);
    set_output_type(0, output_type, shape_infer(this, input_shapes)[0]);
}

std::shared_ptr<Node> FeedForward::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<FeedForward>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    new_args.at(4),
                                    m_output_type);
}

std::vector<ov::PartialShape> shape_infer(const FeedForward* op, std::vector<ov::PartialShape> input_shapes) {
    // ov::op::v1::VariadicSplit variadic_split;
    // std::vector<int64_t> axis = { op->get_axis() };
    // std::vector<int64_t> split_lengths = { op->get_split_lengths(), -1 };

    // std::unordered_map<size_t, ov::Tensor> const_data;
    // const_data.emplace(1, ov::Tensor(ov::element::i64, ov::Shape{}, static_cast<void*>(axis.data())));
    // const_data.emplace(2, ov::Tensor(ov::element::i64, ov::Shape{split_lengths.size()}, static_cast<void*>(split_lengths.data())));

    // return ov::op::v1::shape_infer(&variadic_split, input_shapes, ov::make_tensor_accessor(const_data));
    // return  shape_infer(&variadic_split, input_shapes, ov::make_tensor_accessor(const_data));
    return input_shapes;
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov