// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "ov_ops/feed_forward.hpp"
// #include "glu_shape_inference.hpp"
#include "feed_forward_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(feed_forward);

layout feed_forward_inst::calc_output_layout(feed_forward_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<feed_forward>();
    auto input_layout = impl_param.get_input_layout();
    auto output_type = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);
    auto output_format = input_layout.format;

    return layout(output_type, output_format, desc->output_size);
}

template<typename ShapeType>
std::vector<layout> feed_forward_inst::calc_output_layouts(feed_forward_node const& /*node*/, const kernel_impl_params& impl_param) {
    // auto desc = impl_param.typed_desc<feed_forward>();
    // auto input_layout = impl_param.get_input_layout();
    // auto output_type = impl_param.desc->output_data_types[0].value_or(input_layout.data_type);
    // auto output_format = input_layout.format;

    // // ov::op::internal::GLU op;
    // // op.set_axis(desc->axis);
    // // op.set_split_lengths(desc->split_lengths);

    // std::vector<ShapeType> input_shapes = {impl_param.get_input_layout(0).get<ShapeType>()};

    // std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    // return { layout(output_shapes[0], output_type, output_format) };
    return forward_input0_shape<ShapeType>(impl_param);
}

template std::vector<layout> feed_forward_inst::calc_output_layouts<ov::PartialShape>(feed_forward_node const& node,
                                                                                const kernel_impl_params& impl_param);

std::string feed_forward_inst::to_string(feed_forward_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite feed_forward_info;
    feed_forward_info.add("input_id", input.id());

    node_info->add("feed_forward_info", feed_forward_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

feed_forward_inst::typed_primitive_inst(network& network, feed_forward_node const& node) : parent(network, node) {}

}  // namespace cldnn
