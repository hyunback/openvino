// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_fp32_to_fp16.hpp"

#include "openvino/core/rt_info.hpp"
#include "convert_convolution.hpp"

#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert_like.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

ConvFp32ToFp16::ConvFp32ToFp16() {
    using namespace ov::op;
    using namespace ov::pass::pattern;
    using namespace ov::pass::pattern::op;


    // auto is_target_pattern = [](const Output<Node>& output) {
    //     const auto& conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(output.get_node_shared_ptr());
    //     if (conv) {
    //         std::cout << conv.get()->get_name() << "!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    //         return true;
    //     }
    //     // return false;
    //     return true;
    // };

    auto input_m = any_input();
    // auto weights_m = any_input(has_static_dim(0));
    auto weights_m = any_input();
    // auto bias_val_m = wrap_type<ov::op::v0::Constant>();
    auto bias_val_m = any_input();
    // auto convolution_m = wrap_type<ov::op::v1::Convolution>({ input_m, weights_m }, all_of({type_matches(ov::element::f32), consumers_count(1)}));
    auto convolution_m = wrap_type<op::Convolution>({ input_m, weights_m, bias_val_m }, all_of({type_matches(ov::element::f32), consumers_count(1)}));
    // auto convolution_m = wrap_type<op::Convolution>(is_target_pattern);
    // std::cout << "ConvFp32ToFp16 start!!!" << std::endl;

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto conv_node = std::dynamic_pointer_cast<op::Convolution>(pattern_map.at(convolution_m).get_node_shared_ptr());

        // for (size_t i = 0; i < conv_node->get_input_size(); ++i) {
        //     const auto& input = conv_node->get_input_node_shared_ptr(i);
        // }

        // conv_node
        // fp32 -> div -> reorder(fp16)  -> conv(fp32) -> mul ->

        // ov::NodeVector subgraph_nodes;
        auto div_const = ov::op::v0::Constant::create(element::f32, Shape{1}, {1000.0});
        auto mul_const = ov::op::v0::Constant::create(element::f32, Shape{1}, {1000.0});
        auto div = std::make_shared<ov::op::v1::Divide>(conv_node->get_input_node_shared_ptr(0), div_const);
        auto convert0 = std::make_shared<ov::op::v0::Convert>(div, element::f16);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(conv_node->get_input_node_shared_ptr(1), element::f16);
        auto new_conv = std::make_shared<op::Convolution>(convert0,
                                                          convert1,
                                                          std::make_shared<op::Placeholder>(),
                                                          conv_node->get_strides(),
                                                          conv_node->get_pads_begin(),
                                                          conv_node->get_pads_end(),
                                                          conv_node->get_dilations(),
                                                          conv_node->get_groups(),
                                                          conv_node->get_auto_pad(),
                                                          element::f32);
        auto mul = std::make_shared<ov::op::v1::Multiply>(new_conv, mul_const);

        ov::replace_node(m.get_match_root(), mul);

        return true;
    };


    auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "ConvFp32ToFp16");
    this->register_matcher(m, callback);
    // std::cout << "this->register_matcher(m, callback);;;" << std::endl;
}

}  // namespace intel_gpu
}  // namespace ov
