// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "feed_forward_fusion.hpp"

// #include "intel_gpu/op/swiglu.hpp"
#include "intel_gpu/op/feed_forward.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

FeedForwardFusion::FeedForwardFusion() {
    using namespace ov::pass::pattern;

    // auto last_dim_static = [](const ov::Output<ov::Node>& output) {
    //     auto out_ps = output.get_node()->get_output_partial_shape(0);
    //     return out_ps.rank().is_static() && out_ps[out_ps.rank().get_length() - 1].is_static() && out_ps.size() <= 5;
    // };

    // // Detect SwiGLU decomposition pattern
    // // SwiGLU(Xw, Xv, beta) = (Xw * (1.0 + exp(-beta * Xw))) * Xv
    // auto data_m = any_input(last_dim_static);

    // // VariadicSplit(X, axis, split_lengths) = Xw, Xv
    // auto axis_const_m = wrap_type<ov::op::v0::Constant>();
    // auto split_lengths_const_m = wrap_type<ov::op::v0::Constant>();
    // auto variadic_split_m = wrap_type<ov::op::v1::VariadicSplit>({data_m, axis_const_m, split_lengths_const_m});
    // variadic_split_m->set_output_size(2);

    // // Swish(Xw) = Xw * (1.0 + exp(-beta * Xw))
    // auto swish_m = wrap_type<ov::op::v4::Swish>({variadic_split_m->output(0)});

    // // Mul(Xw, Xv) = Swish(Xw) * Xv
    // auto mul_m = wrap_type<ov::op::v1::Multiply>({swish_m, variadic_split_m->output(1)});

    // auto data = any_input();
    // auto weights = any_input();
    // auto matmul1 = wrap_type<ov::op::v0::MatMul>({ data, weights }, ov::pass::pattern::has_static_rank());
    // auto fully_connected = wrap_type<op::FullyConnected>({data, weights, bias}, consumers_count(1));
    // auto bias = any_input();

    auto matmul1 = any_input();
    auto bios = any_input();
    auto add1 = wrap_type<ov::op::v1::Add>({matmul1, bios}, consumers_count(5));
    auto mul1 = wrap_type<ov::op::v1::Multiply>({add1, add1});
    auto mul2 = wrap_type<ov::op::v1::Multiply>({add1, mul1});
    auto constant1 = any_input();
    auto mul3 = wrap_type<ov::op::v1::Multiply>({mul2, constant1});
    auto add2 = wrap_type<ov::op::v1::Add>({add1, mul3});
    auto constant2 = any_input();
    auto mul4 = wrap_type<ov::op::v1::Multiply>({add2, constant2});
    auto tanh = wrap_type<ov::op::v0::Tanh>({mul4});
    auto constant3 = any_input();
    auto add3 = wrap_type<ov::op::v1::Add>({tanh, constant3});
    auto mul5 = wrap_type<ov::op::v1::Multiply>({add3, add1});
    auto constant4 = any_input();
    auto mul6 = wrap_type<ov::op::v1::Multiply>({mul5, constant4});
    // auto constant5 = any_input();
    // auto matmul2 = wrap_type<ov::op::v0::MatMul>({ mul6, constant5 }, ov::pass::pattern::has_static_rank());

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        // std::cout << "feed_forward_fusion callback !!!" << std::endl;

        const auto& pattern_map = m.get_pattern_value_map();
        // const auto& m_matmul1 = pattern_map.at(matmul1).get_node_shared_ptr();
        const auto& m_add1 = pattern_map.at(add1).get_node_shared_ptr();

        const auto& m_constant1 = pattern_map.at(constant1).get_node_shared_ptr();
        const auto& m_constant2 = pattern_map.at(constant2).get_node_shared_ptr();
        const auto& m_constant3 = pattern_map.at(constant3).get_node_shared_ptr();
        const auto& m_constant4 = pattern_map.at(constant4).get_node_shared_ptr();

        // auto output_type = m.get_match_root()->get_output_element_type(0);
        // std::cout << "match_root(): " << m.get_match_root()->get_friendly_name() << std::endl;

        auto ff = std::make_shared<op::FeedForward>(m_add1, m_constant1, m_constant2, m_constant3, m_constant4);
        ff->set_friendly_name(m.get_match_root()->get_friendly_name() + "_FF");
        ov::copy_runtime_info(m.get_matched_nodes(), ff);
        ov::replace_node(m.get_match_root(), ff);

        // std::cout << m_add1->get_friendly_name() << std::endl;

        // (Tanh(b*(a*x*x*x+x))+c)x*d

        // OPENVINO_ASSERT(pattern_map.count(mul_m));
        // OPENVINO_ASSERT(pattern_map.count(swish_m));
        // OPENVINO_ASSERT(pattern_map.count(variadic_split_m));
        // OPENVINO_ASSERT(pattern_map.count(split_lengths_const_m));
        // OPENVINO_ASSERT(pattern_map.count(axis_const_m));
        // auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(pattern_map.at(mul_m).get_node_shared_ptr());
        // if (!mul || transformation_callback(mul))
        //     return false;
        // if (mul->input_value(1).get_index() != 1)
        //     return false;

        // auto variadic_split = std::dynamic_pointer_cast<ov::op::v1::VariadicSplit>(pattern_map.at(variadic_split_m).get_node_shared_ptr());
        // auto variadic_split_in_ps = variadic_split->get_input_partial_shape(0);
        // auto last_dim = variadic_split_in_ps.rank().get_length() - 1;

        // auto axis = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(axis_const_m).get_node_shared_ptr());
        // bool valid_axis_const_values = ov::op::util::has_constant_value<int64_t>(axis, -1) ||
        //                                ov::op::util::has_constant_value<int64_t>(axis, last_dim);
        // if (!valid_axis_const_values)
        //     return false;
        // auto axis_value = axis->cast_vector<int64_t>()[0];

        // auto split_lengths = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(split_lengths_const_m).get_node_shared_ptr());
        // auto split_lengths_value = split_lengths->cast_vector<int64_t>()[0];
        // // Allow only case that exactly splits in half along the last dimension
        // auto split_length = variadic_split_in_ps[last_dim].get_length() / 2;
        // if (split_lengths_value != split_length)
        //     return false;

        // auto data = pattern_map.at(data_m);
        // auto output_type = m.get_match_root()->get_output_element_type(0);

        // auto swiglu = std::make_shared<op::SwiGLU>(data,
        //                                            axis_value,
        //                                            split_lengths_value,
        //                                            output_type);
        // swiglu->set_friendly_name(m.get_match_root()->get_friendly_name());
        // ov::copy_runtime_info(m.get_matched_nodes(), swiglu);
        // ov::replace_node(m.get_match_root(), swiglu);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul6, "FeedForwardFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov