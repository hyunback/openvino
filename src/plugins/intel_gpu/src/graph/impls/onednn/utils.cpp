// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include "onednn_formats_map.hpp"
#include <oneapi/dnnl/dnnl_debug.h>
#include <numeric>
#include <oneapi/dnnl/dnnl_ocl.hpp>

#include "to_string_utils.h"

namespace cldnn {
namespace onednn {

namespace {
std::string convert_data_format_string(cldnn::format fmt) {
    switch (fmt) {
        case cldnn::format::b_fs_yx_fsv2: return "aBcd2b";
        case cldnn::format::b_fs_zyx_fsv2: return "aBcde2b";
        case cldnn::format::bs_fs_yx_bsv16_fsv2: return "ABcd16a2b";
        case cldnn::format::bs_fs_zyx_bsv16_fsv2: return "ABcde16a2b";
        case cldnn::format::bs_fs_yx_bsv16_fsv4: return "ABcd16a4b";
        case cldnn::format::bs_fs_zyx_bsv16_fsv4: return "ABcde16a4b";
        case cldnn::format::bs_fs_yx_bsv16_fsv32: return "ABcd16a32b";
        case cldnn::format::bs_fs_zyx_bsv16_fsv32: return "ABcde16a32b";
        default: throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to onednn layout string" + fmt_to_str(fmt));
    }
}

}  // namespace

template <typename T>
cldnn::memory::ptr convert_zp_data_to_s32(const memory::ptr zp_memory) {
    auto engine = zp_memory->get_engine();
    auto& stream = engine->get_program_stream();

    auto zp_s32_layout = zp_memory->get_layout();
    zp_s32_layout.data_type = data_types::i32;
    auto zp_s32_memory = engine->allocate_memory(zp_s32_layout, false);

    mem_lock<T, mem_lock_type::read> zp_data(zp_memory, stream);
    mem_lock<int32_t, mem_lock_type::write> zp_s32_data(zp_s32_memory, stream);
    for (size_t i = 0; i < zp_data.size(); i++) {
        zp_s32_data.data()[i] = static_cast<int32_t>(zp_data.data()[i]);
    }

    return zp_s32_memory;
}

template cldnn::memory::ptr convert_zp_data_to_s32<int8_t>(const memory::ptr zp_memory);
template cldnn::memory::ptr convert_zp_data_to_s32<uint8_t>(const memory::ptr zp_memory);

cldnn::format default_fmt_for_dims(size_t dims, bool is_grouped) {
    switch (dims) {
    case 6: return is_grouped ? cldnn::format::goizyx : cldnn::format::bfwzyx;
    case 5: return is_grouped ? cldnn::format::goiyx : cldnn::format::bfzyx;
    default: return cldnn::format::bfyx;
    }
}

dnnl::memory::dims convert_tensor(cldnn::tensor t, size_t dims, bool is_grouped) {
    auto sizes = t.sizes(default_fmt_for_dims(dims, is_grouped));
    dnnl::memory::dims res(sizes.begin(), sizes.end());
    return res;
}

dnnl::memory::dims convert_gemm_tensor(cldnn::tensor t, size_t dims, bool batched_dims_can_be_removed) {
    auto sizes = t.sizes(default_fmt_for_dims(dims, false));
    dnnl::memory::dims res(sizes.begin(), sizes.end());
    if (dims > 3) {
        for (size_t i = 0; i < dims - 3; i++) {
            res[i + 1] *= res[i];
        }
        res.erase(res.begin(), res.begin() + dims - 3);
    }
    if (res.size() == 3 && batched_dims_can_be_removed) {
        res.erase(res.begin());
    }
    return res;
}

dnnl::memory::format_tag convert_gemm_data_format(dnnl::memory::dims dims) {
    if (dims.size() > 3)
        throw std::runtime_error("[clDNN] Unsupported dims size for onednn gemm: should be <= 3");
    return dims.size() == 3 ? dnnl::memory::format_tag::abc : dnnl::memory::format_tag::ab;
}


dnnl::memory::dims convert_spatials(cldnn::tensor t, size_t dims) {
    auto spatials = t.spatial;
    dnnl::memory::dims res(dims);
    for (size_t i = 0; i < dims; i++) {
        res[i] = spatials[dims - i - 1];
    }
    return res;
}

dnnl::memory::dims flatten_tensor(cldnn::tensor t) {
    return {static_cast<int64_t>(t.count())};
}

dnnl::memory::data_type convert_data_type(cldnn::data_types dt) {
    switch (dt) {
        case cldnn::data_types::f32: return dnnl::memory::data_type::f32;
        case cldnn::data_types::f16: return dnnl::memory::data_type::f16;
        case cldnn::data_types::i8: return dnnl::memory::data_type::s8;
        case cldnn::data_types::u8: return dnnl::memory::data_type::u8;
        case cldnn::data_types::i32: return dnnl::memory::data_type::s32;
        default: throw std::invalid_argument("[clDNN] Unsupported conversion from cldnn to onednn type");
    }
}

std::vector<std::pair<cldnn::format, dnnl::memory::format_tag>> format_map = {
        { cldnn::format::bfyx, dnnl::memory::format_tag::nchw },
        { cldnn::format::bfzyx, dnnl::memory::format_tag::ncdhw },
        { cldnn::format::byxf, dnnl::memory::format_tag::nhwc },
        { cldnn::format::bzyxf, dnnl::memory::format_tag::ndhwc },
        { cldnn::format::b_fs_yx_fsv2, dnnl::memory::format_tag::undef },
        { cldnn::format::b_fs_yx_fsv4, dnnl::memory::format_tag::aBcd4b },
        { cldnn::format::b_fs_yx_fsv16, dnnl::memory::format_tag::nChw16c },
        { cldnn::format::b_fs_yx_fsv32, dnnl::memory::format_tag::aBcd32b },
        { cldnn::format::b_fs_zyx_fsv4, dnnl::memory::format_tag::aBcde4b },
        { cldnn::format::b_fs_zyx_fsv16, dnnl::memory::format_tag::nCdhw16c },
        { cldnn::format::b_fs_zyx_fsv32, dnnl::memory::format_tag::aBcde32b },
        { cldnn::format::bs_fs_yx_bsv16_fsv16, dnnl::memory::format_tag::NChw16n16c },
        { cldnn::format::bs_fs_yx_bsv16_fsv32, dnnl::memory::format_tag::NChw16n32c },
        { cldnn::format::bs_fs_yx_bsv32_fsv32, dnnl::memory::format_tag::NChw32n32c },
        { cldnn::format::bs_fs_yx_bsv4_fsv4, dnnl::memory::format_tag::ABcd4a4b },
        { cldnn::format::bs_fs_yx_bsv8_fsv4, dnnl::memory::format_tag::ABcd8a4b },
        { cldnn::format::bs_fs_yx_bsv8_fsv2, dnnl::memory::format_tag::ABcd8a2b },
        { cldnn::format::bs_fs_yx_bsv4_fsv2, dnnl::memory::format_tag::ABcd4a2b },
        { cldnn::format::bs_fs_yx_bsv32_fsv16, dnnl::memory::format_tag::NChw32n16c },
        { cldnn::format::bs_fs_zyx_bsv32_fsv16, dnnl::memory::format_tag::NCdhw32n16c },
        { cldnn::format::bs_fs_zyx_bsv32_fsv32, dnnl::memory::format_tag::NCdhw32n32c },
        { cldnn::format::bs_fs_zyx_bsv16_fsv16, dnnl::memory::format_tag::NCdhw16n16c },
        // { cldnn::format::bs_fs_zyx_bsv16_fsv32, dnnl::memory::format_tag::NCdhw16n32c }, // TODO onednn3.0: Request NCdhw16n32c format
        { cldnn::format::bs_fs_zyx_bsv8_fsv4, dnnl::memory::format_tag::ABcde8a4b },
        { cldnn::format::bs_fs_zyx_bsv8_fsv2, dnnl::memory::format_tag::ABcde8a2b },
};

dnnl::memory::format_tag convert_data_format(cldnn::format fmt) {
    auto ret = std::find_if(format_map.begin(), format_map.end(),
            [fmt](std::pair<cldnn::format, dnnl::memory::format_tag> &e) {
                    return e.first == fmt; });
    if (ret == format_map.end())
        return dnnl::memory::format_tag::undef;

    return ret->second;
}

 cldnn::format convert_data_format(dnnl::memory::format_tag fmt) {
    auto ret = std::find_if(format_map.begin(), format_map.end(),
            [fmt](std::pair<cldnn::format, dnnl::memory::format_tag> &e) {
                    return e.second == fmt; });
    if (ret == format_map.end())
        throw std::invalid_argument("[clDNN] Unsupported onednn layout");

    return ret->first;
}

void combine_bf_with_first_spatial_dim(cldnn::layout& l) {
    auto pshape = l.get_shape();
    ov::Shape new_shape{1, 1};
    for (size_t i = 0; i < pshape.size(); ++i) {
        if (i < 2) {
            new_shape[0] *= pshape[i];
        } else {
            new_shape[1] *= pshape[i];
        }
    }
    l.set_partial_shape(new_shape);
}

int64_t get_f_offset(cldnn::layout&& l, dnnl::memory::desc&& desc) {
    int64_t offset = 0;
    auto f_padding = l.data_padding.lower_size().feature[0];
    if (f_padding != 0) {
        offset = f_padding;
        for (size_t i = 0; i < l.get_spatial_rank(); ++i) {
            offset *= l.spatial(i);
        }
    }

    switch (desc.get_data_type()) {
        case dnnl::memory::data_type::s8:
        case dnnl::memory::data_type::u8:
            return offset;
        case dnnl::memory::data_type::f16:
        case dnnl::memory::data_type::bf16:
            return (offset * 2);
        case dnnl::memory::data_type::f32:
        case dnnl::memory::data_type::s32:
            return (offset * 4);
        default:
            throw std::runtime_error(std::string("Unsupported offset for dnnl_data_type_t ")
                    + dnnl_dt2str(static_cast<dnnl_data_type_t>(desc.get_data_type())));
    }
}

dnnl::memory::desc layout_to_memory_desc(cldnn::layout l, dnnl::memory::format_tag target_fmt, bool flatten) {
    dnnl::memory::dims dims;
    if (target_fmt == dnnl::memory::format_tag::ab && flatten) {
        dims = flatten_tensor(l.get_tensor());
        dims.insert(dims.begin(), 1);
    } else if (target_fmt == dnnl::memory::format_tag::ab) {
        dims.push_back(l.batch());
        dims.push_back(l.get_tensor().count() / l.batch());
    } else if (flatten) {
        dims = flatten_tensor(l.get_tensor());
    } else {
        auto rank = cldnn::format::dimension(l.format);
        dims = convert_tensor(l.get_tensor(), rank, cldnn::format::is_grouped(l.format));
    }

    dnnl::memory::data_type dt = convert_data_type(l.data_type);
    dnnl::memory::format_tag fmt = target_fmt == dnnl::memory::format_tag::undef ? convert_data_format(l.format) : target_fmt;

    if (fmt == dnnl::memory::format_tag::undef) {
        throw ov::Exception("[GPU] Fix me, Unexpected fmt: " + convert_data_format_string(l.format));
    } else {
        dnnl::memory::desc res(dims, dt, fmt);

        return res;
    }
}

static bool isSame(dnnl::memory::desc desc, dnnl::memory::format_tag fmt) {
    dnnl::memory::desc refDesc(desc.get_dims(), desc.get_data_type(), fmt);

    if (desc.get_ndims() != refDesc.get_ndims())
        return false;

    if (desc.get_format_kind() != dnnl::memory::format_kind::blocked || refDesc.get_format_kind() != dnnl::memory::format_kind::blocked)
        throw std::runtime_error("dnnlMemoryDesc::isSame is not implemented for non blocked memory format");

    if (desc.get_inner_nblks() != refDesc.get_inner_nblks())
        return false;

    for (int i = 0; i < desc.get_inner_nblks(); ++i)
        if (desc.get_inner_blks()[i] != refDesc.get_inner_blks()[i])
            return false;

    for (int i = 0; i < desc.get_inner_nblks(); ++i)
        if (desc.get_inner_idxs()[i] != refDesc.get_inner_idxs()[i])
            return false;

    auto actualStrides = desc.get_strides();
    auto refStrides = refDesc.get_strides();

    std::vector<size_t> actualOrder(desc.get_ndims());
    std::iota(actualOrder.begin(), actualOrder.end(), 0);
    std::sort(actualOrder.begin(), actualOrder.end(),
              [&actualStrides] (size_t ind_l, size_t ind_r) {
                  return actualStrides[ind_l] > actualStrides[ind_r];
              });

    std::vector<size_t> refOrder(refDesc.get_ndims());
    std::iota(refOrder.begin(), refOrder.end(), 0);
    std::sort(refOrder.begin(), refOrder.end(),
              [&refStrides] (size_t ind_l, size_t ind_r) {
                  return refStrides[ind_l] > refStrides[ind_r];
              });

    if (actualOrder != refOrder) {
        return false;
    }

    return true;
}

dnnl::memory::format_tag get_format_by_desc(dnnl::memory::desc desc) {
    // TODO [OneDNN]: Previously it was a field of tdesc, but now the brute
    //                force search here. Please avoid of using this method.
    const auto ndims = desc.get_ndims(); // desc.dims().size();

    // There are no suitable format_tag for this
    if (ndims == 0 || ndims > 6)
        return dnnl::memory::format_tag::undef;

    for (const auto fmt : form_tags_by_ndims.at(static_cast<int>(ndims))) {
        if (isSame(desc, fmt))
            return fmt;
    }

    return dnnl::memory::format_tag::undef;
}

static std::vector<size_t> get_order(dnnl::memory::desc desc) {
    auto strides = desc.get_strides();
    std::vector<size_t> order(desc.get_ndims());

    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
                [&strides] (size_t ind_l, size_t ind_r) {
                    return (strides[ind_l] > strides[ind_r]);
                });
    return order;
}

static bool compare_strides(std::vector<size_t> a, std::vector<size_t> b) {
    return std::equal(a.begin(), a.end(), b.begin());
}

cldnn::format find_data_format(dnnl::memory::desc desc) {
    auto onednn_desc = get_format_by_desc(desc);

    if (onednn_desc != dnnl::memory::format_tag::undef) {
        return convert_data_format(onednn_desc);
    } else {
        auto order = get_order(desc);
        for (int32_t fmt_idx = format::bfyx ; fmt_idx < format::format_num ; fmt_idx++) {
            auto candidate_trait = format::traits(static_cast<format::type>(fmt_idx));
            if (desc.get_ndims() == static_cast<int>(candidate_trait._order.size())
                && desc.get_inner_nblks() == static_cast<int>(candidate_trait.block_sizes.size())
                && compare_strides(order, candidate_trait._order)) {
                bool is_match = true;
                for (size_t idx = 0 ; idx < candidate_trait.block_sizes.size() ; idx++) {
                    if (desc.get_inner_blks()[idx] != static_cast<int>(candidate_trait.block_sizes[idx].second)
                        || desc.get_inner_idxs()[idx] != static_cast<int>(candidate_trait.block_sizes[idx].first)) {
                        is_match = false;
                        break;
                    }
                }

                if (is_match)
                    return static_cast<format::type>(fmt_idx);
            }
        }

        std::stringstream msg;
        msg << "Unsupported onednn dnnl::memory::desc find_data_format. "
            << "ndims: " << desc.get_ndims()
            << ", inner_nblks: " << desc.get_inner_nblks()
            << ", inner_blks: ";
        for (int i = 0; i < desc.get_inner_nblks(); i++)
            msg << "(blk " << desc.get_inner_blks()[i] << ", idx " << desc.get_inner_idxs()[i] << ") ";

        throw std::runtime_error(msg.str());
    }
}


// onednn -> cldnn
static cldnn::format convert_format(dnnl::memory::format_tag fmt, bool is_grouped) {
    if (is_grouped) {
        switch (fmt) {
        case dnnl::memory::format_tag::abcde: return cldnn::format::goiyx;
        case dnnl::memory::format_tag::abcdef: return cldnn::format::goizyx;
        case dnnl::memory::format_tag::Abcdef16a: return cldnn::format::gs_oizyx_gsv16;
        case dnnl::memory::format_tag::Abcde16a: return cldnn::format::gs_oiyx_gsv16;
        case dnnl::memory::format_tag::Abcde32a: return cldnn::format::gs_oiyx_gsv32;
        case dnnl::memory::format_tag::Abcdef32a: return cldnn::format::gs_oizyx_gsv32;
        case dnnl::memory::format_tag::aCBde16c16b: return cldnn::format::g_is_os_yx_isv16_osv16;
        case dnnl::memory::format_tag::aBCde2b8c8b2c: return cldnn::format::g_os_is_yx_osa2_isa8_osv8_isv2;
        case dnnl::memory::format_tag::aBCde4b8c8b4c: return cldnn::format::g_os_is_yx_osa4_isa8_osv8_isv4;
        case dnnl::memory::format_tag::aBCde4b8c8b2c: return cldnn::format::g_os_is_yx_osa4_isa8_osv8_isv2;
        case dnnl::memory::format_tag::aBCde8b2c: return cldnn::format::g_os_is_yx_osv8_isv2;
        case dnnl::memory::format_tag::aBCde8b4c: return cldnn::format::g_os_is_yx_osv8_isv4;
        case dnnl::memory::format_tag::aBcde8b: return cldnn::format::g_os_iyx_osv8;
        case dnnl::memory::format_tag::aBCd2b8c16b4c: return cldnn::format::g_os_is_yx_osa2_isa8_osv16_isv4;
        case dnnl::memory::format_tag::aBCd2b8c16b2c: return cldnn::format::g_os_is_yx_osa2_isa8_osv16_isv2;
        case dnnl::memory::format_tag::aBCdef16c16b: return cldnn::format::g_os_is_zyx_isv16_osv16;
        case dnnl::memory::format_tag::aBCdef4b8c8b2c: return cldnn::format::g_os_is_zyx_osa4_isa8_osv8_isv2;
        case dnnl::memory::format_tag::aBCdef4b8c8b4c: return cldnn::format::g_os_is_zyx_osa4_isa8_osv8_isv4;
        default: throw std::runtime_error(std::string("Unsupported grouped onednn fmt ") + dnnl_fmt_tag2str((dnnl_format_tag_t)fmt));
        }
    } else {
        switch (fmt) {
        case dnnl::memory::format_tag::ab: return cldnn::format::oiyx;
        case dnnl::memory::format_tag::abcd: return cldnn::format::oiyx;
        case dnnl::memory::format_tag::bacd: return cldnn::format::ioyx;
        case dnnl::memory::format_tag::bcda: return cldnn::format::iyxo;
        case dnnl::memory::format_tag::BAcd16b16a: return cldnn::format::is_os_yx_isv16_osv16;
        case dnnl::memory::format_tag::ABcd16b16a: return cldnn::format::os_is_yx_isv16_osv16;
        case dnnl::memory::format_tag::abcde: return cldnn::format::oizyx;
        case dnnl::memory::format_tag::ABcd4a8b8a4b: return cldnn::format::os_is_yx_osa4_isa8_osv8_isv4;
        case dnnl::memory::format_tag::ABcd4a8b8a2b: return cldnn::format::os_is_yx_osa4_isa8_osv8_isv2;
        case dnnl::memory::format_tag::ABcde4a8b8a2b: return cldnn::format::os_is_zyx_osa4_isa8_osv8_isv2;
        case dnnl::memory::format_tag::ABcde4a8b8a4b: return cldnn::format::os_is_zyx_osa4_isa8_osv8_isv4;
        case dnnl::memory::format_tag::ABcd8a4b: return cldnn::format::os_is_yx_osv8_isv4;
        case dnnl::memory::format_tag::ABcde8a4b: return cldnn::format::os_is_zyx_osv8_isv4;
        case dnnl::memory::format_tag::ABcde8a2b: return cldnn::format::os_is_zyx_osv8_isv2;
        case dnnl::memory::format_tag::ABcd8a2b: return cldnn::format::os_is_yx_osv8_isv2;
        case dnnl::memory::format_tag::Acdb16a: return cldnn::format::os_yxi_osv16;
        case dnnl::memory::format_tag::Acdeb16a: return cldnn::format::os_zyxi_osv16;
        case dnnl::memory::format_tag::ABcde16b16a: return cldnn::format::os_is_zyx_isv16_osv16;
        case dnnl::memory::format_tag::aBcd16b: return cldnn::format::o_is_yx_isv16;
        case dnnl::memory::format_tag::Abcd16a: return cldnn::format::os_iyx_osv16;
        case dnnl::memory::format_tag::ABcd2a8b8a2b: return cldnn::format::os_is_yx_osa2_isa8_osv8_isv2;
        case dnnl::memory::format_tag::ABcd2a8b16a4b: return cldnn::format::os_is_yx_osa2_isa8_osv16_isv4;
        case dnnl::memory::format_tag::ABcd2a8b16a2b: return cldnn::format::os_is_yx_osa2_isa8_osv16_isv2;
        case dnnl::memory::format_tag::BAcd4b8a8b4a: return cldnn::format::is_os_yx_isa4_osa8_isv8_osv4;
        default: throw std::runtime_error(std::string("Unsupported onednn fmt ") + dnnl_fmt_tag2str((dnnl_format_tag_t)fmt));
        }
    }
}

cldnn::format find_format(dnnl::memory::desc desc, bool is_grouped) {
    auto onednn_desc = get_format_by_desc(desc);

    if (onednn_desc != dnnl::memory::format_tag::undef) {
        return convert_format(onednn_desc, is_grouped);
    } else {
        auto ndims = desc.get_ndims();
        auto inner_nblks = desc.get_inner_nblks();
        auto inner_blks = desc.get_inner_blks();
        auto inner_idxs = desc.get_inner_idxs();

        auto order = get_order(desc);
        if (is_grouped) {
            if (ndims == 5 && inner_nblks == 3
                && inner_blks[0] == 8 && inner_blks[1] == 8 && inner_blks[2] == 2
                && inner_idxs[0] == 2 && inner_idxs[1] == 1 && inner_idxs[2] == 2
                && compare_strides(order, {0, 1, 2, 3, 4})) {
                return cldnn::format::g_os_is_yx_isa8_osv8_isv2;
            }  else if (ndims == 5 && inner_nblks == 3
                && inner_blks[0] == 8 && inner_blks[1] == 8 && inner_blks[2] == 4
                && inner_idxs[0] == 2 && inner_idxs[1] == 1 && inner_idxs[2] == 2
                && compare_strides(order, {0, 1, 2, 3, 4})) {
                return cldnn::format::g_os_is_yx_isa8_osv8_isv4;
            } else if (ndims == 5 && inner_nblks == 2
                && inner_blks[0] == 8 && inner_blks[1] == 2
                && inner_idxs[0] == 1 && inner_idxs[1] == 2) {
                    if (compare_strides(order, {0, 1, 3, 4, 2}))        return cldnn::format::g_os_yx_is_osv8_isv2;
                    else if (compare_strides(order, {0, 1, 3, 2, 4}))   return cldnn::format::g_os_y_is_x_osv8_isv2;
            } else if (ndims == 5 && inner_nblks == 2
                && inner_blks[0] == 8 && inner_blks[1] == 4
                && inner_idxs[0] == 1 && inner_idxs[1] == 2) {
                    if (compare_strides(order, {0, 1, 3, 4, 2}))        return cldnn::format::g_os_yx_is_osv8_isv4;
                    else if (compare_strides(order, {0, 1, 3, 2, 4}))   return cldnn::format::g_os_y_is_x_osv8_isv4;
            } else if (ndims == 6 && inner_nblks == 2
                && inner_blks[0] == 8 && inner_blks[1] == 2
                && inner_idxs[0] == 1 && inner_idxs[1] == 2) {
                    if (compare_strides(order, {0, 1, 3, 4, 5, 2}))       return cldnn::format::g_os_zyx_is_osv8_isv2;
                    else if (compare_strides(order, {0, 1, 3, 4, 2, 5}))  return cldnn::format::g_os_zy_is_x_osv8_isv2;
            } else if (ndims == 6 && inner_nblks == 2
                && inner_blks[0] == 8 && inner_blks[1] == 4
                && inner_idxs[0] == 1 && inner_idxs[1] == 2) {
                    if (compare_strides(order, {0, 1, 3, 4, 5, 2}))       return cldnn::format::g_os_zyx_is_osv8_isv4;
                    else if (compare_strides(order, {0, 1, 3, 4, 2, 5}))  return cldnn::format::g_os_zy_is_x_osv8_isv4;
            } else if (ndims == 6 && inner_nblks == 3
                && inner_blks[0] == 8 && inner_blks[1] == 8 && inner_blks[2] == 2
                && inner_idxs[0] == 2 && inner_idxs[1] == 1 && inner_idxs[2] == 2
                && compare_strides(order, {0, 1, 2, 3, 4, 5})) {
                return cldnn::format::g_os_is_zyx_isa8_osv8_isv2;
            } else if (ndims == 6 && inner_nblks == 3
                && inner_blks[0] == 8 && inner_blks[1] == 8 && inner_blks[2] == 4
                && inner_idxs[0] == 2 && inner_idxs[1] == 1 && inner_idxs[2] == 2
                && compare_strides(order, {0, 1, 2, 3, 4, 5})) {
                return cldnn::format::g_os_is_zyx_isa8_osv8_isv4;
            }
        } else {
            if (ndims == 4 && inner_nblks == 4
                && inner_blks[0] == 4 && inner_blks[1] == 8 && inner_blks[2] == 8 && inner_blks[3] == 4
                && inner_idxs[0] == 0 && inner_idxs[1] == 1 && inner_idxs[2] == 0 && inner_idxs[3] == 1
                && compare_strides(order, {1, 0, 2, 3})) {
                return cldnn::format::is_os_yx_osa4_isa8_osv8_isv4;
            } else if (ndims == 4 && inner_nblks == 4
                && inner_blks[0] == 2 && inner_blks[1] == 8 && inner_blks[2] == 8 && inner_blks[3] == 2
                && inner_idxs[0] == 1 && inner_idxs[1] == 0 && inner_idxs[2] == 1 && inner_idxs[3] == 0
                && compare_strides(order, {1, 0, 2, 3})) {
                return cldnn::format::is_os_yx_isa2_osa8_isv8_osv2;
            } else if (ndims == 4 && inner_nblks == 2
                && inner_blks[0] == 16 && inner_blks[1] == 4 && inner_idxs[0] == 0 && inner_idxs[1] == 1
                && compare_strides(order, {0, 1, 2, 3})) {
                return cldnn::format::os_is_yx_osv16_isv4;
            } else if (ndims == 4 && inner_nblks == 2
                && inner_blks[0] == 16 && inner_blks[1] == 8 && inner_idxs[0] == 1 && inner_idxs[1] == 0
                && compare_strides(order, {0, 1, 2, 3})) {
                return cldnn::format::is_os_yx_isv16_osv8;
            } else if (ndims == 4 && inner_nblks == 3
                && inner_blks[0] == 8 && inner_blks[1] == 8 && inner_blks[2] == 2
                && inner_idxs[0] == 1 && inner_idxs[1] == 0 && inner_idxs[2] == 1) {
                if (compare_strides(order, {0, 1, 2, 3}))           return cldnn::format::os_is_yx_isa8_osv8_isv2;
                else if (compare_strides(order, {1, 0, 2, 3}))      return cldnn::format::is_os_yx_isa8_osv8_isv2;
            } else if (ndims == 4 && inner_nblks == 3
                && inner_blks[0] == 8 && inner_blks[1] == 8 && inner_blks[2] == 4
                && inner_idxs[0] == 1 && inner_idxs[1] == 0 && inner_idxs[2] == 1) {
                if (compare_strides(order, {0, 1, 2, 3}))           return cldnn::format::os_is_yx_isa8_osv8_isv4;
                else if (compare_strides(order, {1, 0, 2, 3}))      return cldnn::format::is_os_yx_isa8_osv8_isv4;
            } else if (ndims == 4 && inner_nblks == 2
                && inner_blks[0] == 8 && inner_blks[1] == 2
                && inner_idxs[0] == 0 && inner_idxs[1] == 1) {
                if (compare_strides(order, {0, 2, 1, 3}))           return cldnn::format::os_y_is_x_osv8_isv2;
                else if (compare_strides(order, {0, 2, 3, 1}))      return cldnn::format::os_yx_is_osv8_isv2;
            } else if (ndims == 4 && inner_nblks == 2
                && inner_blks[0] == 8 && inner_blks[1] == 4
                && inner_idxs[0] == 0 && inner_idxs[1] == 1) {
                if (compare_strides(order, {0, 2, 1, 3}))           return cldnn::format::os_y_is_x_osv8_isv4;
                else if (compare_strides(order, {0, 2, 3, 1}))      return cldnn::format::os_yx_is_osv8_isv4;
            } else if (ndims == 5 && inner_nblks == 2 &&
                inner_blks[0] == 8 && inner_blks[1] == 2 &&
                inner_idxs[0] == 0 && inner_idxs[1] == 1) {
                if (compare_strides(order, {0, 2, 3, 4, 1}))        return cldnn::format::os_zyx_is_osv8_isv2;
                else if (compare_strides(order, {0, 2, 3, 1, 4}))   return cldnn::format::os_zy_is_x_osv8_isv2;
            } else if (ndims == 5 && inner_nblks == 2 &&
                inner_blks[0] == 8 && inner_blks[1] == 4 &&
                inner_idxs[0] == 0 && inner_idxs[1] == 1) {
                if (compare_strides(order, {0, 2, 3, 4, 1}))        return cldnn::format::os_zyx_is_osv8_isv4;
                else if (compare_strides(order, {0, 2, 3, 1, 4}))   return cldnn::format::os_zy_is_x_osv8_isv4;
            } else if (ndims == 5 && inner_nblks == 3
                && inner_blks[0] == 8 && inner_blks[1] == 8 && inner_blks[2] == 2
                && inner_idxs[0] == 1 && inner_idxs[1] == 0 && inner_idxs[2] == 1) {
                if (compare_strides(order, {0, 1, 2, 3, 4}))        return cldnn::format::os_is_zyx_isa8_osv8_isv2;
                else if (compare_strides(order, {1, 0, 2, 3, 4}))   return cldnn::format::is_os_zyx_isa8_osv8_isv2;
            } else if (ndims == 5 && inner_nblks == 3
                && inner_blks[0] == 8 && inner_blks[1] == 8 && inner_blks[2] == 4
                && inner_idxs[0] == 1 && inner_idxs[1] == 0 && inner_idxs[2] == 1) {
                if (compare_strides(order, {0, 1, 2, 3, 4}))        return cldnn::format::os_is_zyx_isa8_osv8_isv4;
                else if (compare_strides(order, {1, 0, 2, 3, 4}))   return cldnn::format::is_os_zyx_isa8_osv8_isv4;
            } else if (ndims == 5 && inner_nblks == 4 &&
                inner_blks[0] == 2 && inner_blks[1] == 8 && inner_blks[2] == 8 && inner_blks[3] == 2 &&
                inner_idxs[0] == 0 && inner_idxs[1] == 1 && inner_idxs[2] == 0 && inner_idxs[3] == 1) {
                return cldnn::format::os_is_zyx_osa2_isa8_osv8_isv2;
            } else if (ndims == 5 && inner_nblks == 4 &&
                inner_blks[0] == 4 && inner_blks[1] == 8 && inner_blks[2] == 8 && inner_blks[3] == 4 &&
                inner_idxs[0] == 0 && inner_idxs[1] == 1 && inner_idxs[2] == 0 && inner_idxs[3] == 1) {
                return cldnn::format::os_is_zyx_osa4_isa8_osv8_isv4;
            }
        }

        std::stringstream msg;
        msg << "Unsupported " << (is_grouped ? "grouped" : "") << "onednn dnnl::memory::desc find_format. "
            << "ndims: " << ndims
            << ", inner_nblks: " << desc.get_inner_nblks()
            << ", inner_blks: ";
        for (int i = 0; i < desc.get_inner_nblks(); i++)
            msg << "(blk " << desc.get_inner_blks()[i] << ", idx " << desc.get_inner_idxs()[i] << ") ";
        msg << ", strides_order: ";
        for (const auto& value : order)
            msg << value << " ";

        throw std::runtime_error(msg.str());
    }
}

// Currently, usage of alpha and beta between cldnn::pow and dnnl::eltwise::pow is different : d = pow(src, a) / d = a * pow(src, b)
dnnl::algorithm convert_activation_func(cldnn::activation_func func) {
    switch (func) {
        case cldnn::activation_func::relu: return dnnl::algorithm::eltwise_relu;
        case cldnn::activation_func::relu_negative_slope: return dnnl::algorithm::eltwise_relu;
        case cldnn::activation_func::gelu: return dnnl::algorithm::eltwise_gelu_tanh;   // TODO check is it correct? return dnnl::algorithm::eltwise_gelu;
        case cldnn::activation_func::elu: return dnnl::algorithm::eltwise_elu;
        case cldnn::activation_func::mish: return dnnl::algorithm::eltwise_mish;
        case cldnn::activation_func::swish: return dnnl::algorithm::eltwise_swish;
        case cldnn::activation_func::hswish: return dnnl::algorithm::eltwise_hardswish;
        case cldnn::activation_func::abs: return dnnl::algorithm::eltwise_abs;
        case cldnn::activation_func::exp: return dnnl::algorithm::eltwise_exp;
        case cldnn::activation_func::logistic: return dnnl::algorithm::eltwise_logistic;
        case cldnn::activation_func::clamp: return dnnl::algorithm::eltwise_clip;
        case cldnn::activation_func::hyperbolic_tan: return dnnl::algorithm::eltwise_tanh;
        case cldnn::activation_func::pow: return dnnl::algorithm::eltwise_pow;
        case cldnn::activation_func::sqrt: return dnnl::algorithm::eltwise_sqrt;
        case cldnn::activation_func::hard_sigmoid: return dnnl::algorithm::eltwise_hardsigmoid;
        default: throw std::runtime_error("Unsupported activation func for onednn primitive " + std::to_string(static_cast<int>(func)));
    }
}

template <typename T>
bool is_per_tensor(cldnn::data_node& node, int32_t& zp_val) {
    auto ptr = node.get_attached_memory_ptr();
    auto engine = ptr->get_engine();
    auto& stream = engine->get_program_stream();
    auto num_elems = node.get_output_layout().count();
    mem_lock<T, mem_lock_type::read> old_data {ptr, stream};
    auto val = old_data[0];
    for (size_t i = 1; i < num_elems; i++) {
        if (val != old_data[i]) {
            zp_val = DNNL_RUNTIME_S32_VAL;
            return false;
        }
    }

    zp_val = val;
    return true;
}

template bool is_per_tensor<int8_t>(cldnn::data_node& node, int32_t& zp_val);
template bool is_per_tensor<uint8_t>(cldnn::data_node& node, int32_t& zp_val);

}  // namespace onednn
}  // namespace cldnn
