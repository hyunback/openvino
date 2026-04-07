// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_stream.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"

#include "ze_counter_based_event_factory.hpp"
#include "ze_event_factory.hpp"
#include "ze_events.hpp"
#include "ze_empty_event.hpp"

#include "ze_event.hpp"
#include "ze_kernel.hpp"
#include "ze_memory.hpp"
#include "ze_common.hpp"

#include <ze_api.h>
#include "compute_runtime/ze_intel_gpu.h"
#include "compute_runtime/ze_stypes.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <memory>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ze.hpp>
#endif

namespace cldnn {
namespace ze {

namespace {
inline int get_regular_cmd_list_poc_level() {
    const char* val = std::getenv("OV_GPU_ZE_FORCE_REGULAR_CMD");
    if (val == nullptr) return 0;
    int level = std::atoi(val);
    return level > 0 ? level : 0;
}

inline bool use_regular_cmd_list_poc() {
    return get_regular_cmd_list_poc_level() >= 1;
}

inline void regular_poc_log(const std::string& msg) {
    if (get_regular_cmd_list_poc_level() >= 2) {
        std::cerr << "[REGULAR_POC] " << msg << std::endl;
    }
}

inline ze_group_count_t to_group_count(const std::vector<size_t>& v) {
     switch (v.size()) {
        case 1:
            return {uint32_t(v[0]), uint32_t(1), uint32_t(1)};
        case 2:
            return {uint32_t(v[0]), uint32_t(v[1]), uint32_t(1)};
        case 3:
            return {uint32_t(v[0]), uint32_t(v[1]), uint32_t(v[2])};
        default:
            return {uint32_t(1), uint32_t(1), uint32_t(1)};
    }
}

template<typename T>
ze_result_t set_kernel_arg_scalar(ze_kernel_handle_t& kernel, uint32_t idx, const T& val) {
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set scalar " << idx << " (" << ov::element::from<T>().get_type_name() << ")" << val << "\n";
    return zeKernelSetArgumentValue(kernel, idx, sizeof(T), &val);
}

ze_result_t set_kernel_arg_local_memory(ze_kernel_handle_t& kernel, uint32_t idx, size_t size) {
    if (size == 0)
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;

    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set arg " << idx << " local memory size: " << size << std::endl;
    return zeKernelSetArgumentValue(kernel, idx, size, NULL);
}

ze_result_t set_kernel_arg(ze_kernel_handle_t& kernel, uint32_t idx, cldnn::memory::cptr mem) {
    if (!mem)
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;

    OPENVINO_ASSERT(memory_capabilities::is_usm_type(mem->get_allocation_type()), "Unsupported alloc type");
    const auto& buf = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_buffer();
    auto mem_type = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_allocation_type();
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set arg (" << mem_type << ") " << idx
                            << " mem: " << buf.get() << " size: " << mem->size() << std::endl;

    auto ptr = buf.get();
    return zeKernelSetArgumentValue(kernel, idx, sizeof(ptr), &ptr);
}

void set_arguments_impl(ze_kernel_handle_t kernel,
                         const arguments_desc& args,
                         const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;

    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        ze_result_t status = ZE_RESULT_NOT_READY;
        switch (args[i].t) {
            case args_t::INPUT:
                if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.inputs[args[i].index]);
                }
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.fused_op_inputs[args[i].index]);
                }
                break;
            case args_t::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.intermediates[args[i].index]);
                }
                break;
            case args_t::OUTPUT:
                if (args[i].index < data.outputs.size() && data.outputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.outputs[args[i].index]);
                }
                break;
            case args_t::WEIGHTS:
                status = set_kernel_arg(kernel, i, data.weights);
                break;
            case args_t::BIAS:
                status = set_kernel_arg(kernel, i, data.bias);
                break;
            case args_t::WEIGHTS_ZERO_POINTS:
                status = set_kernel_arg(kernel, i, data.weights_zero_points);
                break;
            case args_t::ACTIVATIONS_ZERO_POINTS:
                status = set_kernel_arg(kernel, i, data.activations_zero_points);
                break;
            case args_t::COMPENSATION:
                status = set_kernel_arg(kernel, i, data.compensation);
                break;
            case args_t::SCALE_TABLE:
                status = set_kernel_arg(kernel, i, data.scale_table);
                break;
            case args_t::SLOPE:
                status = set_kernel_arg(kernel, i, data.slope);
                break;
            case args_t::SCALAR:
                if (data.scalars && args[i].index < data.scalars->size()) {
                    const auto& scalar = (*data.scalars)[args[i].index];
                    switch (scalar.t) {
                        case scalar_t::UINT8:
                            status = set_kernel_arg_scalar<uint8_t>(kernel, i, scalar.v.u8);
                            break;
                        case scalar_t::UINT16:
                            status = set_kernel_arg_scalar<uint16_t>(kernel, i, scalar.v.u16);
                            break;
                        case scalar_t::UINT32:
                            status = set_kernel_arg_scalar<uint32_t>(kernel, i, scalar.v.u32);
                            break;
                        case scalar_t::UINT64:
                            status = set_kernel_arg_scalar<uint64_t>(kernel, i, scalar.v.u64);
                            break;
                        case scalar_t::INT8:
                            status = set_kernel_arg_scalar<int8_t>(kernel, i, scalar.v.s8);
                            break;
                        case scalar_t::INT16:
                            status = set_kernel_arg_scalar<int16_t>(kernel, i, scalar.v.s16);
                            break;
                        case scalar_t::INT32:
                            status = set_kernel_arg_scalar<int32_t>(kernel, i, scalar.v.s32);
                            break;
                        case scalar_t::INT64:
                            status = set_kernel_arg_scalar<int64_t>(kernel, i, scalar.v.s64);
                            break;
                        case scalar_t::FLOAT32:
                            status = set_kernel_arg_scalar<float>(kernel, i, scalar.v.f32);
                            break;
                        case scalar_t::FLOAT64:
                            status = set_kernel_arg_scalar<double>(kernel, i, scalar.v.f64);
                            break;
                        default:
                            break;
                    }
                }
                break;
            case args_t::CELL:
                status = set_kernel_arg(kernel, i, data.cell);
                break;
            case args_t::SHAPE_INFO:
                status = set_kernel_arg(kernel, i, data.shape_info);
                break;
            case args_t::LOCAL_MEMORY_SIZE:
                OPENVINO_ASSERT(args[i].index < data.local_memory_args->size() && data.local_memory_args->at(args[i].index),
                                "The allocated local memory is necessary to set kernel arguments.");
                status = set_kernel_arg_local_memory(kernel, i,  data.local_memory_args->at(args[i].index));
                break;
            default:
                break;
        }
        if (status != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", error code: " + std::to_string(status) + "\n");
        }
    }
}

}  // namespace

ze_stream::ze_stream(const ze_engine &engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config))
    , _engine(engine) {
    const auto &info = engine.get_device_info();
    static std::atomic<uint16_t> stream_id{0};
    uint32_t index = stream_id++ % info.num_ccs;

    ze_command_queue_desc_t command_queue_desc = {};
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.pNext = nullptr;
    command_queue_desc.index = index;
    command_queue_desc.ordinal = info.compute_queue_group_ordinal;
    command_queue_desc.flags = m_queue_type == QueueTypes::out_of_order ? 0 : ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    zex_intel_queue_copy_operations_offload_hint_exp_desc_t cp_offload_desc = {};
    cp_offload_desc.stype = ZEX_INTEL_STRUCTURE_TYPE_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_PROPERTIES;
    cp_offload_desc.copyOffloadEnabled = true;
    cp_offload_desc.pNext = nullptr;
    bool use_cp_offload = info.supports_cp_offload;
    if (use_cp_offload) {
        command_queue_desc.pNext = &cp_offload_desc;
    }

    m_use_regular_cmd_queue = use_regular_cmd_list_poc();
    if (m_use_regular_cmd_queue) {
        regular_poc_log("ze_stream ctor: create single shared regular queue/list");
        ze_command_list_desc_t command_list_desc = {};
        command_list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
        command_list_desc.pNext = nullptr;
        command_list_desc.commandQueueGroupOrdinal = info.compute_queue_group_ordinal;
        command_list_desc.flags = m_queue_type == QueueTypes::out_of_order ? 0 : ZE_COMMAND_LIST_FLAG_IN_ORDER;

        OV_ZE_EXPECT(zeCommandQueueCreate(_engine.get_context(), _engine.get_device(), &command_queue_desc, &m_command_queue));
        OV_ZE_EXPECT(zeCommandListCreate(_engine.get_context(), _engine.get_device(), &command_list_desc, &m_command_list));
    } else {
        OV_ZE_EXPECT(zeCommandListCreateImmediate(_engine.get_context(), _engine.get_device(), &command_queue_desc, &m_command_list));
    }
    bool use_counter_based_events = m_queue_type == QueueTypes::in_order && info.supports_counter_based_events;
    m_user_ev_factory = std::make_shared<ze_event_factory>(engine, config.get_enable_profiling());
    if (use_counter_based_events) {
        m_ev_factory = std::make_shared<ze_counter_based_event_factory>(engine, config.get_enable_profiling());
    } else {
        // If counter based events are not supported or not used, use the same factory for both user and base events
        m_ev_factory = m_user_ev_factory;
    }
    GPU_DEBUG_INFO << "[GPU] Created L0 stream ("
        << "index=" << index
        << ", regular_cmd_queue=" << m_use_regular_cmd_queue
        << ", use_cp_offload=" << use_cp_offload
        << ", use_counter_based_events=" << use_counter_based_events
        << ")" << std::endl;
}

ze_stream::~ze_stream() {
#ifdef ENABLE_ONEDNN_FOR_GPU
    // Destroy OneDNN stream before destroying command list
    _onednn_stream.reset();
#endif
    if (m_command_list != nullptr)
        OV_ZE_WARN(zeCommandListDestroy(m_command_list));
    if (m_onednn_command_list != nullptr)
        OV_ZE_WARN(zeCommandListDestroy(m_onednn_command_list));
    if (m_command_queue != nullptr)
        OV_ZE_WARN(zeCommandQueueDestroy(m_command_queue));
    if (m_onednn_command_queue != nullptr)
        OV_ZE_WARN(zeCommandQueueDestroy(m_onednn_command_queue));
}

void ze_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    auto& ze_kernel = downcast<ze::ze_kernel>(kernel);
    auto kern = ze_kernel.get_kernel_handle();
    set_arguments_impl(kern, args_desc.arguments, args);
}

event::ptr ze_stream::enqueue_kernel(kernel& kernel,
                                     const kernel_arguments_desc& args_desc,
                                     const kernel_arguments_data& /* args */,
                                     std::vector<event::ptr> const& deps,
                                     bool is_output) {
    auto& ze_kernel = downcast<ze::ze_kernel>(kernel);

    auto kern = ze_kernel.get_kernel_handle();

    // If command list was submitted but not yet reset, sync and reset before appending new work
    ensure_cmd_list_ready();

    std::vector<ze_event_handle_t> dep_events;
    std::vector<ze_event_handle_t>* dep_events_ptr = nullptr;
    if (m_sync_method == SyncMethods::events) {
        for (auto& dep : deps) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get_handle() != nullptr)
                    dep_events.push_back(ze_base_ev->get_handle());
            }
        }
        dep_events_ptr = &dep_events;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
    }
    bool set_output_event = m_sync_method == SyncMethods::events || is_output;

    auto ev = set_output_event ? create_base_event() : std::make_shared<ze_empty_event>(++m_queue_counter);
    auto global = to_group_count(args_desc.workGroups.global);
    auto local = to_group_count(args_desc.workGroups.local);
    ze_group_count_t args = { global.groupCountX / local.groupCountX, global.groupCountY / local.groupCountY, global.groupCountZ / local.groupCountZ };
    OV_ZE_EXPECT(zeKernelSetGroupSize(kern, local.groupCountX, local.groupCountY, local.groupCountZ));
    OV_ZE_EXPECT(zeCommandListAppendLaunchKernel(m_command_list,
                                             kern,
                                             &args,
                                             set_output_event ? std::dynamic_pointer_cast<ze_base_event>(ev)->get_handle() : nullptr,
                                             dep_events_ptr == nullptr ? 0 : static_cast<uint32_t>(dep_events_ptr->size()),
                                             dep_events_ptr == nullptr ? 0 : &dep_events_ptr->front()));
    if (m_use_regular_cmd_queue) {
        m_regular_has_pending_cmds = true;
        if (m_sync_method == SyncMethods::events || is_output) {
            flush();
        }
    }

    return ev;
}

void ze_stream::enqueue_barrier() {
    ensure_cmd_list_ready();
    OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    if (m_use_regular_cmd_queue) {
        m_regular_has_pending_cmds = true;
    }
}

event::ptr ze_stream::enqueue_marker(std::vector<ze_event::ptr> const& deps, bool is_output) {
    ensure_cmd_list_ready();
    if (deps.empty()) {
        auto ev = create_base_event();
        OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, std::dynamic_pointer_cast<ze_base_event>(ev)->get_handle(), 0, nullptr));
        return ev;
    }

    if (m_sync_method  == SyncMethods::events) {
        std::vector<ze_event_handle_t> dep_events;
        for (auto& dep : deps) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get_handle() != nullptr)
                    dep_events.push_back(ze_base_ev->get_handle());
            }
        }
        if (dep_events.empty())
            return create_user_event(true);

        auto ev = create_base_event();
        OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list,
                                            std::dynamic_pointer_cast<ze_base_event>(ev)->get_handle(),
                                            static_cast<uint32_t>(dep_events.size()),
                                            &dep_events.front()));
        if (m_use_regular_cmd_queue) {
            m_regular_has_pending_cmds = true;
            if (is_output) {
                flush();
            }
        }
        return ev;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
        assert(m_last_barrier_ev != nullptr);
        return m_last_barrier_ev;
    } else {
        return create_user_event(true);
    }
}

ze_event::ptr ze_stream::group_events(std::vector<ze_event::ptr> const& deps) {
    return std::make_shared<ze_events>(deps, _engine);
}

void ze_stream::wait() {
    finish();
}

event::ptr ze_stream::create_user_event(bool set) {
    auto ev = m_user_ev_factory->create_event(++m_queue_counter);
    if (set)
        ev->set();

    return ev;
}

event::ptr ze_stream::create_base_event() {
    return m_ev_factory->create_event(++m_queue_counter);
}

std::unique_ptr<surfaces_lock> ze_stream::create_surfaces_lock(const std::vector<memory::ptr> &mem) const {
    // Level Zero engine currently does not support surfaces lock
    return nullptr;
}

void ze_stream::flush() const {
    if (!m_use_regular_cmd_queue) {
        return;
    }

    regular_poc_log("ze_stream::flush begin pending=" + std::to_string(m_regular_has_pending_cmds) + " submitted=" + std::to_string(m_regular_has_submitted));

    if (m_regular_has_pending_cmds) {
        if (!m_regular_list_is_closed) {
            OV_ZE_EXPECT(zeCommandListClose(m_command_list));
            m_regular_list_is_closed = true;
        }

        ze_command_list_handle_t cmd_lists[] = {m_command_list};
        OV_ZE_EXPECT(zeCommandQueueExecuteCommandLists(m_command_queue, 1, cmd_lists, nullptr));
        // Note: no synchronize here — submit only. finish() will sync + reset.
        m_regular_has_submitted = true;
        m_regular_has_pending_cmds = false;
    }

    regular_poc_log("ze_stream::flush end");
}

void ze_stream::finish() const {
    if (m_use_regular_cmd_queue) {
        flush();
        OV_ZE_EXPECT(zeCommandQueueSynchronize(m_command_queue, endless_wait));
        if (m_regular_has_submitted) {
            OV_ZE_EXPECT(zeCommandListReset(m_command_list));
            m_regular_list_is_closed = false;
            m_regular_has_submitted = false;
        }
        return;
    }

    OV_ZE_EXPECT(zeCommandListHostSynchronize(m_command_list, endless_wait));
}

void ze_stream::mark_onednn_pending() const {
    if (m_use_regular_cmd_queue) {
        m_regular_has_pending_cmds = true;
        regular_poc_log("mark_onednn_pending: set pending=true");
    }
}

void ze_stream::ensure_cmd_list_ready() const {
    regular_poc_log("ensure_cmd_list_ready: submitted=" + std::to_string(m_regular_has_submitted));
    if (m_use_regular_cmd_queue && m_regular_has_submitted) {
        OV_ZE_EXPECT(zeCommandQueueSynchronize(m_command_queue, endless_wait));
        OV_ZE_EXPECT(zeCommandListReset(m_command_list));
        m_regular_list_is_closed = false;
        m_regular_has_submitted = false;
    }
}

void ze_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (m_use_regular_cmd_queue) {
        flush();

        bool needs_sync = false;
        for (auto& ev : events) {
            auto* ze_base_ev = dynamic_cast<ze_base_event*>(ev.get());
            if (ze_base_ev == nullptr || ze_base_ev->get_handle() == nullptr) {
                needs_sync = true;
            }
        }

        if (needs_sync) {
            finish();
        }

        return;
    }

    bool needs_sync = false;
    for (auto& ev : events) {
        auto* ze_base_ev = dynamic_cast<ze_base_event*>(ev.get());
        if (ze_base_ev->get_handle() != nullptr) {
            ze_base_ev->wait();
        } else {
            needs_sync = true;
        }
        // Block thread and wait for event signal
        ev->wait();
    }

    if (needs_sync) {
        finish();
    }
}

void ze_stream::sync_events(std::vector<event::ptr> const& deps, bool is_output) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* ze_base_ev = dynamic_cast<ze_base_event*>(dep.get());
        assert(ze_base_ev != nullptr);
        if (ze_base_ev->get_queue_stamp() > m_last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        if (is_output) {
            m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_base_event());
            m_last_barrier_ev->set_queue_stamp(m_queue_counter.load());
            OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, m_last_barrier_ev->get_handle(), 0, nullptr));
        } else {
            OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
        }
        m_last_barrier = ++m_queue_counter;
    }

    if (!m_last_barrier_ev) {
        m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_user_event(true));
        m_last_barrier_ev->set_queue_stamp(m_queue_counter.load());
    }
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& ze_stream::get_onednn_stream() {
    OPENVINO_ASSERT(m_queue_type == QueueTypes::in_order, "[GPU] Can't create onednn stream handle as onednn doesn't support out-of-order queue");
    OPENVINO_ASSERT(_engine.get_device_info().vendor_id == INTEL_VENDOR_ID, "[GPU] Can't create onednn stream handle as for non-Intel devices");
    if (!_onednn_stream) {
        if (m_use_regular_cmd_queue) {
            regular_poc_log("ze_stream::get_onednn_stream create shared regular oneDNN stream");
            OPENVINO_ASSERT(m_command_queue != nullptr, "[GPU] regular command queue is not initialized");
            OPENVINO_ASSERT(m_command_list != nullptr, "[GPU] regular command list is not initialized");
            _onednn_stream = std::make_shared<dnnl::stream>(dnnl::ze_interop::make_stream(_engine.get_onednn_engine(),
                                                                                           m_command_queue,
                                                                                           m_command_list,
                                                                                           m_ev_factory->is_profiling_enabled()));
        } else {
            _onednn_stream = std::make_shared<dnnl::stream>(dnnl::ze_interop::make_stream(_engine.get_onednn_engine(),
                                                                                           m_command_list,
                                                                                           m_ev_factory->is_profiling_enabled()));
        }
    }

    return *_onednn_stream;
}
#endif

}  // namespace ze
}  // namespace cldnn
