// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

// clang-format off
#include "openvino/openvino.hpp"

#include "samples/args_helper.hpp"
#include "samples/common.hpp"
#include "samples/latency_metrics.hpp"
#include "samples/slog.hpp"
// clang-format on

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#endif

// Helper function to get the current Working Set in MB
static double get_current_working_set_mb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        // WorkingSetSize is in bytes, convert to MB
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

using Ms = std::chrono::duration<double, std::ratio<1, 1000>>;
long long MY_SLEEP = 0;
int main(int argc, char* argv[]) {
    std::cout << "app:main.cpp - sync_benchmark_app start main" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(MY_SLEEP));
    try {
        slog::info << "OpenVINO:" << slog::endl;
        slog::info << ov::get_openvino_version();

        std::string device_name = "CPU";
        if (argc == 3) {
            device_name = argv[2];
        } else if (argc != 2) {
            slog::info << "Usage : " << argv[0] << " <path_to_model> <device_name>(default: CPU)" << slog::endl;
            return EXIT_FAILURE;
        }
        // Optimize for latency. Most of the devices are configured for latency by default,
        // but there are exceptions like GNA
        ov::AnyMap latency{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}};

        // Create ov::Core and use it to compile a model.
        // Select the device by providing the name as the second parameter to CLI.
        // Using MULTI device is pointless in sync scenario
        // because only one instance of ov::InferRequest is used
        ov::Core core;
        std::cout << "app:main.cpp - create core " << get_current_working_set_mb() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(MY_SLEEP));
        ov::CompiledModel compiled_model = core.compile_model(argv[1], device_name, latency);
        std::cout << "app:main.cpp - core.compile_model" << get_current_working_set_mb() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(MY_SLEEP));
        ov::InferRequest ireq = compiled_model.create_infer_request();
        std::cout << "app:main.cpp - compiled_model.create_infer_request" << get_current_working_set_mb() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(MY_SLEEP));
        // Fill input data for the ireq
        for (const ov::Output<const ov::Node>& model_input : compiled_model.inputs()) {
            fill_tensor_random(ireq.get_tensor(model_input));
        }
        std::cout << "app:main.cpp - fill_tensor_random model_inputs" << get_current_working_set_mb() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(MY_SLEEP));
        // Warm up
        ireq.infer();
        std::cout << "app:main.cpp - ireq.infer() warm up" << get_current_working_set_mb() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(MY_SLEEP));
#if 0
        // Benchmark for seconds_to_run seconds and at least niter iterations
        std::chrono::seconds seconds_to_run{10};
        size_t niter = 10;
        std::vector<double> latencies;
        latencies.reserve(niter);
        auto start = std::chrono::steady_clock::now();
        auto time_point = start;
        auto time_point_to_finish = start + seconds_to_run;
        while (time_point < time_point_to_finish || latencies.size() < niter) {
            ireq.infer();
            auto iter_end = std::chrono::steady_clock::now();
            latencies.push_back(std::chrono::duration_cast<Ms>(iter_end - time_point).count());
            time_point = iter_end;
        }
        std::cout << "app:main.cpp - ireq.infer() END" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(MY_SLEEP));
        auto end = time_point;
        double duration = std::chrono::duration_cast<Ms>(end - start).count();
        // Report results
        slog::info << "Count:      " << latencies.size() << " iterations" << slog::endl
                   << "Duration:   " << duration << " ms" << slog::endl
                   << "Latency:" << slog::endl;
        size_t percent = 50;
        LatencyMetrics{latencies, "", percent}.write_to_slog();
        slog::info << "Throughput: " << double_to_string(latencies.size() * 1000 / duration) << " FPS" << slog::endl;
#endif
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
