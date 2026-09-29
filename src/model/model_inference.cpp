#include <iostream>
#include <chrono>
#include <limits>
#include <algorithm>
#include <ncnn/benchmark.h>

#include "model/model_inference.h"

namespace DigitalHuman {
namespace Model {

struct ModelInference::Impl {
    ncnn::Net* net_ptr = nullptr;
    InferenceConfig config;
    float last_latency = 0.0f;
    LatencyStats latency_stats;

    void applyConfig() {
        if (!net_ptr) {
            return;
        }

        net_ptr->opt.num_threads = config.num_threads;
        net_ptr->opt.use_vulkan_compute = config.use_vulkan;
        net_ptr->opt.use_fp16_packed = config.use_fp16;
        net_ptr->opt.use_fp16_storage = config.use_fp16;
        net_ptr->opt.use_fp16_arithmetic = config.use_fp16;
        net_ptr->opt.lightmode = config.light_mode;
    }

    int infer_internal(const ncnn::Mat& audio, const ncnn::Mat& face, ncnn::Mat& out) {
        if (!net_ptr) {
            std::cerr << "[Inference] Error: Model not bound!" << std::endl;
            return -1;
        }

        if (audio.empty() || face.empty()) {
            std::cerr << "[Inference] Error: Empty input tensor." << std::endl;
            return -1;
        }

        // 先应用配置，再创建 extractor
        applyConfig();

        ncnn::Extractor ex = net_ptr->create_extractor();
        ex.set_light_mode(config.light_mode);

        // This PNNX-exported Wav2Lip model exposes mel as in0 and face as in1.
        int ret = ex.input("in0", audio);
        if (ret != 0) {
            std::cerr << "[Inference] Error setting input 'in0' (code " << ret << ")" << std::endl;
            return -1;
        }

        ret = ex.input("in1", face);
        if (ret != 0) {
            std::cerr << "[Inference] Error setting input 'in1' (code " << ret << ")" << std::endl;
            return -1;
        }

        auto start = std::chrono::high_resolution_clock::now();

        ret = ex.extract("out0", out);

        auto end = std::chrono::high_resolution_clock::now();
        last_latency = std::chrono::duration<float, std::milli>(end - start).count();
        ++latency_stats.sample_count;
        latency_stats.total_ms += last_latency;
        if (latency_stats.sample_count == 1) {
            latency_stats.min_ms = last_latency;
            latency_stats.max_ms = last_latency;
        } else {
            latency_stats.min_ms = std::min(latency_stats.min_ms, last_latency);
            latency_stats.max_ms = std::max(latency_stats.max_ms, last_latency);
        }

        if (ret != 0) {
            std::cerr << "[Inference] Error extracting output 'pred' (code " << ret << ")" << std::endl;
            return -1;
        }

        if (out.empty()) {
            std::cerr << "[Inference] Error: Output tensor is empty!" << std::endl;
            return -1;
        }

        if (out.w != 96 || out.h != 96 || out.c != 3) {
            std::cerr << "[Inference] Warning: Unexpected output shape "
                      << out.w << "x" << out.h << "x" << out.c << std::endl;
        }

        return 0;
    }
};

ModelInference::ModelInference() : pImpl(std::make_unique<Impl>()) {}
ModelInference::~ModelInference() = default;
ModelInference::ModelInference(ModelInference&&) noexcept = default;
ModelInference& ModelInference::operator=(ModelInference&&) noexcept = default;

void ModelInference::bindModel(ncnn::Net* net) {
    pImpl->net_ptr = net;
}

void ModelInference::setConfig(const InferenceConfig& config) {
    pImpl->config = config;
    pImpl->applyConfig();
}

int ModelInference::infer(const ncnn::Mat& audio_tensor,
                          const ncnn::Mat& face_tensor,
                          ncnn::Mat& out_tensor) {
    return pImpl->infer_internal(audio_tensor, face_tensor, out_tensor);
}

float ModelInference::getLastLatency() const {
    return pImpl->last_latency;
}

LatencyStats ModelInference::getLatencyStats() const {
    return pImpl->latency_stats;
}

void ModelInference::resetLatencyStats() {
    pImpl->latency_stats = LatencyStats{};
}

} // namespace Model
} // namespace DigitalHuman
