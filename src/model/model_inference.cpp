#include <iostream>
#include <chrono>
#include <ncnn/benchmark.h>

#include "model/model_inference.h"

namespace DigitalHuman {
namespace Model {
struct ModelInference::Impl {


    
    ncnn::Net* net_ptr = nullptr;
    InferenceConfig config;
    float last_latency = 0.0f;

    // 核心推理逻辑
    int infer_internal(const ncnn::Mat& audio, const ncnn::Mat& face, ncnn::Mat& out) {
        // 首先指针校验
        if(!net_ptr) {
            std::cerr << "[Inference] Error: Model not bound!" << std::endl;
            return -1;
        }

        // 创建提取器
        ncnn::Extractor ex = net_ptr->create_extractor();

        // 应用配置
        net_ptr->opt.num_threads = config.num_threads;
        net_ptr->opt.use_vulkan_compute = config.use_vulkan;
        net_ptr->opt.use_fp16_packed = config.use_fp16;
        net_ptr->opt.use_fp16_storage = config.use_fp16;

        net_ptr->opt.lightmode = config.light_model;

        // 绑定输入
        // in0: Audio [1, 1, 80, 16]
        // in1: Face  [1, 6, 96, 96]
        int ret = 0;
        ret = ex.input("in0", audio);
        if (ret != 0) {
            std::cerr << "[Inference] Error setting input 'in0' (code " << ret << ")" << std::endl;
            return -1;
        }

        ret = ex.input("in1", face);
        if (ret != 0) {
            std::cerr << "[Inference] Error setting input 'in1' (code " << ret << ")" << std::endl;
            return -1;
        }

        // 执行计算并提取输出
        // out0: Generated Face [1, 3, 96, 96]
        auto start = std::chrono::high_resolution_clock::now();
        ret = ex.extract("out0", out);

        auto end = std::chrono::high_resolution_clock::now();
        last_latency = std::chrono::duration<float, std::milli>(end - start).count();
        if (ret != 0) {
            std::cerr << "[Inference] Error extracting 'out0' (code " << ret << ")" << std::endl;
            return -1;
        }

        // 结果校验
        if (out.empty()) {
            std::cerr << "[Inference] Error: Output tensor is empty!" << std::endl;
            return -1;
        }

        // 维度校验
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
}

int ModelInference::infer(const ncnn::Mat& audio_tensor,
                          const ncnn::Mat& face_tensor,
                          ncnn::Mat& out_tensor) {
    return pImpl->infer_internal(audio_tensor, face_tensor, out_tensor);
}

float ModelInference::getLastLatency() const {
    return pImpl->last_latency;
}


} // namespace Model
} // namespace DigitalHuman
