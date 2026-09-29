#include <iostream>
#include <chrono>
#include <thread>
#include <filesystem>
#include <ncnn/net.h>

#include "model/model_loader.h"

namespace fs = std::filesystem;

namespace DigitalHuman {
namespace Model {

struct ModelLoader::Impl {
    ncnn::Net net;
    bool is_loaded = false;

    // 用于异步加载的线程
    std::thread loading_thread;

    // 模型预热
    void perform_warmup() {
        if (!is_loaded) {
            return;
        }

        // 构造 Wav2Lip 的输入 尺寸为 96 x 96
        // 1. Audio: [Batch=1, Channel=1, Mel=80, Time=16] -> ncnn: w=16, h=80, c=1
        ncnn::Mat audio_in(16, 80, 1);
        audio_in.fill(0.0f);

        // 2. Face: [Batch=1, Channel=6, H=96, W=96] -> ncnn: w=96, h=96, c=6
        ncnn::Mat face_in(96, 96, 6);
        face_in.fill(0.0f);

        // 做一次推理
        ncnn::Extractor ex = net.create_extractor();
        // Match production inference. Retaining every intermediate activation
        // during warm-up can exceed the memory budget on CPU-only devices.
        ex.set_light_mode(true);

        // PNNX-exported Wav2Lip mapping: in0=mel, in1=face.
        ex.input("in0", audio_in);
        ex.input("in1", face_in);

        ncnn::Mat out;
        int ret = ex.extract("out0", out);

        if (ret != 0 || out.empty()) {
            std::cerr << "[ModelLoader] Warmup failed for Wav2Lip out0 output." << std::endl;
        }
    }

    // 加载逻辑
    bool load_internal(const std::string& param_path, bool use_gpu, float& cost_ms) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // 检查文件有效性
        if (!fs::exists(param_path)) {
            std::cerr << "[ModelLoader] Error: Param file not found: " << param_path << std::endl;
            return false;
        }

        std::string bin_path = param_path;
        if (bin_path.find(".param") != std::string::npos) {
            bin_path.replace(bin_path.find(".param"), 6, ".bin");
        } else {
            bin_path += ".bin";
        }

        if (!fs::exists(bin_path)) {
            std::cerr << "[ModelLoader] Error: Bin file not found: " << bin_path << std::endl;
            return false;
        }

        // 配置 ncnn 选项
        net.opt.use_vulkan_compute = use_gpu;
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.num_threads = 4;

        // 加载前先情况，防止多次 load 内存泄漏
        net.clear();


        // 开始加载
        int ret_p = net.load_param(param_path.c_str());
        int ret_b = net.load_model(bin_path.c_str());
        // 有效性检查
        if (ret_p != 0 || ret_b != 0) {
            std::cerr << "[ModelLoader] Error: ncnn load failed (ret_p=" << ret_p << ", ret_b=" << ret_b << ")" << std::endl;
            return false;
        }

        is_loaded = true;

        // 预热
        perform_warmup();

        auto end_time = std::chrono::high_resolution_clock::now();
        cost_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

        return true;
    }
};

ModelLoader::ModelLoader() : pImpl(std::make_unique<Impl>()){}

ModelLoader::~ModelLoader() {
    if (pImpl->loading_thread.joinable()) {
        pImpl->loading_thread.join();
    }

    pImpl->net.clear();
}

ModelLoader::ModelLoader(ModelLoader&&) noexcept = default;
ModelLoader& ModelLoader::operator=(ModelLoader&&) noexcept = default;

// 模型加载函数
bool ModelLoader::load(const std::string& model_path, bool use_gpu) {
    float cost = 0;
    bool success = pImpl->load_internal(model_path, use_gpu, cost);
    if (success) {
        std::cout << "[ModelLoader] Sync Load Success. Cost: " << cost << " ms" << std::endl;
    }

    return success;
}

// 异步加载
void ModelLoader::loadAsync(const std::string& model_path, LoadCallback callback) {
    // 如果之前还有线程在跑，先回收
    if (pImpl->loading_thread.joinable()) {
        pImpl->loading_thread.join();
    }

    // 启动新线程
    pImpl->loading_thread = std::thread([this, model_path, callback]() {
        float cost = 0;
        bool success = this->pImpl->load_internal(model_path, false, cost);

        if (success) {
            std::cout << "[ModelLoader] Async Load Success. Cost: " << cost << " ms" << std::endl;
            if (callback) {
                callback(&this->pImpl->net, cost);
            } 
        } else {
            callback(nullptr, 0);
        }
    });
}

ncnn::Net* ModelLoader::getNet() {
    if (!pImpl->is_loaded) {
        return nullptr;
    }

    return &pImpl->net;
}

bool ModelLoader::isLoaded() const {
    return pImpl->is_loaded;
}

} // namespace Model
} // namespace DigitalHuman
