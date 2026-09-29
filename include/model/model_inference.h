#pragma once

#include <vector>
#include <memory>
#include <string>
#include <cstdint>
#include <ncnn/net.h>

namespace DigitalHuman {
namespace Model {

/**
 * @brief 推理配置参数
 */
struct InferenceConfig {
    int num_threads = 4;         // 线程数（匹配 4 vCPU 环境）
    bool use_vulkan = false;     // 是否使用 GPU ，需编译时开启 Vulkan
    bool use_fp16 = false;        // 是否使用半精度 (加速)
    bool light_mode = true;      // 开启省内存模式
};

struct LatencyStats {
    std::uint64_t sample_count = 0;
    double total_ms = 0.0;
    float min_ms = 0.0f;
    float max_ms = 0.0f;

    float averageMs() const {
        return sample_count == 0 ? 0.0f
                                 : static_cast<float>(total_ms / sample_count);
    }
};

/**
 * @brief 模型推理引擎，主要封装 ncnn::Extractor 的生命周期和调用逻辑
 */
class ModelInference {
public:
    ModelInference();
    ~ModelInference();

    ModelInference(const ModelInference&) = delete;
    ModelInference& operator=(const ModelInference&) = delete;

    ModelInference(ModelInference&&) noexcept;
    ModelInference& operator=(ModelInference&&) noexcept;

    /**
     * @brief 绑定模型
     * @param net 加载好的 ncnn::Net 指针
     */
    void bindModel(ncnn::Net* net);

    /**
     * @brief 更新推理配置
     */
    void setConfig(const InferenceConfig& config);

    /**
     * @brief 执行单帧推理
     * @param audio_tensor 音频输入张量 [1, 1, 80, 16]
     * @param face_tensor  人脸输入张量 [1, 6, 96, 96]
     * @param out_tensor   输出结果张量 [1, 3, 96, 96]
     * @return 0: 成功, -1: 失败
     */
    int infer(const ncnn::Mat& audio_tensor,
              const ncnn::Mat& face_tensor,
              ncnn::Mat& out_tensor);
    
    /**
     * @brief 获取最后一次推理的耗时 (ms)
     */
    float getLastLatency() const;
    LatencyStats getLatencyStats() const;
    void resetLatencyStats();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Model
} // namespace DigitalHuman
