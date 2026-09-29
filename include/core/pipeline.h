#pragma once

#include <memory>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include "core/pipeline_config.h"

namespace DigitalHuman {
namespace Core {

struct InferenceTask {
    double pts_ms = 0.0;
    std::vector<float> audio_feature; 
    cv::Mat base_face;                
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    Pipeline(Pipeline&&) noexcept;
    Pipeline& operator=(Pipeline&&) noexcept;

    /**
     * @brief 启动流水线
     * @param model_dir 模型路径
     * @param headless 是否无界面
     * @param output_video_path 如果指定路径，则进入离线视频导出模式
     */
    bool start(const std::string& model_dir, bool headless = false, const std::string& output_video_path = "");

    void stop();

    bool pushTask(const InferenceTask& task);

    void pushAudioData(const std::vector<float>& pcm_data);

    bool isHealthy() const;

    // 获取队列当前积压量，用于外部判断后台渲染是否结束
    size_t getTaskQueueSize() const;
    size_t getRenderQueueSize() const;

    // 对静态图片，只检测一次人脸，后续缓存复用就行
    void setStaticFaceCacheEnabled(bool enabled);

    // 实时预览模式
    void setRealtimePreviewMode(bool enabled);

    // 设置使用AudioPlayer
    void setUseAudioPlayerClock(bool enbaled);

    void setConfig(const PipelineConfig& config);
    PipelineConfig getConfig() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} 
}
