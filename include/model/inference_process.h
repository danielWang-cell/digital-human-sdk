#pragma once

#include <memory>
#include <atomic>
#include <string>

template <typename T> 
class ThreadSafeQueue;

namespace ncnn {
    class Net;
}

namespace DigitalHuman {

// 前向声明
namespace Core
{
    struct InferenceTask;
    template <typename T> class ThreadSafeQueue;
    class FrameScheduler;
} // namespace Core

namespace Model
{
    
/**
 * @brief 支持的推理模型枚举
 */
enum class InferenceModelType {
    Wav2Lip,
    MuseTalk 
};

/**
 * @brief 独立推理线程模块 
 * 负责从输入队列提取任务，执行深度学习模型前向传播，处理容错与重试，并将结果送往渲染调度器
 */
class InferenceProcessor {
public:
    InferenceProcessor(::ThreadSafeQueue<Core::InferenceTask>& input_queue,
                        Core::FrameScheduler& frame_scheduler);
    ~InferenceProcessor();

    InferenceProcessor(const InferenceProcessor&) = delete;
    InferenceProcessor& operator=(const InferenceProcessor&) = delete;

    /**
     * @brief 绑定并初始化 Wav2Lip引擎
     * @param net 已经加载到内存的 ncnn::Net 网络
     * @param use_gpu 是否开启 vulkan 加速
     */
    bool initWav2Lip(ncnn::Net* net, bool use_gpu = false);

    /**
     * @brief 绑定并初始化 MuseTalk 引擎
     * @param model_dir 包含 MuseTalk 模型的目录路径
     */
    bool initMuseTalk(const std::string& model_dir);

    /**
     * @brief 设置当前模型类型
     */
    void setModelType(InferenceModelType type);

    /**
     * @brief 设置是否启用人脸缓存
     * 静态图片 + 音频，在离线时使用
     * 在实时视频流时关闭
     */
    void setStaticFaceCacheEnabled(bool enabled);

    void setInferenceThreads(int threads);
    void setInferenceOptions(bool use_vulkan, bool use_fp16, bool light_mode);
    void setSharpening(bool enabled, double strength = 0.25);

    /**
     * @brief 启动后台推理线程
     */
    bool start();

    /**
     * @brief 停止后台线程
     */
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Model
} // namespace DigitalHuman
