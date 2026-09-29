#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Audio {

/**
 * @brief 音频实时处理线程模块
 * 负责从原始连续的 PCM 流中滑动提取定长帧，应用窗函数，计算 Mel 频谱，并推送到推理队列
 * 这里做了解耦设计，纯粹输出音频特征，推理任务的组装和队列转发靠别的线程实现
 */
class AudioProcessor {
public:
    // 这里定义输出特征的回调函数类型
    using FeatureCallback = std::function<void(double pts_ms, std::vector<float>mel_feature)>;

    /**
     * @brief 构造函数
     * @param target_queue 目标推理队列的引用
     * @param sample_rate 采样率，默认是16000
     */
    explicit AudioProcessor(FeatureCallback output_cb,
                            int sample_rate = 16000);
    
    ~AudioProcessor();

    AudioProcessor(const AudioProcessor&) = delete;
    AudioProcessor& operator=(const AudioProcessor&) = delete;

    /**
     * @brief 启动后台处理线程
     */
    bool start();

    /**
     * @brief 停止并销毁后台线程
     */
    void stop();

    /**
     * @brief 可以让其他线程推入原始连续音频数据
     * @param pcm_data 连续的 pcm 采样数据
     * @param start_pts_ms 该块音频其实的基准时间戳
     */
    void pushRawAudio(const std::vector<float>& pcm_data, double start_pts_ms);
    
    /**
     * @brief 离线模式判断音频特征是否彻底吐完
     */
    bool isDrained() const;

    /**
     * @brief 通知音频输入已经全部推送完成
     * 离线模式下调用，用于触发尾部 Mel padding，补齐最后几帧视频。
     */
    void markInputFinished();

    /**
     * @brief 设置实时模式
     * @param enbaled
     *              false : 离线模式，不会丢弃任何一帧音频，保证完整生成
     *              true:   实时模式，队列过长时允许丢弃旧音频，避免延迟
     */
    void setRealtimeMode(bool enabled);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};


}
}
