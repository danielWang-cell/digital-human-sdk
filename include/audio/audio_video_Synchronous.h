#pragma once

#include <atomic>
#include <vector>
#include <memory>

namespace DigitalHuman {
namespace Audio {

/**
 * @brief 音频播放状态
 */
enum class PlayState {
    Stopped,        // 停止播放
    Playing,        // 正在播放
    Paused          // 暂停播放
};

/**
 * @brief 基于 PortAudio 的核心音频播放器 
 * 主要作用是：
 * 1. 驱动声卡硬件进行低延迟播放
 * 2. 作为系统的主时钟，提供高精度的时间戳供视频同步使用
 */
class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();

    AudioPlayer(const AudioPlayer&) = delete;
    AudioPlayer& operator=(const AudioPlayer&) = delete;
    AudioPlayer(AudioPlayer&&) noexcept;
    AudioPlayer& operator=(AudioPlayer&&) noexcept;

    /**
     * @brief 初始化音频设备
     * @param sample_rate 采样率 ，默认 16000
     * @param channels 声道数 ，默认 1 个通道
     * @param frames_per_buffer 每次回调处理的帧数 ，默认 512
     * @return 是否初始化成功
     */
    bool open(int sample_rate = 16000, int channels = 1, int frames_per_buffer = 512);

    /**
     * @brief 写入待播放的 PCM 数据 (生产者调用)
     * @param pcm_data 格式须为 float32
     */
    void pushData(const std::vector<float>& pcm_data);

    /**
     * @brief 播放控制 
     */
    void play();
    void pause();
    void stop();

    /**
     * @brief 获取当前音频精确播放时间 (毫秒)，视频渲染调度器将调用此接口获取基准时间
     * @return 当前播放时间 (ms)
     */
    double getCurrentTime() const;

    /**
     * @brief 获取当前播放状态 
     */
    PlayState getState() const;


    /**
     * @brief 获取缓冲区中剩余数据的时长（毫秒） 
     */
    double getBufferedDuration() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

}   // namespace Audio
}   // namespace DigitalHuman

