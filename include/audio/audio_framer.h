#pragma once

#include <vector>
#include <string>
#include <memory>

namespace DigitalHuman {
namespace Audio {

/***
 * @brief 窗函数类型
 */
enum class WindowType {
    None,       // 矩形窗，不处理
    Hamming,    // 汉明窗
    Hanning     // 汉宁窗
};


/***
 * @brief 音频分帧器，负责将连续的PCM信号切分为重叠的帧，并应用窗函数
 */
class AudioFramer {
public:

    /**
     * @brief 构造函数
     * @param sample_rate 采样率，默认16000
     * @param frame_duration_ms 帧长，默认25ms
     * @param stride_duration_ms 帧移，默认10ms
     * @param win_tyep 窗函数类型，默认是汉明窗 Hamming
     */
    AudioFramer(int sample_rate = 16000,
                int frame_duration_ms = 25,
                int stride_duration_ms = 10,
                WindowType win_type = WindowType::Hamming);
    
    ~AudioFramer();

    AudioFramer(AudioFramer&&) noexcept;
    AudioFramer& operator=(AudioFramer&&) noexcept;
    AudioFramer(const AudioFramer&);
    AudioFramer& operator=(const AudioFramer&);

    /**
     * @brief 分帧处理
     * @param pcm_data 原始PCM数据，单声道Float
     * @return std::vector<std::vector<float>> 分针后的二维数组 [num_frames, frame_size]
     */
    std::vector<std::vector<float>>process(const std::vector<float>& pcm_data);

    /***
     * @brief 获取当前配置的帧大小，采样点数
     * @return int 当前采样点数
     */
    int getFrameSize() const;

    /***
     * @brief 获取当前配置的步长
     * @return int 当前采样点数
     */
    int getStrideSize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};



} // namespace Audio
} // namespace DigitalHuman