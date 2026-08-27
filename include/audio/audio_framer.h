#pragma once

#include "audio_loader.h"
#include <memory>
#include <vector>

namespace DigitalHuman {
namespace Audio {

/**
 * @brief 窗函数类型
 */

enum class WindowType {
    None,       // 不使用窗函数，相当于矩形窗
    Hamming,    // 汉明窗
    Hanning     // 汉宁窗
};

class AudioFramer {
public:

    /**
     * @brief 构造函数
     * @param sample_rate 采样率，默认 16000 Hz
     * @param frame_duration_ms 帧长，单位 ms，默认 50.0 ms
     * @param stride_duration_ms 帧移，单位 ms，默认 12.5 ms
     * @param win_type 窗函数类型，默认使用汉明窗 Hamming
     */
    AudioFramer(int sample_rate = 16000,
                double frame_duration_ms = 50.0,
                double stride_duration_ms = 12.5,
                WindowType win_type = WindowType::Hamming);
    
    ~AudioFramer();

    AudioFramer(AudioFramer&&) noexcept;
    AudioFramer& operator=(AudioFramer&&) noexcept;

    // 禁止拷贝
    AudioFramer(const AudioFramer&) = delete;
    AudioFramer& operator=(const AudioFramer&) = delete;

    /**
     * @brief 批量分帧处理
     * @param pcm_data 原始 PCM 数据，通常为单声道 float PCM，范围约为 [-1.0, 1.0] 
     * @param pad_tail 是否对尾部不足一帧的数据进行补零，默认 true
     * @return std::vector<std::vector<float>>
     *         返回分帧后的二维数组，格式为：
     *         [num_frames, frame_size]
     */
    std::vector<std::vector<float>> process(const std::vector<float>& pcm_data,
                                            bool pad_tail = true) const;
    
    /**
     * @brief 从指定起始位置创建一帧原始音频帧
     * @param pcm_data 原始 PCM 数据，通常为单声道 float PCM
     * @param start_index 当前帧在 pcm_data 中的起始采样点下标
     * @param pad_tail 当剩余采样点不足一帧时，是否补零
     * @return std::vector<float>
     *         返回一帧长度为 frame_size 的音频数据
     */
    std::vector<float> createFrame(const std::vector<float>& pcm_data,
                                   size_t start_index,
                                   bool pad_tail) const;

    /**
     * @brief 对单帧音频原地应用窗函数
     * @param frame 输入/输出音频帧，长度通常应等于 frame_size
     */
    void applyWindow(std::vector<float>& frame) const;

    /**
     * @brief 获取当前配置的帧大小
     * @return int 当前帧包含的采样点数
     */
    int getFrameSize() const;

    /**
     * @brief 获取当前配置的帧移大小
     * @return int 相邻两帧之间的采样点间隔
     */
    int getStrideSize() const;

    /**
     * @brief 获取当前配置的采样率
     * @return int 当前采样率，单位 Hz
     */
    int getSampleRate() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;

};

} // namespace Audio

} // namespace DigitalHuman