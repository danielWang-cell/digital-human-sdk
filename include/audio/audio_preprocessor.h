#pragma once

#include <cstddef>
#include <functional>
#include <vector>
#include <memory>
#include <string>

namespace DigitalHuman {
namespace Audio {

/**
 * @brief 语音片段结构体，记录语音在 PCM 数据中的起始和结束索引
 */

struct SpeechSegment {
    size_t start_idx; // 采样点开始索引
    size_t end_idx;   // 采样点结束索引
};

/**
 * @brief 音频预处理配置
 */
struct AudioPreprocessConfig {
    bool enable_normalize = false;   // 归一化
    bool enable_denoise = false;     // 降噪
    bool enable_pre_emphasis = true; // 预加重，这是音频处理的必要步骤

    // 相应阈值
    float normalize_peak = 0.95f;
    float denoise_threshold_db = -60.0f;
    float pre_emphasis_alpha = 0.97f;

    // 是否使用流式预加重
    bool streaming_pre_emphasis = true;
};

/**
 * @brief 音频预处理模块，负责音频的清洗，增强和分析
 */
class AudioPreprocessor {
public:
    AudioPreprocessor();
    ~AudioPreprocessor();

    AudioPreprocessor(const AudioPreprocessor&) = delete;
    AudioPreprocessor& operator=(const AudioPreprocessor&) = delete;

    AudioPreprocessor(AudioPreprocessor&&) noexcept;
    AudioPreprocessor& operator=(AudioPreprocessor&&) noexcept;

    /**
     * @brief 音量归一化，将音频的最大振幅调整到指定目标值，防止爆音或者音量过小
     * @param pcm_data 输入输出音频数据，原地修改
     * @param target_peak 目标峰值，(0.0 - 1.0)，默认0.95 
     */
    void normalize(std::vector<float>& pcm_data, float target_pead = 0.95f);

    /**
     * @brief 预加重 y[t] = x[t] - alpha * x[t - 1]
     * @param pcm_data 输入输出音频数据，原地修改
     * @param alpha 预加重系数，默认 0.97 
     */
    void preEmphasize(std::vector<float>& pcm_data, float alpha = 0.97f);

    /**
     * @brief 流式预加重，适合连续音频流
     * 与 preEmphasize 的区别：
     * - preEmphasize() 每帧第一个点使用 0 作为前一个点
     * - preEmphasizeStreaming() 每帧第一个点使用上一帧最后一个原始采样点
     * 这样帧与帧之间的预加重是连续的
     */
    void preEmphasizeStreaming(std::vector<float>& pcm_data, float alpha = 0.97f);

    /**
     * @brief 重置流式预加重状态
     * 每次开始处理一段新的音频或重新 start AudioProcessor 时应该调用
     */
    void resetStreamingState();

    /**
     * @brief 静音检测 VAD 基于短时能量检测语音活动区域
     * @param pcm_data 输入输出音频数据，原地修改
     * @param sample_rate 采样率（默认 16000）
     * @return std::vector<SpeechSegment> 有效的语音段列表
     */
    std::vector<SpeechSegment> detectSpeech(const std::vector<float>& pcm_data, int sample_rate = 16000);

    /**
     * @brief 简单降噪，将低于阈值的背景底噪直接置零（时域门限降噪）
     * @param pcm_data 输入输出音频数据
     * @param threshold_db 噪声门限（分贝），默认 -60dB*/
    void denoise(std::vector<float>& pcm_data, float threshold_db = -60.0f);

    /**
     * @brief 根据配置统一处理一帧音频
     */
    void processFrame(std::vector<float>& pcm_frame, const AudioPreprocessConfig& config);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Audio
} // namespace DigitalHuman