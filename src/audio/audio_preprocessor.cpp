#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "audio/audio_preprocessor.h"

namespace DigitalHuman {
namespace Audio {


struct AudioPreprocessor::Impl {

    // 依次实现4个接口函数
    // 1. 音量归一化
    void normalize(std::vector<float>& pcm_data, float target_peak) {
        if (pcm_data.empty()) {
            return;
        }

        // 寻找全局最大绝对值
        float max_val = 0.0f;
        for (float sample : pcm_data) {
            float abs_val = std::abs(sample);
            if (abs_val > max_val) {
                max_val = abs_val;
            }
        }

        // 计算增益
        const float epsilon = 1e-6f;
        float gain = target_peak / (max_val + epsilon);

        // 安全截断，如果底噪极小(如 0.0001)，gain 会变得巨大(10000)，会放大噪音
        // 设置最大增益上限为 20.0 (26dB)
        if (gain > 20.0f) {
            gain = 20.0f;
        }

        if (gain < 0.0f) {
            gain = 1.0f;
        }

        // 应用增益，如果 gain 接近1，无需处理
        if (std::abs(gain - 1.0f) > 1e-3f) {
            for (auto& sample : pcm_data) {
                sample *= gain;
            }
        }
    }

    // 2.预加重，这里主要根据公式来处理
    // 这里需要注意的是，我们必须要倒序遍历，否则前面的值会被覆盖，计算会出错
    void preEmphasize(std::vector<float>& pcm_data, float alpha) {
        if (pcm_data.empty()) {
            return;
        }

        for (size_t i = pcm_data.size() - 1; i > 0; --i) {
            pcm_data[i] = pcm_data[i] - alpha * pcm_data[i-1];
        }

        // 单独处理第一个点
        pcm_data[0] = pcm_data[0] - alpha * 0.0f;
    }

    // 3.降噪
    void denoise(std::vector<float>& pcm_data, float threshold_db) {
        // 首先需要 dB 转幅度 10 ^ (db / 20)
        float threshold_amp = std::pow(10.0f, threshold_db / 20.0f);

        for (auto& sample : pcm_data) {
            if (std::abs(sample) < threshold_amp) {
                // 低于阈值直接置零
                sample = 0.0f;
            }
        }
    }

    // 4. VAD
    std::vector<SpeechSegment> detectSpeech(const std::vector<float>& pcm_data, int sample_rate) {
        std::vector<SpeechSegment> segments;
        if (pcm_data.empty())  {
            return segments;
        }

        // VAD 参数
        const int frame_ms = 20; // 20ms 一帧
        const int frame_size = sample_rate * frame_ms / 1000;
        const size_t total_samples = pcm_data.size();
        
        // 动态阈值计算
        // 计算全段平均能量 ，绝对值平均
        double total_energy = 0;
        for (float s : pcm_data) total_energy += std::abs(s);
        float avg_energy = total_energy / total_samples;
        
        // 阈值设为平均能量的 20% 
        // 设置一个最低门限，防止在纯静音环境下把底噪当语音
        const float min_threshold = 0.01f; 
        float threshold = std::max(avg_energy * 0.5f, min_threshold);

        // 状态机变量
        bool is_speech = false;
        int hangover_counter = 0;
        const int HANGOVER_FRAMES = 10; // 保持 10 帧 (200ms)
        
        size_t current_start = 0;

        for (size_t i = 0; i < total_samples; i += frame_size) {
            // 计算当前帧的 RMS 能量
            float sum_sq = 0.0f;
            size_t count = 0;
            for (size_t j = 0; j < frame_size && (i + j) < total_samples; ++j) {
                float val = pcm_data[i + j];
                sum_sq += val * val;
                count++;
            }
            if (count == 0) {
                break;
            }
            
            float rms = std::sqrt(sum_sq / count);

            // 状态机逻辑
            if (rms > threshold) {
                // 能量超过阈值 -> 语音
                if (!is_speech) {
                    is_speech = true;
                    current_start = i; // 记录开始点
                }
                hangover_counter = HANGOVER_FRAMES; // 重置挂起计数器
            } else {
                // 能量低于阈值 -> 可能是静音，也可能是气口
                if (is_speech) {
                    if (hangover_counter > 0) {
                        hangover_counter--; // 还在挂起期，保持语音状态
                    } else {
                        // 挂起期结束，确认静音，记录结束点
                        is_speech = false;
                        segments.push_back({current_start, i});
                    }
                }
            }
        }

        // 处理最后一段
        if (is_speech) {
            segments.push_back({current_start, total_samples});
        }

        return segments;
    }

};

AudioPreprocessor::AudioPreprocessor() : pImpl(std::make_unique<Impl>()) {}
AudioPreprocessor::~AudioPreprocessor() = default;

AudioPreprocessor::AudioPreprocessor(AudioPreprocessor&&) noexcept = default;
AudioPreprocessor& AudioPreprocessor::operator=(AudioPreprocessor&&) noexcept = default;

void AudioPreprocessor::normalize(std::vector<float>& pcm_data, float target_peak) {
    pImpl->normalize(pcm_data, target_peak);
}

void AudioPreprocessor::preEmphasize(std::vector<float>& pcm_data, float alpha) {
    pImpl->preEmphasize(pcm_data, alpha);
}

void AudioPreprocessor::denoise(std::vector<float>& pcm_data, float threshold_db) {
    pImpl->denoise(pcm_data, threshold_db);
}

std::vector<SpeechSegment> AudioPreprocessor::detectSpeech(const std::vector<float>& pcm_data, int sample_rate) {
    return pImpl->detectSpeech(pcm_data, sample_rate);
}


} // namespace Audio
} // namespace DigitalHuman