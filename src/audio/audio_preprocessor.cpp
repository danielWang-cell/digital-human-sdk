#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "audio/audio_preprocessor.h"

namespace DigitalHuman {
namespace Audio {

struct AudioPreprocessor::Impl {
    float last_raw_sample = 0.0f;
    bool has_last_raw_sample = false;

    void resetStreamingState() {
        last_raw_sample = 0.0f;
        has_last_raw_sample = false;
    }

    // 依次实现 4 个接口函数
    // 1. 音量归一化
    void normalize(std::vector<float>& pcm_data, float target_peak) {
        if (pcm_data.empty()) {
            return;
        }

        if (target_peak <= 0.0f) {
            return;
        }

        target_peak = std::min(1.0f, target_peak);
        
        // 寻找全局最大绝对值
        float max_val = 0.0f;
        for (float sample : pcm_data) {
            max_val = std::max(max_val, std::abs(sample));
        }
        // 几乎全静音时不做归一化，避免把底噪放大
        constexpr float kSilenceEpsilon = 1e-6f;
        if (max_val < kSilenceEpsilon) {
            return;
        }

        float gain = target_peak / max_val;

        // 限制最大增益，防止弱噪声被过度放大
        if (gain > 20.0f) {
            gain = 20.0f;
        }

        if (gain < 0.0f) {
            gain = 1.0f;
        }

        // 应用增益，如果 gain 接近 1，无需处理
        if (gain - 1.0f > 1e-3f) {
            for (auto& sample : pcm_data) {
                sample *= gain;
                // 安全限制，避免异常值超过 [-1.0, 1.0] 太多
                sample = std::max(-1.0f, std::min(sample, 1.0f));
            }
        }
    }

    void preEmphasize(std::vector<float>& pcm_data, float alpha) {
        if (pcm_data.empty()) {
            return;
        }

        alpha = std::max(0.0f, std::min(1.0f, alpha));

        // 必须倒序遍历，如果正序遍历，pcm_data[i - 1] 会已经被修改，公式就错了
        for (size_t i = pcm_data.size() - 1; i > 0; --i) {
            pcm_data[i] = pcm_data[i] - alpha * pcm_data[i - 1];
        }
        
        // 独立音频段的第一个采样点没有前一个点，默认 x[-1] = 0，单独处理第一个点
        pcm_data[0] = pcm_data[0] - alpha * 0.0f;
    }

    void preEmphasizeStreaming(std::vector<float>& pcm_data, float alpha) {
        if (pcm_data.empty()) {
            return;
        }

        alpha = std::max(0.0f, std::min(1.0f, alpha));

        // 先保存当前帧最后一个“原始采样点”，因为后面会原地修改 pcm_data
        float current_last_raw_sample = pcm_data.back();

        // 倒序处理帧内采样点
        for (size_t i = pcm_data.size() - 1; i > 0; --i) {
            pcm_data[i] = pcm_data[i] - alpha * pcm_data[i - 1];
        }

        // 第一个点使用上一帧最后一个原始采样点
        float previous = has_last_raw_sample ? last_raw_sample : 0.0f;
        pcm_data[0] = pcm_data[0] - alpha * previous;

        last_raw_sample = current_last_raw_sample;
        has_last_raw_sample = true;
    }

    void denoise(std::vector<float>& pcm_data, float threshold_db) {
        if (pcm_data.empty()) {
            return;
        }

        // dB 转线性幅度
        // amplitude = 10 ^ (db / 20)
        float threshold_amp  = std::pow(10.0f, threshold_db / 20.0f);

        threshold_amp = std::max(0.0f, threshold_amp);

        for (float& sample : pcm_data) {
            if (std::abs(sample) < threshold_amp) {
                // 低于阈值直接置零
                sample = 0.0f;
            }
        }
    }

    std::vector<SpeechSegment> detectSpeech(const std::vector<float>& pcm_data, int sample_rate) {
        
        std::vector<SpeechSegment> segments;
        
        if (pcm_data.empty() || sample_rate <= 0) {
            return segments;
        }

        // VAD 参数
        const int frame_ms = 20;
        const int frame_size = sample_rate * frame_ms / 1000;
        const size_t total_samples = pcm_data.size();

        if (frame_size <= 0) {
            return segments;
        }

        double total_abs_energy = 0.0;
        for (float sample : pcm_data) {
            total_abs_energy += std::abs(sample);
        }

        float avg_energy = static_cast<float>(total_abs_energy / static_cast<double>(total_samples));

        // 动态阈值：
        // 太小会把底噪当语音，太大又会漏掉轻声
        float threshold = std::max(0.01f, avg_energy * 0.5f);

        bool is_speech = false;
        int hangover_counter = 0;

        // hangover 表示从语音切到静音时再等待几帧，避免把一句话中间很短的停顿切开
        const int kHangoverFrames = 10;
        size_t current_start = 0;
        for (size_t i = 0; i < total_samples; i += static_cast<size_t>(frame_size)) {
            float sum_sq = 0.0f;
            size_t count = 0;

            for (size_t j = 0; j < static_cast<size_t>(frame_size) && (i + j) < total_samples; ++j) {
                float val = pcm_data[i + j];
                sum_sq += val * val;
                ++count;
            }

            if (count == 0) {
                break;
            }

            float rms = std::sqrt(sum_sq / static_cast<float>(count));

            // 状态机逻辑
            if (rms > threshold) {
                // 能量超过阈值 -> 语音
                if (!is_speech) {
                    is_speech = true;
                    current_start = i; // 记录开始点
                }

                hangover_counter = kHangoverFrames; // 重置挂起计数器
            } else {
                // 能量低于阈值 -> 可能是静音，也可能是气口
                if (is_speech) {
                    if (hangover_counter > 0) {
                        hangover_counter--; //还在挂起期，保持语音状态
                    } else {
                        // 挂起期结束，结束语音，记录结束点
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

    void processFrame(std::vector<float>& pcm_frame, const AudioPreprocessConfig& config) {
        if (pcm_frame.empty()) {
            return;
        }

        if (config.enable_normalize) {
            normalize(pcm_frame, config.normalize_peak);
        }

        if (config.enable_denoise) {
            denoise(pcm_frame, config.denoise_threshold_db);
        }

        if (config.enable_pre_emphasis) {
            if (config.streaming_pre_emphasis) {
                preEmphasizeStreaming(pcm_frame, config.pre_emphasis_alpha);
            } else {
                preEmphasize(pcm_frame, config.pre_emphasis_alpha);
            }
        }
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

void AudioPreprocessor::preEmphasizeStreaming(std::vector<float>& pcm_data, float alpha) {
    pImpl->preEmphasizeStreaming(pcm_data, alpha);
}

void AudioPreprocessor::resetStreamingState() {
    pImpl->resetStreamingState();
}

void AudioPreprocessor::denoise(std::vector<float>& pcm_data, float threshold_db) {
    pImpl->denoise(pcm_data, threshold_db);
}

std::vector<SpeechSegment> AudioPreprocessor::detectSpeech(const std::vector<float>& pcm_data, int sample_rate) {
    return pImpl->detectSpeech(pcm_data, sample_rate);
}

void AudioPreprocessor::processFrame(std::vector<float>& pcm_frame, const AudioPreprocessConfig& config) {
    pImpl->processFrame(pcm_frame, config);
}

} // namespace Audio
} // namespace DigitalHuman