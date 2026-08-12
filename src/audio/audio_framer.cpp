#include "audio/audio_framer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace DigitalHuman {
namespace Audio {

/**
 * @brief 模块的核心职责是：
 * 1. 根据采样率、帧长、帧移计算 frame_size 和 stride_size；
 * 2. 根据窗函数类型生成 window 系数；
 * 3. 从连续 PCM 数据中切出固定长度音频帧；
 * 4. 对音频帧应用窗函数；
 * 5. 支持一次性批量分配 process()。
 */
struct AudioFramer::Impl {
    int sample_rate = 16000;
    int frame_size = 800;
    int stride_size = 200;
    WindowType win_type = WindowType::Hamming;

    // 窗函数系数数组
    std::vector<float> window;

    Impl(int sr, double frame_ms, double stride_ms, WindowType wt) 
        : sample_rate(sr), win_type(wt) {
        frame_size = static_cast<int>(std::round(sample_rate * frame_ms / 1000.0));
        stride_size = static_cast<int>(std::round(sample_rate * stride_ms / 1000.0));

        if (frame_size <= 0) {
            frame_size = 1;
        }

        if (stride_size <= 0) {
            stride_size = 1;
        }

        // 根据 frame_size 和 win_type 生成窗函数系数
        generateWindow();
    }

    // 生成窗函数系数
    void generateWindow() {
        // 先初始化为全 1
        window.resize(frame_size, 1.0f);

        // 当 frame_size <= 1 时，没有必要计算窗函数，同时避免 frame_size - 1 为 0，导致除 0
        if (frame_size <= 1) {
            return;
        }

        // 逐点计算窗函数系数
        for (int i = 0; i < frame_size; i++) {
            switch(win_type) {
                case WindowType::Hamming:
                    /**
                     * Hamming 窗公式：
                     * w[n] = 0.54 - 0.46 * cos(2pi * n / (N-1))
                     */
                    window[i] = 0.54f - 0.46f * std::cos(
                        2.0f * static_cast<float>(M_PI) * i / (frame_size - 1)
                    );
                    break;
                
                case WindowType::Hanning:
                    /**
                     * Hanning 窗公式：
                     * w[n] = 0.5 * (1 - (cos(2pi * n / (N-1)))
                    */
                    window[i] = 0.5f * (1.0f - std::cos(
                        2.0f * static_cast<float>(M_PI) * i / (frame_size - 1)
                    ));
                    break;

                case WindowType::Node:
                default:
                    // 不加窗，window[i] = 1.0，表示 frame[i] 乘以 1 后保持不变
                    window[i] = 1.0f;
                    break;
            }
        }
    }

    /**
     * @brief 从连续 PCM 数据中创建一帧原始音频帧
     * @param pcm_data 输入的一维 float PCM 数据
     * @param start_index 当前帧起始采样点位置
     * @param pad_tail 如果尾部不足一帧，是否补零
     * @return std::vector<float>
     *          返回长度为 frame_size 的一帧数据
     */
    std::vector<float> createFrame(const std::vector<float>& pcm_data,
                                   size_t start_index,
                                   bool pad_tail) const 
    {
        /**
         * 默认创建一帧全 0 数据。这样当尾部不足一帧且 pad_tail = true 时，
         * 未被实际 PCM 覆盖的部分天然就是 0，相当于补零
         */
        std::vector<float> frame(frame_size, 0.0f);

        // 如果输入为空，或者起始位置已经超过输入长度，则直接返回一帧全 0 数据
        if (pcm_data.empty() || start_index >= pcm_data.size()) {
            return frame;
        }
        // available 表示从 start_index 开始，输入 pcm_data 中还剩多少个采样点
        size_t available = pcm_data.size() - start_index;

        size_t copy_len = std::min(available, static_cast<size_t>(frame_size));

        /**
         * 将 pcm_data 中的一段数据复制到 frame 开头
         * 复制范围：
         *  [start_index, start_index + copy_len) 
         */
        std::copy(
            pcm_data.begin() + static_cast<std::ptrdiff_t>(start_index),
            pcm_data.begin() + static_cast<std::ptrdiff_t>(start_index + copy_len),
            frame.begin()
        );

        /**
         * 如果尾部不足一帧，并且不允许补零，则清空frame
         */
        if (!pad_tail && copy_len < static_cast<size_t>(frame_size)) {
            frame.clear();
        }
        return frame;
    }

    /**
     * @brief 对一帧音频原地应用窗函数
     * @param frame 输入/输出音频帧
     */
    void applyWindow(std::vector<float>& frame) const {
        if (frame.size() != static_cast<size_t>(frame_size)) {
            return;
        }
        /**
         * 加窗：
         * frame[i] = frame[i] * window[i] 
         */
        for (int i = 0; i < frame_size; i++) {
            frame[i] = frame[i] * window[i];
        }
    }

    /**
     * @brief 批量分帧处理
     * @param pcm_data 输入的一维 float PCM 数据
     * @param pad_tail 尾部不足一帧时是否补零
     * @return std::vector<std::vector<float>> frames[帧编号][帧内采样点] 
     */
    std::vector<std::vector<float>> process(const std::vector<float>& pcm_data,
                                            bool pad_tail) const 
    {
        std::vector<std::vector<float>> frames;

        // 空校验
        if (pcm_data.empty()) {
            return frames;
        }

        size_t start = 0;

        while (start < pcm_data.size()) {
            std::vector<float> frame = createFrame(pcm_data, start, pad_tail);

            // 此时如果 frame 为空，表示尾部不足一帧，pad_tail = false，createFrame清空了
            if (frame.empty()) {
                break;
            }
            
            // 对当前帧应用窗函数
            applyWindow(frame);

            // 保存当前帧
            frames.push_back(std::move(frame));

            // 移动到下一帧起始位置
            start += static_cast<size_t>(stride_size);

        }
        return frames;
    }

};

AudioFramer::AudioFramer(int sample_rate,
                        double frame_duration_ms,
                        double stride_duration_ms,
                        WindowType win_type) 
    : pImpl(std::make_unique<Impl>(
        sample_rate,
        frame_duration_ms,
        stride_duration_ms,
        win_type
    )) {}

AudioFramer::~AudioFramer() = default;

AudioFramer::AudioFramer(AudioFramer&&) noexcept = default;
AudioFramer& AudioFramer::operator=(AudioFramer&&) noexcept = default;

std::vector<std::vector<float>> AudioFramer::process(const std::vector<float>& pcm_data,
                                                    bool pad_tail) const {
    return pImpl->process(pcm_data, pad_tail);
}

std::vector<float> AudioFramer::createFrame(const std::vector<float>& pcm_data,
                                            size_t start_index,
                                            bool pad_tail) const {
    return pImpl->createFrame(pcm_data, start_index, pad_tail);
}

void AudioFramer::applyWindow(std::vector<float>& frame) const {
    pImpl->applyWindow(frame);
}

int AudioFramer::getFrameSize() const {
    return pImpl->frame_size;
}

int AudioFramer::getStrideSize() const {
    return pImpl->stride_size;
}

int AudioFramer::getSampleRate() const {
    return pImpl->sample_rate;

}

} // namespace Audio
} // namespace DigitalHuman