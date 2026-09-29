#include <iostream>
#include <algorithm>
#include <cmath>

#include "audio/audio_framer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace DigitalHuman {
namespace Audio {

struct AudioFramer::Impl
{
    /* data */
    // 采样率，帧长，步长
    int sample_rate, frame_size, stride_size;
    WindowType win_type;
    std::vector<float> window;      // 预计算的窗函数数组

    Impl(int sr, int frame_ms, int stride_ms, WindowType wt) : sample_rate(sr), win_type(wt) {
        // 计算采样点个数
        frame_size = static_cast<int>(sr * frame_ms / 1000.0f);
        stride_size = static_cast<int>(sr * stride_ms / 1000.0f);

        // 预计算窗口
        generateWindow();
    }

    // 预计算窗口
    void generateWindow() {
        // 预分配内存，减少动态扩容次数
        window.resize(frame_size);
        for (int i = 0; i < frame_size; i++) {
            switch (win_type) {
                case WindowType::Hamming:
                    // Hamming: 0.54 - 0.46 * cos(2pi * n / (N-1))
                    window[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (frame_size - 1));
                    break;
                case WindowType::Hanning:
                    // Hanning: 0.5 * (1 - cos(2pi * n / (N-1)))
                    window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (frame_size - 1)));
                    break;
                case WindowType::None:
                default:
                    window[i] = 1.0f;
                    break;
            }
        }
    }

    std::vector<std::vector<float>> process(const std::vector<float>& pcm_data) {
        std::vector<std::vector<float>> frames;

        // 空检查
        if (pcm_data.empty()) {
            return frames;
        }

        long long signal_len = pcm_data.size();

        // 计算需要的帧数 (向上取整)
        // num_frames = ceil((L - frame_size) / stride) + 1
        // 如果信号长度小于帧长，至少要补齐一帧
        int num_frames = 0;
        if (signal_len <= frame_size) {
            num_frames = 1;
        } else {
            num_frames = 1 + static_cast<int>(ceil((signal_len - frame_size) / (float)stride_size));
        }

        // 准备填充后的数据
        // 需要的总长度是：（num_frames - 1) * stride + frame_size
        long long needed_len = (long long)(num_frames - 1) * stride_size + frame_size;
        long long pad_len = needed_len - signal_len;

        // 复制原始数据
        std::vector<float> padded_signal = pcm_data;
        // 开始填充零
        if (pad_len > 0) {
            // 在当前帧的末尾填充 pad_len 个 0
            padded_signal.insert(padded_signal.end(), pad_len, 0.0f);
        }

        frames.reserve(num_frames);  // 分配 frames空间，预留 num_frames的大小
        
        // 做滑动窗口切分
        for (int i = 0; i < num_frames; i++) {
            int start_index = i * stride_size;

            // 创建一个新帧
            std::vector<float> frame(frame_size);

            // 复制并加窗
            for (int j = 0; j < frame_size; j++) {
                // frame[j] = signal[start + j] * window[j]
                frame[j] = padded_signal[start_index + j] * window[j];
            }

            frames.push_back(std::move(frame));
        }

        return frames;
    }
};


// pImpl外部接口
AudioFramer::AudioFramer(int sr, int f_ms, int s_ms, WindowType wt)
    : pImpl(std::make_unique<Impl>(sr, f_ms, s_ms, wt)){}

AudioFramer::~AudioFramer() = default;

AudioFramer::AudioFramer(AudioFramer&&) noexcept = default;
AudioFramer& AudioFramer::operator=(AudioFramer&&) noexcept = default;
AudioFramer::AudioFramer(const AudioFramer&) = delete;
AudioFramer& AudioFramer::operator=(const AudioFramer&) = delete;

std::vector<std::vector<float>> AudioFramer::process(const std::vector<float>& pcm_data) {
    return pImpl->process(pcm_data);
}

int AudioFramer::getFrameSize() const {
    return pImpl->frame_size;
}

int AudioFramer::getStrideSize() const {
    return pImpl->stride_size;
}


}
}