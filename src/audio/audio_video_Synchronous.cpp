#include <iostream>
#include <portaudio.h>
#include <atomic>
#include <algorithm>
#include <cstring>
#include <spdlog/spdlog.h>

#include "audio/audio_buffer.h"
#include "audio/audio_video_Synchronous.h"



namespace DigitalHuman {
namespace Audio {

struct AudioPlayer::Impl {
    PaStream* stream = nullptr;
    Audio::AudioBuffer buffer;      // 内部环形缓冲区

    // --- 跨线程共享的核心状态 ---
    std::atomic<int64_t> played_samples{0};
    std::atomic<PlayState> state{PlayState::Stopped};

    int sample_rate = 16000;    // 采样率
    int channels = 1;           // 声道数

    // 初始化 PortAudio
    Impl() : buffer(16000 * 10) {
        PaError err = Pa_Initialize();
        if (err != paNoError) {
            spdlog::error("[AudioPlayer] PortAudio Init Error: {}", Pa_GetErrorText(err));
        }
    }

    // 清理资源
    ~Impl() {
        if (stream) {
            Pa_StopStream(stream);
            Pa_CloseStream(stream);
        }
        Pa_Terminate();
    }

    // PortAudio 核心回调函数，高优先级实时线程
    // 以极快的速度从 AudioBuffer 获取数据塞给声卡，并更新系统时钟
    static int paCallback(const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void *userData)
    {
        auto* impl = static_cast<Impl*>(userData);
        float* out = static_cast<float*>(outputBuffer);

        // 如果不是播放状态，填充静音，保持音频流开启以维持极低延迟
        if (impl->state.load(std::memory_order_acquire) != PlayState::Playing) {
            std::fill(out, out + framesPerBuffer * impl->channels, 0.0f);
            return paContinue;
        }
        
        // 从环形缓冲区直接拉取原始数据到 out (零内存分配，避免高优先级回调卡顿)
        size_t samples_needed = framesPerBuffer * impl->channels;
        bool success = impl->buffer.pullRaw(out, samples_needed, -1);

        if (success) {

            // 原子增加已播放样本数，这里会推动全局的时间轴前进
            impl->played_samples.fetch_add(framesPerBuffer, std::memory_order_release);
        } else {
            // 如果数据不足，填充静音防止爆音，时钟暂停前进
            std::fill(out, out + framesPerBuffer * impl->channels, 0.0f);
        }

        return paContinue;
    }   
};

AudioPlayer::AudioPlayer() : pImpl(std::make_unique<Impl>()) {}
AudioPlayer::~AudioPlayer() = default;
AudioPlayer::AudioPlayer(AudioPlayer&&) noexcept = default;
AudioPlayer& AudioPlayer::operator=(AudioPlayer&&) noexcept = default;


bool AudioPlayer::open(int sample_rate, int channels, int frames_per_buffer) {
    if (pImpl->stream) {
        return true;
    }

    pImpl->sample_rate = sample_rate;
    pImpl->channels = channels;

    // 打开默认的音频输出流
    PaError err = Pa_OpenDefaultStream(
        &pImpl->stream,         
        0,                      // 无输入音频
        channels,               // 声道数
        paFloat32,              // 32 bit floating point output (与网络模型一致)
        sample_rate,
        frames_per_buffer,      // 缓冲区帧数
        Impl::paCallback,       // 回调函数
        pImpl.get()             // 将 Impl 对象作为 userData 传入
    );

    if (err != paNoError) {        
        spdlog::error("[AudioPlayer] Pa_OpenDefaultStream Error: {}", Pa_GetErrorText(err));
        return false;    
    }    

    // 启动流，但此时 state 为 Stopped，回调中会输出静音    
    err = Pa_StartStream(pImpl->stream);    
    if (err != paNoError) {        
        spdlog::error("[AudioPlayer] Pa_StartStream Error: {}", Pa_GetErrorText(err));
        
        // 清理已被 Opened 的 stream 句柄，防止状态残留
        Pa_CloseStream(pImpl->stream);
        pImpl->stream = nullptr;
        return false;    
    }

    return true;
}

void AudioPlayer::pushData(const std::vector<float>& pcm_data){
    // 默认使用 Overwrite 或者 Block 策略
    pImpl->buffer.push(pcm_data, Audio::BufferOverflowStrategy::Block);
}

void AudioPlayer::play() {
    if (!pImpl->stream) {
        return;
    }

    pImpl->state.store(PlayState::Playing, std::memory_order_release);
}

void AudioPlayer::pause() {
    pImpl->state.store(PlayState::Paused, std::memory_order_release);
}

void AudioPlayer::stop() {
    pImpl->state.store(PlayState::Stopped, std::memory_order_release);
    pImpl->buffer.clear();
    pImpl->played_samples.store(0, std::memory_order_release); // 重置时钟
}

double AudioPlayer::getCurrentTime() const {
    if (pImpl->sample_rate == 0) {
        return 0.0;
    }

    // 时间戳公式：已播放的样本总数 / 采样率 * 1000 = 当前毫秒数    
    int64_t samples = pImpl->played_samples.load(std::memory_order_acquire);
    return (static_cast<double>(samples) / pImpl->sample_rate) * 1000.0;
}

PlayState AudioPlayer::getState() const {
    return pImpl->state.load(std::memory_order_acquire);
}

double AudioPlayer::getBufferedDuration() const {
    if (pImpl->sample_rate == 0) {
        return 0.0;
    }
    size_t remaining_samples = pImpl->buffer.size();
    return (static_cast<double>(remaining_samples) / pImpl->sample_rate) * 1000.0;
}

}   // namespace Audio
}   // namespace DigitalHuman

