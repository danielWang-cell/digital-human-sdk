#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <chrono>

#include "audio/audio_video_Synchronous.h"

using namespace DigitalHuman::Audio;

// 生成 440Hz 测试用正弦波 (模拟语音数据)
std::vector<float> generateSineWave(int sample_rate, float duration_sec) {
    int num_samples = static_cast<int>(sample_rate * duration_sec);
    std::vector<float> pcm(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float t = (float)i / sample_rate;
        pcm[i] = 0.5f * sinf(2.0f * 3.14159f * 440.0f * t);
    }
    return pcm;
}

int main() {
    std::cout << "=== Digital Human SDK: A/V Sync Comprehensive Test ===" << std::endl;

    AudioPlayer player;
    
    if (!player.open(16000, 1, 512)) {
        std::cerr << "Failed to open AudioPlayer." << std::endl;
        return -1;
    }

    // 1. 提前灌入 5 秒钟的音频数据
    auto dummy_audio = generateSineWave(16000, 5.0f);
    player.pushData(dummy_audio);
    std::cout << "[Info] Pushed 5 seconds of audio data." << std::endl;

    // 2. 启动音频播放
    player.play();
    std::cout << "[Info] Audio Playback Started. Audio Master Clock is TICKING...\n" << std::endl;

    // 3. 模拟视频渲染主循环 (Video Rendering Loop)
    const double target_fps = 25.0;
    const double frame_duration_ms = 1000.0 / target_fps; // 40ms/frame
    int video_frame_index = 0;

    auto loop_start = std::chrono::steady_clock::now();
    int current_phase = 0;

    // 循环运行 3 秒钟
    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sys_time = std::chrono::duration<double, std::milli>(now - loop_start).count();
        if (elapsed_sys_time > 3000.0) {
            break; // 跑 3 秒就结束
        }

        // -------------------------------------------------------------
        // 动态调整模拟渲染耗时，以测试不同的同步策略
        // -------------------------------------------------------------
        int simulated_render_cost = 35; // 默认 35ms
        int new_phase = 0;

        if (elapsed_sys_time < 1000.0) {
            new_phase = 1;
            simulated_render_cost = 35; // Phase 1: 正常渲染 (耗时 < 40ms)
        } else if (elapsed_sys_time < 2000.0) {
            new_phase = 2;
            simulated_render_cost = 80; // Phase 2: 系统卡顿，视频滞后 (耗时 > 40ms)
        } else {
            new_phase = 3;
            simulated_render_cost = 10; // Phase 3: 渲染极快，视频超前 (耗时 远小于 40ms)
        }

        // 打印阶段切换提示
        if (new_phase != current_phase) {
            std::cout << "\n========================================================" << std::endl;
            if (new_phase == 1) std::cout << ">>> Phase 1: Normal Rendering (~35ms/frame) -> Expect RENDER" << std::endl;
            if (new_phase == 2) std::cout << ">>> Phase 2: Heavy Load/Lag (~80ms/frame) -> Expect DROPs" << std::endl;
            if (new_phase == 3) std::cout << ">>> Phase 3: Fast Rendering (~10ms/frame) -> Expect WAITs" << std::endl;
            std::cout << "========================================================\n" << std::endl;
            current_phase = new_phase;
        }

        // 同步过程 1: 查询 Audio Master Clock
        double audio_pts = player.getCurrentTime();
        
        // 同步过程 2: 计算当前准备渲染的视频帧的期望 PTS
        double video_pts = video_frame_index * frame_duration_ms;

        // 同步过程 3: 计算偏差并制定策略
        double diff = video_pts - audio_pts;

        if (diff > 40.0) {
            // 视频比声音快了超过一帧 -> 策略：Wait (等待)
            std::cout << "\033[33m[SYNC WAIT] Video is too fast! (Audio: " << audio_pts 
                      << "ms, Video: " << video_pts << "ms). Waiting...\033[0m\n";
            // 视频线程休眠，让音频继续播
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            // 因为这里休眠了，所以不增加 video_frame_index (重复上一帧)
            continue; 
        } 
        else if (diff < -40.0) {
            // 视频比声音慢了超过一帧 -> 策略：Drop (丢帧)
            std::cout << "\033[31m[SYNC DROP] Video is too slow! (Audio: " << audio_pts 
                      << "ms, Video: " << video_pts << "ms). Dropping frame " << video_frame_index << ".\033[0m\n";
            video_frame_index++; // 丢弃当前帧，直接计算下一帧 (不消耗渲染时间)
            continue; 
        } 
        else {
            // 在容忍误差内 [-40, 40] -> 策略：Render (渲染)
            std::cout << "\033[32m[RENDER]    Sync OK. Frame " << video_frame_index 
                      << " displayed at Audio Time " << audio_pts << " ms (Diff: " << diff << "ms)\033[0m\n";
            
            // 模拟视频推理/渲染耗时
            std::this_thread::sleep_for(std::chrono::milliseconds(simulated_render_cost)); 
            
            video_frame_index++;
        }
    }

    player.stop();
    std::cout << "\n=== Test Finished ===" << std::endl;
    return 0;
}