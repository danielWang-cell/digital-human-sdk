#include <iostream>
#include <vector>
#include <cmath>

#include "audio/audio_framer.h"

using namespace DigitalHuman::Audio;

// 生成 1秒 440Hz 正弦波 (16k采样率)
std::vector<float> generateSineWave(int sample_rate, float duration_sec, float freq) {
    int num_samples = static_cast<int>(sample_rate * duration_sec);
    std::vector<float> pcm(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float t = (float)i / sample_rate;
        pcm[i] = sinf(2.0f * 3.14159f * freq * t);
    }
    return pcm;
}

int main() {
    std::cout << "=== Digital Human SDK: Audio Framer Test ===" << std::endl;

    // 1. 生成测试音频
    int sr = 16000;
    auto pcm = generateSineWave(sr, 0.1f, 440.0f); // 0.1秒音频
    std::cout << "[Info] Generated PCM size: " << pcm.size() << " samples" << std::endl;

    // 2. 初始化分帧器 (25ms 帧长, 10ms 帧移, Hamming窗)
    // Frame size = 16000 * 0.025 = 400
    // Stride size = 16000 * 0.010 = 160
    AudioFramer framer(sr, 25, 10, WindowType::Hamming);
    
    std::cout << "[Info] Frame Size: " << framer.getFrameSize() << std::endl;
    std::cout << "[Info] Stride Size: " << framer.getStrideSize() << std::endl;

    // 3. 执行分帧
    auto frames = framer.process(pcm);
    std::cout << "[Result] Total Frames: " << frames.size() << std::endl;

    // 4. 验证数据 (打印第一帧的前10个和后10个数值)
    // 由于加了 Hamming 窗，边缘数值应该被压低接近 0
    if (!frames.empty()) {
        const auto& f0 = frames[0];
        std::cout << "\n--- Frame 0 Data Check (With Hamming Window) ---" << std::endl;
        std::cout << "Head (should be small): ";
        for(int i=0; i<5; ++i) printf("%.4f ", f0[i]);
        std::cout << "..." << std::endl;
        
        std::cout << "Center (should be large): ";
        for(int i=198; i<203; ++i) printf("%.4f ", f0[i]);
        std::cout << "..." << std::endl;

        std::cout << "Tail (should be small): ";
        for(int i=395; i<400; ++i) printf("%.4f ", f0[i]);
        std::cout << std::endl;
    }

    // 5. 边界测试 (输入极短音频)
    std::cout << "\n[Test] Boundary Condition (Tiny Input)..." << std::endl;
    std::vector<float> tiny_pcm(100, 1.0f); // 只有100个采样点 (小于一帧400)
    auto tiny_frames = framer.process(tiny_pcm);
    std::cout << "Input 100 samples -> Frames: " << tiny_frames.size() << std::endl;
    if (!tiny_frames.empty()) {
        std::cout << "Frame 0 size: " << tiny_frames[0].size() << " (Should be padded to 400)" << std::endl;
    }

    return 0;
}