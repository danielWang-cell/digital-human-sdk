#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 

#include "audio/audio_framer.h"       
#include "audio/audio_mel_feature_extract.h" 

using namespace DigitalHuman::Audio; 

// 生成 1秒 440Hz 正弦波 + 880Hz 泛音
std::vector<float> generateSignal() {
    int sr = 16000;
    std::vector<float> pcm(sr); // 1秒
    for (int i = 0; i < sr; ++i) {
        float t = (float)i / sr;
        // 440Hz 主音 + 880Hz 弱泛音
        pcm[i] = 0.6f * sinf(2 * 3.14159f * 440 * t) + 
                 0.3f * sinf(2 * 3.14159f * 880 * t);
    }
    return pcm;
}

int main() {
    std::cout << "=== Digital Human SDK: Mel Spectrogram Test (Audio Module) ===" << std::endl;

    // 1. 生成音频
    auto pcm = generateSignal();
    std::cout << "[Info] Signal generated: " << pcm.size() << " samples" << std::endl;

    // 2. 分帧 
    AudioFramer framer(16000, 50, 12.5, WindowType::Hamming);
    auto frames = framer.process(pcm);
    std::cout << "[Info] Framed into " << frames.size() << " frames." << std::endl;

    // 3. 提取梅尔谱 
    MelFeatureExtractor mel_extractor(16000, 800, 80);
    cv::Mat mel_spectrogram = mel_extractor.extractBatch(frames);

    std::cout << "[Result] Mel Spectrogram Shape: " 
              << mel_spectrogram.rows << "x" << mel_spectrogram.cols << std::endl;

    // 4. 可视化保存
    cv::Mat vis;
    mel_spectrogram.convertTo(vis, CV_8U, 255.0);
    cv::transpose(vis, vis);
    cv::flip(vis, vis, 0);
    cv::resize(vis, vis, cv::Size(800, 400), 0, 0, cv::INTER_NEAREST);

    cv::imwrite("mel_spectrogram_result.jpg", vis);
    std::cout << "[Success] Saved visualization to mel_spectrogram_result.jpg" << std::endl;

    return 0;
}