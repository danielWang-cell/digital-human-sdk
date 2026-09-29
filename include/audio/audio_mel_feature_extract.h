#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Audio {

/**
 * @brief 梅尔频谱特征提取器，这是Wav2Lip的预处理步骤，负责将分帧之后的音频数据转换为 Mel-Spectrogram 特征
 */
class MelFeatureExtractor {
public:
    /**
     * @brief 构造函数
     * @param sample_rate 采样率，默认 16000
     * @param n_fft FFT点数，默认 800 
     * @param n_mels 梅尔滤波器数量，默认80
     * @param fmin 最低频率，默认55
     * @param fmax 最高频率，默认7600
     */
    MelFeatureExtractor(int sample_rate = 16000,
                        int n_fft = 800,
                        int n_mels = 80,
                        float fmin = 55.0f,
                        float fmax = 7600.0f);
    
    ~MelFeatureExtractor();

    MelFeatureExtractor(const MelFeatureExtractor&);
    MelFeatureExtractor& operator=(const MelFeatureExtractor&);

    MelFeatureExtractor(MelFeatureExtractor&&) noexcept;
    MelFeatureExtractor& operator=(MelFeatureExtractor&&) noexcept;

    /**
     * @brief 提取单帧音频的梅尔特性
     * @param pcm_frame 加窗后的音频帧数据，长度应该 ≤ n_fft
     * @return cv::Mat 1*80的行向量 CV_32F, 范围[0.0, 1.0]
     */
    cv::Mat extract(const std::vector<float>& pcm_frame);

    /**
     * @brief 批量提取，将多帧拼接成一张频谱图
     * @param frames 分帧之后的数据数组
     * @return cv::Mat [N_frames * 80]的矩阵，给模型提供输入
     */
    cv::Mat extractBatch(const std::vector<std::vector<float>>& frames);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};


} // namespace Audio
} // namespace DigitalHuman