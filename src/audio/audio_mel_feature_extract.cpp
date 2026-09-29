#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc.hpp>

#include "audio/audio_mel_feature_extract.h"

namespace DigitalHuman {
namespace Audio {

struct MelFeatureExtractor::Impl {
    // 定义基本变量
    int sample_rate, n_fft, n_mels;
    float fmin, fmax;

    // 预先定义的梅尔滤波器组矩阵 [n_mels, n_fft / 2 + 1]
    cv::Mat mel_basis;

    // 归一化参数
    const float min_level_db = -100.0f;  // 绝对静音的阈值，防止背景噪音被归一化算法方法
    const float ref_level_db = 20.0f;

    Impl (int sr, int fft, int mels, float min_f, float max_f) 
            : sample_rate(sr), n_fft(fft), n_mels(mels), fmin(min_f), fmax(max_f){
        
        // 初始化梅尔滤波器矩阵
        initMelBasis();
    }

    // 构建梅尔矩阵
    // 频率域转换算法
    static float hz_to_mel(float fre) {
        return 2595.0f * std::log10(1.0f + fre / 700.0f);
    }

    static float mel_to_hz(float mel) {
        return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }

    void initMelBasis() {
        int n_fft_bins = n_fft / 2 + 1;
        mel_basis = cv::Mat::zeros(n_mels, n_fft_bins, CV_32F);

        // 频率域转换
        float mel_min = hz_to_mel(fmin);
        float mel_max = hz_to_mel(fmax);

        // 定义梅尔滤波器矩阵
        std::vector<float> mel_points(n_mels + 2);
        float step = (mel_max - mel_min) / (n_mels + 1);  // 均匀切分

        for (int i = 0; i < n_mels + 2; i++) {
            mel_points[i] = mel_to_hz(mel_min + i * step);
        }

        // 将 Hz 映射到 FFT bin 索引
        std::vector<int> bins(n_mels + 2);
        for (int i = 0; i < n_mels + 2; i++) {
            bins[i] = std::floor(mel_points[i] * n_fft_bins / (sample_rate / 2.0f));
            bins[i] = std::min(bins[i], n_fft_bins - 1);
        }

        // 填充三角滤波器
        for (int i = 0; i < n_mels; i++) {
            int left = bins[i];
            int center = bins[i+1];
            int right = bins[i + 2];

            // 绘制升坡
            for (int k = left; k < center; k++) {
                mel_basis.at<float>(i, k) = (float)(k - left) / (center - left);
            }

            // 降坡
            for (int k = center; k < right; ++k) {
                mel_basis.at<float>(i, k) = (float)(right - k) / (right - center);
            }
        }
    }

    // 提取单帧
    cv::Mat extract(const std::vector<float>& pcm_frame) {
        // 1. 准备输入
        cv::Mat input_frame(1, n_fft, CV_32F, cv::Scalar(0));
        int copy_len = std::min((int)pcm_frame.size(), n_fft);
        memcpy(input_frame.ptr<float>(0), pcm_frame.data(), copy_len * sizeof(float));

        // 2. FFT 变换
        cv::Mat planes[] = {input_frame, cv::Mat::zeros(input_frame.size(), CV_32F)};
        cv::Mat complex_img;
        cv::merge(planes, 2, complex_img);
        cv::dft(complex_img, complex_img);

        cv::split(complex_img, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);
        cv::Mat mag_spec = planes[0].colRange(0, n_fft / 2 + 1);

        // 4. 梅尔滤波 
        cv::Mat mel_spec;
        cv::gemm(mag_spec, mel_basis, 1.0, cv::Mat(), 0.0, mel_spec, cv::GEMM_2_T);

        // 5. 归一化：输出 [0, 1]
        const float max_abs_value = 4.0f;

        for (int i = 0; i < n_mels; i++) {
            float val = mel_spec.at<float>(0, i);

            // amplitude_to_db
            val = 20.0f * std::log10(std::max(val, 1e-5f));

            // ref_level_db = 20
            val -= ref_level_db;

            // min_level_db = -100
            val = std::max(val, min_level_db);

            // normalize to [0, 1]
            float norm = (val - min_level_db) / (-min_level_db);

            // symmetric normalize to [-4, 4]
            float sym = 2.0f * max_abs_value * norm - max_abs_value;

            // clip
            sym = std::max(-max_abs_value, std::min(max_abs_value, sym));

            mel_spec.at<float>(0, i) = sym;
        }
        
        return mel_spec;
    }

};

MelFeatureExtractor::MelFeatureExtractor(int sr, int fft, int mels, float minf, float maxf)
    : pImpl(std::make_unique<Impl>(sr, fft, mels, minf, maxf)) {}

MelFeatureExtractor::~MelFeatureExtractor() = default;

MelFeatureExtractor::MelFeatureExtractor(MelFeatureExtractor&&) noexcept = default;
MelFeatureExtractor& MelFeatureExtractor::operator=(MelFeatureExtractor&&) noexcept = default;

MelFeatureExtractor::MelFeatureExtractor(const MelFeatureExtractor&) = delete;
MelFeatureExtractor& MelFeatureExtractor::operator=(const MelFeatureExtractor&)  = delete;

// 单次提取函数
cv::Mat MelFeatureExtractor::extract(const std::vector<float>& pcm_frame) {
    return pImpl->extract(pcm_frame);
}

// 批量提取
cv::Mat MelFeatureExtractor::extractBatch(const std::vector<std::vector<float>>& frames) {
    if (frames.empty()) {
        return cv::Mat();
    }

    // 预分配大矩阵 [N, 80] 
    cv::Mat batch_mels(frames.size(), pImpl->n_mels, CV_32F);

    for (size_t i = 0; i < frames.size(); i++) {
        cv::Mat row = pImpl->extract(frames[i]);
        row.copyTo(batch_mels.row(i));
    }

    return batch_mels;
}

} // namespace Audio
} // namespace DigitalHuman