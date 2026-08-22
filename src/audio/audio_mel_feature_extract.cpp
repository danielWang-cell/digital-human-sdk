#include "audio/audio_mel_feature_extract.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <opencv2/imgproc.hpp>

namespace DigitalHuman {
namespace Audio {
    
struct MelFeatureExtractor::Impl {
    // 基础变量
    int sample_rate = 16000;
    int n_fft = 800;
    int n_mels = 80;
    float fmin = 55.0f;
    float fmax = 7600.0f;

    // 滤波器组矩阵
    cv::Mat mel_basis;

    // 最小动态范围，低于该值的 db 会被裁剪
    const float min_level_db = -100.0f;

    // 参考 db 值，用于整体平移
    const float ref_level_db = 20.0f;

    // 最终归一化后的最大绝对值
    // 输出 Mel 特征会被压到 [-4, 4]
    const float max_abs_value = 4.0f;

    Impl(int sr, int fft, int mels, float min_f, float max_f)
        : sample_rate(sr),
          n_fft(fft),
          n_mels(mels),
          fmin(min_f),
          fmax(max_f) {
        
        initMelBasis();
    }

    /**
     * @brief Hz 频率转换为 Mel 频率
     * @param hz 线性频率，单位 Hz
     * @return float Mel 频率
     * 公式：
            mel = 2595 * log10(1 + hz / 700)
     */
    static float hzToMel(float hz) {
        return 2595.0f * std::log10(1.0f + hz / 700.0f);
    }

    /**
     * @brief Mel 频率转换回 Hz 频率
     * @param mel Mel 频率
     * @return float 线性频率，单位 Hz
     * 公式：
            hz = 700 * (10^(mel / 2595) - 1) 
     */
    static float melToHz(float mel) {
        return 700 * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
    }

    // 作用是提前构建一个 Mel 滤波器矩阵
    void initMelBasis() {
        // 对于实数信号，FFT 结果具有共轭对称性，因此只需要保留前一半频谱
        const int n_fft_bins = n_fft / 2 + 1;
        // 创建 Mel 滤波器组矩阵
        mel_basis = cv::Mat::zeros(n_mels, n_fft_bins, CV_32F);

        // 将最低频率和最高频率从 Hz 转成 Mel
        const float mel_min = hzToMel(fmin);
        const float mel_max=  hzToMel(fmax);

        // Mel 滤波器需要 n_mels + 2 个边界点
        std::vector<float> mel_points(n_mels + 2);

        // 在 Mel 尺度上均匀划分
        const float step = (mel_max - mel_min) / static_cast<float>(n_mels + 1);

        for (int i = 0; i < n_mels + 2; ++i) {
            // 先在 Mel 轴上取点，再转回 Hz
            mel_points[i] = melToHz(mel_min + step * i);
        }

        // 将 Hz 频率点映射到 FFT bin 下标
        // n_fft_bins 覆盖 0Hz 到 sample_rate / 2
        // bin = hz * n_fft_bins / (sample_rate / 2)
        std::vector<int> bins(n_mels + 2);
        for (int i = 0; i < n_mels + 2; ++i) {
            bins[i] = static_cast<int>(
                std::floor(mel_points[i] * n_fft_bins / (sample_rate / 2.0f))
            );
            bins[i] = std::max(0, std::min(bins[i], n_fft_bins - 1));
        }

        // 逐个构建三角 Mel 滤波器
        for (int i = 0; i < n_mels; ++i) {
            int left = bins[i];
            int center = bins[i + 1];
            int right = bins[i + 2];

            if (center <= left) {
                center = left + 1;
            }
            if (right <= center) {
                right = center + 1;
            }

            right = std::min(right, n_fft_bins - 1);

            for (int k = 0; k < center && k < n_fft_bins; ++k) {
                mel_basis.at<float>(i, k) = 
                    static_cast<float>(k - left) / static_cast<float>(center - left);
            }
            for (int k = center; k < right && k < n_fft_bins; ++k) {
                mel_basis.at<float>(i, k) = 
                    static_cast<float>(right - k) / static_cast<float>(right - center);
            }
        }
    }

    // 从一帧 PCM 中提取 Mel 特征
    cv::Mat extract(const std::vector<float>& pcm_frame) {
        if (pcm_frame.empty()) {
            return cv::Mat();
        }

        // 1. 准备输入
        // 创建 FFT 输入数组
        // input_frame 的大小固定为 [1, n_fft]
        // 如果 pcm_frame 长度小于 n_fft：后面自动补 0
        // 如果 pcm_frame 长度大于 n_fft：只复制前 n_fft 个采样点
        
        cv::Mat input_frame(1, n_fft, CV_32F, cv::Scalar(0.0f));

        int copy_len = std::min(static_cast<int>(pcm_frame.size()), n_fft);
        std::memcpy(input_frame.ptr<float>(0),
                    pcm_frame.data(),
                    copy_len * sizeof(float));
        
        // 2. FFT变换
        // OpenCV dft 需要复数输入时，通常用双通道矩阵表示。channel 0：实部，channel 1：虚部
        // 所以这里当输入为实数信号时，实部 = input_frame，虚部 = 全0

        cv::Mat planes[] =  {
            input_frame,
            cv::Mat::zeros(input_frame.size(), CV_32F)
        };

        cv::Mat complex_img;
        cv::merge(planes, 2, complex_img);
        cv::dft(complex_img, complex_img);

        cv::split(complex_img, planes);
        cv::magnitude(planes[0], planes[1], planes[0]);

        cv::Mat mag_spec = planes[0].colRange(0, n_fft / 2 + 1);

        // 3. 梅尔滤波
        cv::Mat mel_spec;
        cv::gemm(mag_spec, mel_basis, 1.0, cv::Mat(), 0.0, mel_spec, cv::GEMM_2_T);

        // 4. 归一化：输出 [0, 1]
        /**
         * 对每个 Mel bin 做后处理：
         * 1. 转 dB;
         * 2. 减去参考电平 ref_level_db;
         * 3. 裁剪最小动态范围 min_level_db;
         * 4. 归一化到 [0, 1];
         * 5. 映射到 [-max_abs_value, max_abs_value];
         * 6. 最终裁剪到 [-4, 4]
         */
        for (int i = 0; i < n_mels; ++i) {
            float val = mel_spec.at<float>(0, i);
            val = 20.0f * std::log10(std::max(val, 1e-5f));
            val -= ref_level_db;
            val = std::max(min_level_db, val);
            
            float norm = (val - min_level_db) / (-min_level_db);
            float sym = 2.0f * max_abs_value * norm - max_abs_value;
            sym = std::max(-max_abs_value, std::min(max_abs_value, sym));

            mel_spec.at<float>(0, i) = sym;
        }

        return mel_spec;

    }

    // 提取单帧 Mel 特征，并转换为 std::vector<float>
    std::vector<float> extractVector(const std::vector<float>& pcm_frame) {
        cv::Mat row = extract(pcm_frame);
        if (row.empty()) {
            return {};
        }

        return std::vector<float>(row.begin<float>(), row.end<float>());
    }
};

MelFeatureExtractor::MelFeatureExtractor(int sample_rate,
                                         int n_fft,
                                         int n_mels,
                                         float fmin,
                                         float fmax) 
    : pImpl(std::make_unique<Impl>(sample_rate, n_fft, n_mels, fmin, fmax)) {}

MelFeatureExtractor::~MelFeatureExtractor() = default;
MelFeatureExtractor::MelFeatureExtractor(MelFeatureExtractor&&) noexcept = default;
MelFeatureExtractor& MelFeatureExtractor::operator=(MelFeatureExtractor&&) noexcept = default;

cv::Mat MelFeatureExtractor::extract(const std::vector<float>& pcm_frame) {
    return pImpl->extract(pcm_frame);
}

std::vector<float> MelFeatureExtractor::extractVector(const std::vector<float>& pcm_frame) {
    return pImpl->extractVector(pcm_frame);
}

/**
 * @brief 批量提取 Mel 特征
 * @param frames 输入多帧 PCM: frames[num_frames][frame_size]
 * @return cv::Mat
 */
cv::Mat MelFeatureExtractor::extractBatch(const std::vector<std::vector<float>>& frames) {
    if (frames.empty()) {
        return cv::Mat();
    }

    cv::Mat batch_mels(static_cast<int>(frames.size()), pImpl->n_mels, CV_32F);

    for (size_t i = 0; i < frames.size(); ++i) {
        // 对每一帧单独提取 Mel 特征
        cv::Mat row = pImpl->extract(frames[i]);
        if (row.empty()) {
            // 如果某一帧无效，则该行填 0
            batch_mels.row(static_cast<int>(i)).setTo(0.0f);
        } else {
            row.copyTo(batch_mels.row(static_cast<int>(i)));
        }
    }
    return batch_mels;
}

int MelFeatureExtractor::getMelBins() const {
    return pImpl->n_mels;
}

int MelFeatureExtractor::getFftSize() const {
    return pImpl->n_fft;
}

int MelFeatureExtractor::getSampleRate() const {
    return pImpl->sample_rate;
}

} // namespace Audio
} // namespace DigitalHuman