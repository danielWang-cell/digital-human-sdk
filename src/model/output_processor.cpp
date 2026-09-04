#include <algorithm>
#include <vector>
#include <spdlog/spdlog.h>

#include <opencv2/imgproc.hpp>

#include "model/output_processor.h"

namespace DigitalHuman {
namespace Model {

struct OutputProcessor::Impl {
    // 最近一次成功或失败后记录的输出图像
    cv::Mat last_result;

    // 最近一次输出图像的质量检查报告
    QualityReport last_report;

    // 重置并记录空输出状态
    void markEmptyOutput(const std::string& message) {
        last_result = cv::Mat();
        last_report.is_valid = false;
        last_report.sharpness_score = 0.0;
        last_report.message = message;
    }

    // 对输出图像进行轻量级质量检查
    void checkQuality(const cv::Mat& image) {
        if (image.empty()) {
            markEmptyOutput("Empty output");
            return;
        }

        last_report.is_valid = true;
        last_report.message = "OK";
        last_report.sharpness_score = 0.0;

        cv::Mat gray;
        cv::Mat lap;

        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);  // 生成灰度图 gray
        cv::Laplacian(gray, lap, CV_64F);               // 拉普拉斯边缘检测，lap 为响应图

        cv::Scalar mean;                                // 均值
        cv::Scalar stddev;                              // 标准差
        cv::meanStdDev(lap, mean, stddev);

        const double variance = stddev.val[0] * stddev.val[0];  // 计算拉普拉斯方差
        last_report.sharpness_score = variance;

        // 50.0 是经验阈值，只用于提示输出是否可能过于模糊
        if (variance < 50.0) {
            last_report.message = "Low sharpness";
        }
    }

    // 执行实际的 tensor 到图像转换
    // 这里假设模型输出已经是 Wav2Lip 所需的 96x96x3 格式
    /**
     * @brief 将 ncnn 输出 tensor 转换为 OpenCV BGR 图像
     * @param tensor 模型输出张量
     * @return cv::Mat 96 x 96 BGR 图像，输入非法时返回空 Mat 
     */
    cv::Mat tensorToImage(const ncnn::Mat& tensor) {
        if (tensor.empty()) {
            markEmptyOutput("Empty tensor");
            return cv::Mat();
        }

        if (tensor.w != 96 || tensor.h != 96 || tensor.c != 3) {
            spdlog::error("[OutputProcessor] Unexpected tensor shape {}x{}x{}, expected 96x96x3",
                    tensor.w, tensor.h, tensor.c);

            markEmptyOutput("Unexpected tensor shape");
            return cv::Mat();
        }

        cv::Mat image(96, 96, CV_8UC3);

        const float* b_channel = tensor.channel(0);
        const float* g_channel = tensor.channel(1);
        const float* r_channel = tensor.channel(2);

        for (int y = 0; y < 96; ++y) {
            cv::Vec3b* row = image.ptr<cv::Vec3b>(y);

            for (int x = 0; x < 96; ++x) {
                const int idx = y * 96 + x;
                
                // 模型输出理论范围是 [-1, 1]（Wav2Lip 输出为 tanh）
                // 防止极越界值造成颜色溢出
                const float b = std::clamp(b_channel[idx], 0.0f, 1.0f);
                const float g = std::clamp(g_channel[idx], 0.0f, 1.0f);
                const float r = std::clamp(r_channel[idx], 0.0f, 1.0f);

                row[x] = cv::Vec3b(
                    static_cast<unsigned char>(b * 255.0f + 0.5f),
                    static_cast<unsigned char>(g * 255.0f + 0.5f),
                    static_cast<unsigned char>(r * 255.0f + 0.5f)
                );
            }
        }

        return image;
    }
    
    // 增强图像中高频的边缘部分，从而提升细节对比度和清晰度
    void sharpenImage(cv::Mat& img) {
        if (img.empty()) {
            return;
        }
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            0, -1,  0,
           -1,  5, -1,
            0, -1,  0);
        cv::filter2D(img, img, img.depth(), kernel);
    }

    void correctColor(cv::Mat& gen, const cv::Mat& org) {
        if (gen.empty() || org.empty()) {
            return;
        }
        if (gen.size() != org.size()) {
            return;
        }

        // 1. 转为 32 位浮点型并转换到 Lab 色彩空间
        cv::Mat gen_f, org_f;
        gen.convertTo(gen_f, CV_32FC3, 1.0 / 255.0);
        org.convertTo(org_f, CV_32FC3, 1.0 / 255.0);

        cv::Mat gen_lab, org_lab;
        cv::cvtColor(gen_f, gen_lab, cv::COLOR_BGR2Lab);
        cv::cvtColor(org_f, org_lab, cv::COLOR_BGR2Lab);

        // 2. 分别计算 gen 和 org 在 Lab 空间下的均值与标准差
        cv::Scalar mean_gen, std_gen;
        cv::Scalar mean_org, std_org;
        cv::meanStdDev(gen_lab, mean_gen, std_gen);
        cv::meanStdDev(org_lab, mean_org, std_org);

        // 3. 通道分离
        std::vector<cv::Mat> gen_channels(3);
        cv::split(gen_lab, gen_channels);

        // 4. 对 L, a, b 每个通道按公式做线性映射
        for (int i = 0; i < 3; ++i) {
            // 防止除以 0
            float s_gen = static_cast<float>(std_gen.val[i]);
            float s_org = static_cast<float>(std_org.val[i]);
            if (s_gen < 1e-6f) s_gen = 1e-6f;

            // C_out = (C_gen - mean_gen) * (std_org / std_gen) + mean_org
            gen_channels[i] = (gen_channels[i] - mean_gen.val[i]) * (s_org / s_gen) + mean_org.val[i];
        }

        // 5. 通道合并并转回 BGR 色彩空间
        cv::Mat corrected_lab;
        cv::merge(gen_channels, corrected_lab);

        cv::Mat corrected_f;
        cv::cvtColor(corrected_lab, corrected_f, cv::COLOR_Lab2BGR);

        // 6. 转回 8 位无符号 BGR 图像 (自带 0-255 饱和截断)
        corrected_f.convertTo(gen, CV_8UC3, 255.0);
    }


    cv::Mat blendImages(const cv::Mat& gen, const cv::Mat& org, const cv::Mat& mask) {
        if (gen.empty()) {
            return cv::Mat();
        }

        if (org.empty() || mask.empty()) {
            return gen.clone();
        }

        cv::Mat mask_f;
        if (mask.type() != CV_32FC1) {
            mask.convertTo(mask_f, CV_32F, 1.0 / 255.0);
        } else {
            mask_f = mask;
        }

        cv::Mat mask_3c;
        cv::cvtColor(mask_f, mask_3c, cv::COLOR_GRAY2BGR);

        cv::Mat gen_f, org_f;
        gen.convertTo(gen_f, CV_32F);
        org.convertTo(org_f, CV_32F);

        cv::Mat part1 = gen_f.mul(mask_3c);
        cv::Mat part2 = org_f.mul(cv::Scalar::all(1.0f) - mask_3c);

        cv::Mat result_f;
        cv::add(part1, part2, result_f);

        cv::Mat result;
        result_f.convertTo(result, CV_8UC3);
        return result;

    }
 
};

OutputProcessor::OutputProcessor() : pImpl(std::make_unique<Impl>()) {}
OutputProcessor::~OutputProcessor() = default;

OutputProcessor::OutputProcessor(OutputProcessor&&) noexcept = default;
OutputProcessor& OutputProcessor::operator=(OutputProcessor&&) noexcept = default;

cv::Mat OutputProcessor::process(const ncnn::Mat& tensor,
                                 const cv::Mat& original_face,
                                 const cv::Mat& mask) {
    cv::Mat generated = pImpl->tensorToImage(tensor);
    if (generated.empty()) {
        return cv::Mat();
    }

    // 先关掉锐化，避免把异常输出进一步放大
    // pImpl->sharpenImage(generated);
    
    // 颜色矫正：把生成图的色彩分布对齐原脸，避免融合处出现色差/接缝
    if (!original_face.empty()) {
        pImpl->correctColor(generated, original_face);
    }

    if (!original_face.empty() && !mask.empty()) {
        pImpl->last_result = pImpl->blendImages(generated, original_face, mask);
    } else {
        pImpl->last_result = generated;
    }

    pImpl->checkQuality(pImpl->last_result);
    return pImpl->last_result;
}

QualityReport OutputProcessor::validateLastOutput() const {
    return pImpl->last_report;
}


}
}