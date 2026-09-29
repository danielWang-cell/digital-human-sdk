#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <ncnn/net.h> 

namespace DigitalHuman {
namespace Model {

/**
 * @brief 图像质量报告
 */
struct QualityReport {
    bool is_valid = true;
    double sharpness_score = 0.0; // 清晰度评分
    std::string message;
};

/**
 * @brief 模型输出处理器，负责将 ncnn 推理结果转换为图像，并进行融合与增强
 */
class OutputProcessor {
public:
    OutputProcessor();
    ~OutputProcessor();

    OutputProcessor(const OutputProcessor&) = delete;
    OutputProcessor& operator=(const OutputProcessor&) = delete;

    /**
     * @brief 核心处理函数
     * @param tensor 模型输出的张量 [1, 3, 96, 96]
     * @param original_face 原始对齐人脸 [96, 96] (用于背景融合)
     * @param mask 嘴部掩码 [96, 96] (单通道, 0~1 或 0~255)
     * @return cv::Mat 融合后的最终图像
     */
    cv::Mat process(const ncnn::Mat& tensor, const cv::Mat& original_face, const cv::Mat& mask);

    /**
     * @brief 验证最后一次输出的质量
     */
    QualityReport validateLastOutput() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Model
} // namespace DigitalHuman