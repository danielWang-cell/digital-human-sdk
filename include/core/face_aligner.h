#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Core {

/**
 * @brief 人脸对齐与预处理模块，负责将检测到的人脸矫正为模型所需的标准格式
 */
class FaceAligner {
public:
    FaceAligner();
    ~FaceAligner();

    FaceAligner(FaceAligner&&) noexcept;
    FaceAligner& operator=(FaceAligner&&) noexcept;

    FaceAligner(const FaceAligner&) = delete;
    FaceAligner& operator=(const FaceAligner&) = delete;

    /**
     * @brief 对齐并裁剪人脸 (Wav2Lip 标准预处理)
     * @param image 原始图像 (BGR)
     * @param landmarks 68个关键点 (由 FaceDetector 获取)
     * @param target_size 目标输出尺寸 (默认 96x96)
     * @return cv::Mat 处理后的图像 (CV_32FC3, 数值范围 [-1, 1])
     */
    cv::Mat align(const cv::Mat& image, const std::vector<cv::Point>& landmarks, int target_size = 96);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Core
} // namespace DigitalHuman