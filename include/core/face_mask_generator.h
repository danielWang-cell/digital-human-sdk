#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Core {

/**
 * @brief 人脸掩码生成器，用于生成嘴部区域的混合掩码 (Mask)
 */
class FaceMaskGenerator {
public:
    FaceMaskGenerator();
    ~FaceMaskGenerator();

    FaceMaskGenerator(FaceMaskGenerator&&) noexcept;
    FaceMaskGenerator& operator=(FaceMaskGenerator&&) noexcept;

    FaceMaskGenerator(const FaceMaskGenerator&) = delete;
    FaceMaskGenerator& operator=(const FaceMaskGenerator&) = delete;

    /**
     * @brief 生成嘴部掩码
     * @param image_size 原始图像大小 ,因为生成的掩码需与原图一致
     * @param landmarks 68个关键点
     * @param dilate_radius 扩展半径 (像素)，默认 0 表示不扩展
     * @param blur_sigma 羽化程度 (高斯模糊核大小)，必须是奇数
     * @return cv::Mat 单通道掩码 (CV_32FC1, 范围 0.0-1.0)
     */
    cv::Mat generateMouthMask(const cv::Size& image_size,
                              const std::vector<cv::Point>& landmarks,
                              int dilate_radius = 5,
                              int blur_sigma = 15);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Core
} // namespace DigitalHuman