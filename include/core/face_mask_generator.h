#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Core {

/**
 * @brief 人脸 mask 生成器
 * mask 的作用：Wav2Lip 生成的是一张 96 x 96 的新嘴部图像，但是我们不希望整张脸都被替换，
 * 只希望嘴巴附近被替换。
 * 所以需要生成一个 alpha mask：
 *          alpha = 1.0 的区域使用生成图；
 *          alpha = 0.0 的区域保留原图；
 *          alpha 介于 0~1 的区域做柔和融合。
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
     * @param image_size 原始图像大小，因为生成的掩码需与原图一致
     * @param landmarks 68 个关键点
     * @param dilate_radius 扩展半径（像素），默认 0 表示不扩展
     * @param blur_sigma 羽化程度（高斯模糊大小），必须是奇数
     * @return cv::Mat 单通道掩码（CV_32FC1，范围 0.0-1.0）
     */

    cv::Mat generateMouthMask(const cv::Size& image_size,
                               const std::vector<cv::Point>& landmarks,
                               int dilate_radius = 5,
                               int blur_sigma = 15);

    /**
     * @brief 生成 96 x 96 对齐空间下的精细嘴部 alpha mask
     * @param landmarks_96 已经映射到 96 x 96 空间的 68 点
     * @return CV_32FC1，范围 0.0~1.0
     */
    cv::Mat generatePreciseMouthAlphaMask96(const std::vector<cv::Point2f>& landmarks_96);


    /**
     * @brief 把单通道 alpha mask 转成三通道 mask
     * 图像融合时，BGR 图像有 3 个通道，所以 alpha mask 也需要扩展成 3 通道，方便逐通道相乘
     */
    cv::Mat to3ChannelMask(const cv::Mat& alpha_mask)const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Core
} // namespace DigitalHuman
