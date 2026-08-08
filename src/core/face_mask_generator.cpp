#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/imgproc.hpp>

#include "core/face_mask_generator.h"

namespace DigitalHuman {
namespace Core {

struct FaceMaskGenerator::Impl {
    // 实现内部的掩码生成函数
    cv::Mat generateMouthMask(const cv::Size& image_size,
                              const std::vector<cv::Point>& landmarks,
                              int dilate_radius,
                              int blur_ksize) 
    {
        // 1. 首先初始化全黑画布 CV_8UC1
        cv::Mat mask = cv::Mat::zeros(image_size, CV_8UC1);

        // 检查关键点个数
        if (landmarks.size() != 68) {
            std::cerr << "[FaceMaskGenerator] Error : Invalid landmarks count." << std::endl;
            return mask;
        }

        // 2. 提取嘴唇外轮廓 关键点 48-59
        std::vector<cv::Point> mouth_points;
        // 因为 dlib 嘴唇外圈关键点是 48-59
        for (int i = 48; i <= 59; i++) {
            mouth_points.push_back(landmarks[i]);
        }

        // 3. 绘制挖空的区域，必须是指针的指针，因为 fillPoly 支持一次画多个多边形
        const cv::Point* ppt[1] = {mouth_points.data()};
        int npt[] = {static_cast<int>(mouth_points.size())};
        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(255));

        // 4. 区域扩展，这里是扩展嘴部区域
        if (dilate_radius > 0) {
            // 首先需要获取卷积核
            int k_size = dilate_radius * 2 + 1;
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k_size, k_size));
            cv::dilate(mask, mask, kernel);
        }
        
        // 5. 边缘羽化，无缝衔接
        if (blur_ksize > 0) {
            // 检查核的大小是否为奇数
            if (blur_ksize % 2 == 0) {
                blur_ksize++;
            }
            cv::GaussianBlur(mask, mask, cv::Size(blur_ksize, blur_ksize), 0);
        }

        // 6. 归一化并转浮点（0-255 -> 0.0-1.0）
        cv::Mat mask_float;
        mask.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);

        return mask_float;
    }
    
    cv::Mat generatePreciseMouthAlphaMask96(const std::vector<cv::Point2f>& landmarks_96) {
        const cv::Size mask_size(96, 96);
        cv::Mat mask_u8 = cv::Mat::zeros(mask_size, CV_8UC1);

        if (landmarks_96.size() < 68) {
            std::cerr << "[FaceMaskGenerator] Invalid 96 landmarks count."
                      << landmarks_96.size() << std::endl;
            cv::Mat empty_mask;
            mask_u8.convertTo(empty_mask, CV_32FC1, 1.0 / 255.0);
            return empty_mask;
        }

        std::vector<cv::Point> mouth_points;
        mouth_points.reserve(20);

        // 初始化嘴部外接范围
        float mouth_x_min = landmarks_96[48].x;
        float mouth_x_max = landmarks_96[48].x;
        float mouth_y_min = landmarks_96[48].y;
        float mouth_y_max = landmarks_96[48].y;

        // 48-67 包含嘴部外圈和内圈
        // 使用完整嘴部点可以更准确覆盖开口区域
        for (int i = 48; i <= 67; ++i) {
            float x = landmarks_96[i].x;
            float y = landmarks_96[i].y;

            mouth_points.emplace_back(
                static_cast<int>(std::round(x)),
                static_cast<int>(std::round(y))
            );

            mouth_x_min = std::min(mouth_x_min, x);
            mouth_x_max = std::max(mouth_x_max, x);
            mouth_y_min = std::min(mouth_y_min, y);
            mouth_y_max = std::max(mouth_y_max, y);
        }

        // 使用凸包包住所有嘴部点
        // 凸包比直接 fillpoly 更稳定，尤其嘴巴张开时
        std::vector<cv::Point> hull;
        cv::convexHull(mouth_points, hull);

        if (hull.size() < 3) {
            cv::Mat empty_mask;
            mask_u8.convertTo(empty_mask, CV_32FC1, 1.0 / 255.0);
            return empty_mask;
        }

        cv::fillConvexPoly(mask_u8, hull, cv::Scalar(255));

        // 适度扩张嘴部区域
        // 不能扩太多，否则生成区域会覆盖下巴
        cv::Mat kernal = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(5, 5)
        );

        cv::dilate(mask_u8, mask_u8, kernal, cv::Point(-1, -1), 1);

        // 计算嘴巴宽高，用于限制 mask 扩散范围
        const float mouth_w = std::max(1.0f, mouth_x_max - mouth_x_min);
        const float mouth_h = std::max(1.0f, mouth_y_max - mouth_y_min);

        int x_left = static_cast<int>(std::floor(mouth_x_min - 0.55f * mouth_w));
        int x_right = static_cast<int>(std::ceil(mouth_x_max + 0.55f * mouth_w));

        int y_top = static_cast<int>(std::floor(mouth_y_min - 0.90f * mouth_h));
        int y_bottom = static_cast<int>(std::floor(mouth_y_max + 1.00f * mouth_h));

        // 限制在 96 x 96 图像内部
        x_left = std::max(0, x_left);
        x_right = std::min(96, x_right);
        y_top = std::max(0, y_top);
        y_bottom = std::min(96, y_bottom);

        // 清除限制范围外的 mask
        if (x_left > 0) {
            mask_u8.colRange(0, x_left).setTo(0);
        }
        if (x_right + 1 < 96) {
            mask_u8.colRange(x_right + 1, 96).setTo(0);
        }
        if (y_top > 0) {
            mask_u8.rowRange(0, y_top).setTo(0);
        }
        if (y_bottom + 1 < 96) {
            mask_u8.rowRange(y_bottom + 1, 96).setTo(0);
        }

        // 清除边缘，防止 mask 接触 96 x 96 边界
        // 如果 mask 接触边界，反变换回原图时容易出现边框线
        const int border = 2;
        mask_u8.rowRange(0, border).setTo(0);
        mask_u8.rowRange(96 - border, 96).setTo(0);
        mask_u8.colRange(0, border).setTo(0);
        mask_u8.colRange(96 - border, 96).setTo(0);

        // 羽化边缘，让融合更自然
        cv::GaussianBlur(mask_u8, mask_u8, cv::Size(13, 13), 0);

        // 模糊后边缘可能又有一点扩散，再清一次边界
        mask_u8.rowRange(0, border).setTo(0);
        mask_u8.rowRange(96 - border, 96).setTo(0);
        mask_u8.colRange(0, border).setTo(0);
        mask_u8.colRange(96 - border, 96).setTo(0);

        cv::Mat mask_f;
        mask_u8.convertTo(mask_f, CV_32FC1, 1.0 / 255.0);
        
        return mask_f;
    }
    
    cv::Mat to3ChannelMask(const cv::Mat& alpha_mask) const {
        if (alpha_mask.empty()) {
            return cv::Mat();
        }

        cv::Mat alpha_f;

        // 如果输入是 8 位图，先归一化到 0.0-1.0
        if (alpha_mask.type() != CV_32FC1) {
            alpha_mask.convertTo(alpha_f, CV_32FC1, 1.0 / 255.0);
        } else {
            alpha_f = alpha_mask;
        }

        std::vector<cv::Mat> channels(3, alpha_f);

        cv::Mat mask_3c;
        cv::merge(channels, mask_3c);
        return mask_3c;
    }
};

FaceMaskGenerator::FaceMaskGenerator() : pImpl(std::make_unique<Impl>()){}
FaceMaskGenerator::~FaceMaskGenerator() = default;

FaceMaskGenerator::FaceMaskGenerator(FaceMaskGenerator&&) noexcept = default;
FaceMaskGenerator& FaceMaskGenerator::operator=(FaceMaskGenerator&&) noexcept = default;

// 外部调用接口
cv::Mat FaceMaskGenerator::generateMouthMask(const cv::Size& image_size,
                                            const std::vector<cv::Point>& landmarks,
                                            int dilate_radius,
                                            int blur_kernal_size) 
    {
    return pImpl->generateMouthMask(image_size, landmarks, dilate_radius, blur_kernal_size);

    }
cv::Mat FaceMaskGenerator::generatePreciseMouthAlphaMask96(const std::vector<cv::Point2f>& landmarks_96) {
    return pImpl->generatePreciseMouthAlphaMask96(landmarks_96);
}

cv::Mat FaceMaskGenerator::to3ChannelMask(const cv::Mat& alpha_mask) const {
    return pImpl->to3ChannelMask(alpha_mask);
}

}
}