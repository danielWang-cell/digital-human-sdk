#include <iostream>
#include <vector>
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

        // 检查关键个数
        if (landmarks.size() != 68) {
            std::cerr << "[FaceMaskGenerator] Error : Invalid landmarks count." << std::endl;
            return mask;
        }

        // 2. 提取嘴唇外轮廓 关键点 48-59
        std::vector<cv::Point> mouth_points;
        // 因为 dlib 嘴唇外圈关键点是 48-59
        for (int i = 48; i <= 59; ++i) {
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
            // 检查核的大小是否是奇数
            if (blur_ksize % 2 == 0 ){
                blur_ksize++;
            }

            cv::GaussianBlur(mask, mask, cv::Size(blur_ksize, blur_ksize), 0);
        }

        // 6. 归一化并转浮点（0-255 -> 0.0-1.0）
        cv::Mat mask_float;
        mask.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);

        return mask_float;
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
                                             int blur_kernel_size) {
    return pImpl->generateMouthMask(image_size, landmarks, dilate_radius, blur_kernel_size);
}

}
}
