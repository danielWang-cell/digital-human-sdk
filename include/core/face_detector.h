#pragma once

#include <vector>
#include <memory>
#include <string>

#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Core {


/**
 * @brief FaceDetector 人脸检测器
 */

class FaceDetector {

public:
    FaceDetector();
    ~FaceDetector();
    
    //禁用拷贝 (dlib 检测器拷贝成本较高，且 unique_ptr 默认不可拷贝)
    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;

    FaceDetector(FaceDetector&&) noexcept;
    FaceDetector& operator=(FaceDetector&&) noexcept;
/**
 * @brief detect 检测图像中的人脸
 * @param image 输入 BGR 格式的图像
 * @return std::vector<cv::Rect> 检测到的人脸边界框列表。如果没有人脸返回空列表   
 */
    std::vector<cv::Rect> detect(const cv::Mat& image);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
 

}
}