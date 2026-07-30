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


/**
 * @brief 加载关键点检测模型
 * @param modelPath.dat 文件路径
 * @return 是否加载成功
 */
    bool loadLandmarkModel(const std::string& modelPath);

/**
 * @brief
 * @param image 原始图像
 * @param faceRect 人脸框（由 detect 返回）
 * @return 68个点的坐标列表（如果失败返回空）
 */
    std::vector<cv::Point> getLandmarks(const cv::Mat& image, const cv::Rect& faceRect);
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
 

}
}