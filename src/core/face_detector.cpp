#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "core/face_detector.h"

namespace DigitalHuman {
namespace Core {

// Impl
struct FaceDetector::Impl {
    // dlib 的人脸检测器
    dlib::frontal_face_detector detector;
    
    // 形状检测器
    dlib::shape_predictor landmarks_predictor;  
    bool is_model_loaded = false;

    Impl() {
        // 初始化检测器
        detector = dlib::get_frontal_face_detector();
    }

    // 内部实现函数 
    std::vector<cv::Rect> detect(const cv::Mat& image) {
        std::vector<cv::Rect> results;

        if (image.empty()) {
            std::cerr << "[FaceDetector] Warning: Input image is empty." << std::endl;
            return results;
        }

        // 做降采样策略
        cv::Mat process_img;
        float scale = 1.0f;
        const int MAX_WIDTH = 1600; // 限制最大宽度为 1600px
        const int MIN_WIDTH = 800; // 低清图上采样阈值

        // 降采样，处理过大的图片
        if (image.cols > MAX_WIDTH) {
            scale = (float)MAX_WIDTH / image.cols;
            cv::resize(image, process_img, cv::Size(), scale, scale);
        } else if (image.cols < MIN_WIDTH) {
            // 图片太小，放大以提高检出率
            scale = (float)MIN_WIDTH / image.cols;
            cv::resize(image, process_img, cv::Size(), scale, scale);
        } else {
            // 尺寸合适，直接使用
            process_img = image;
        }
        // 1.将 OpenCV 图像包装为 dlib 图像
        try {
            dlib::cv_image<dlib::bgr_pixel> dlib_img(process_img);

            // 执行检测
            std::vector<dlib::rectangle> dets = detector(dlib_img, 0);
            
            // 如果放大后还没检测到，再尝试内部 Pyramid Up
            if (dets.empty() && process_img.cols < 1200) {
                // std::cout << "[FaceDetector] Still no face, trying internal upsampling..." << std::endl;
                dets = detector(dlib_img, 1);
            }

            std::cout << "[FaceDetector] Raw dlib detections: " << dets.size() << std::endl;

            results.reserve(dets.size());

            for (const auto& d : dets) {
                // 将坐标映射回原图尺寸
                int x = static_cast<int>(d.left() / scale); 
                int y = static_cast<int>(d.top() / scale); 
                int w = static_cast<int>(d.width() / scale); 
                int h = static_cast<int>(d.height() / scale); 

                // 边界保护
                x = std::max(0, x);
                y = std::max(0, y);
                w = std::min(image.cols - x, w);
                h = std::min(image.rows - y, h);

                if (w > 0 && h > 0) {
                    results.emplace_back(x, y, w, h);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[FaceDetector] Error: " << e.what() << std::endl;
        }

        return results;
    }
};

FaceDetector::FaceDetector() : pImpl(std::make_unique<Impl>()){}
FaceDetector::~FaceDetector() = default;

FaceDetector::FaceDetector(FaceDetector&&) noexcept = default;
FaceDetector& FaceDetector::operator=(FaceDetector&&) noexcept = default;

std::vector<cv::Rect> FaceDetector::detect(const cv::Mat& image) {
    return pImpl->detect(image);
}

}
}