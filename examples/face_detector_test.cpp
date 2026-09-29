#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "core/image_loader.h"
#include "core/face_detector.h"

using namespace DigitalHuman::Core;

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: Face Detection Test ===" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    try {
        // 1. 首先使用 ImageLoader 加载图像
        ImageLoader loader;
        cv::Mat img = loader.loadFromFile(image_path);
        std::cout << "Image Loaded: " << img.cols << "x" << img.rows << std::endl;

        // 如果是小图，强行放大 2-3 倍
        // dlib 官方建议：人脸至少要有 80x80 像素才能被稳定检测
        if (img.cols < 800) {
            std::cout << "[Info] Image is too small, upsampling 2x..." << std::endl;
            cv::resize(img, img, cv::Size(), 2.0, 2.0); // 放大 2 倍
        }

        // 2. 使用 FaceDetector 检测人脸
        FaceDetector detector;

        // 设置计时
        auto start = std::chrono::high_resolution_clock::now();

        std::vector<cv::Rect> faces = detector.detect(img);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Detection Complete." << std::endl;
        std::cout << "   -> Time Cost: " << duration << " ms" << std::endl;
        std::cout << "   -> Faces Found: " << faces.size() << std::endl;

        // 3. 绘制结果
        if (faces.empty()) {
            std::cout << "No faces detected!" << std::endl;
        } else {
            for (size_t i = 0; i < faces.size(); i++) {
                // 画矩形框
                cv::rectangle(img, faces[i], cv::Scalar(0, 255, 0), 2);

                // 写标签
                std::string label = "Face " + std::to_string(i);
                cv::putText(img, label, cv::Point(faces[i].x, faces[i].y - 5),
                     cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                
                std::cout << "   -> Face " << i << ": " << faces[i] << std::endl;
            }

            // 保存结果
            std::string save_path = "face_detect_result.jpg";
            cv::imwrite(save_path, img);
            std::cout << "[Result] Saved visualization to: " << save_path << std::endl;
        }

    } catch (const std::exception& e) { 
        std::cerr << "[Error] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}