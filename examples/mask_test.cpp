#include <iostream>
#include <opencv2/highgui.hpp>
#include "core/image_loader.h"
#include "core/face_detector.h"
#include "core/face_mask_generator.h" 

using namespace DigitalHuman::Core;

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: Face Mask Test ===" << std::endl;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <model_path>" << std::endl;
        return -1;
    }

    try {
        // 1. 准备数据
        ImageLoader loader;
        cv::Mat img = loader.loadFromFile(argv[1]);
        
        FaceDetector detector;
        detector.loadLandmarkModel(argv[2]);
        auto faces = detector.detect(img);
        
        if (faces.empty()) {
            std::cerr << "No face detected." << std::endl;
            return -1;
        }

        auto landmarks = detector.getLandmarks(img, faces[0]);

        // 2. 生成掩码
        FaceMaskGenerator mask_gen;
        
        // 参数：扩展半径=10px, 羽化核=25px (数值越大越模糊)
        cv::Mat mask = mask_gen.generateMouthMask(img.size(), landmarks, 10, 25);

        std::cout << "[Success] Mask generated." << std::endl;
        std::cout << "   Type: " << mask.type() << " (CV_32FC1)" << std::endl;

        // 3. 可视化保存
        // 因为 mask 是 0.0-1.0 的 float，保存前需要转回 0-255
        cv::Mat save_mask;
        mask.convertTo(save_mask, CV_8UC1, 255.0);
        cv::imwrite("mask_visualization.jpg", save_mask);

        // 4. 模拟融合效果 (把嘴巴变蓝)
        // 原理: Out = Img * (1-Mask) + Blue * Mask
        cv::Mat result = img.clone();
        for (int y = 0; y < img.rows; ++y) {
            for (int x = 0; x < img.cols; ++x) {
                float m = mask.at<float>(y, x); // 0.0 ~ 1.0
                if (m > 0.01) {
                    cv::Vec3b& pixel = result.at<cv::Vec3b>(y, x);
                    // 简单的线性插值混合
                    pixel[0] = pixel[0] * (1 - m) + 255 * m; // B (Blue channel boost)
                    pixel[1] = pixel[1] * (1 - m);           // G
                    pixel[2] = pixel[2] * (1 - m);           // R
                }
            }
        }
        cv::imwrite("mask_blend_test.jpg", result);
        std::cout << "[Success] Blend result saved to mask_blend_test.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}