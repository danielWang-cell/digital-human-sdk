#include <iostream>
#include <opencv2/highgui.hpp>
#include "core/image_loader.h"
#include "core/face_detector.h"
#include "core/face_aligner.h" // 引入新模块

using namespace DigitalHuman::Core;

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: Face Alignment Test ===" << std::endl;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <model_path>" << std::endl;
        return -1;
    }

    try {
        // 1. 加载
        ImageLoader loader;
        cv::Mat img = loader.loadFromFile(argv[1]);
        
        // 2. 检测
        FaceDetector detector;
        detector.loadLandmarkModel(argv[2]);
        auto faces = detector.detect(img);
        
        if (faces.empty()) {
            std::cerr << "No face detected." << std::endl;
            return -1;
        }

        auto landmarks = detector.getLandmarks(img, faces[0]);
        if (landmarks.empty()) {
            std::cerr << "No landmarks detected." << std::endl;
            return -1;
        }

        // 3. 对齐 (测试核心)
        FaceAligner aligner;
        int target_size = 96;
        
        // aligned_img 是 float 类型，范围 [-1, 1]
        cv::Mat aligned_img = aligner.align(img, landmarks, target_size);

        std::cout << "[Result] Aligned Image Info:" << std::endl;
        std::cout << "   Size: " << aligned_img.cols << "x" << aligned_img.rows << std::endl;
        std::cout << "   Type: " << aligned_img.type() << " (CV_32FC3)" << std::endl;

        // 4. 可视化保存 (反归一化)
        // 从 [-1, 1] 变回 [0, 255] 才能正确保存为 jpg
        cv::Mat vis_img;
        aligned_img.convertTo(vis_img, CV_8U, 127.5, 127.5); // x * 127.5 + 127.5

        cv::imwrite("aligned_face_96x96.jpg", vis_img);
        std::cout << "[Success] Saved visualization to aligned_face_96x96.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}