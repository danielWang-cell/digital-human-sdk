#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "core/image_loader.h"
#include "core/face_detector.h"

using namespace DigitalHuman::Core;

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: Landmark Debug Test ===" << std::endl;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <model_path>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];
    std::string model_path = argv[2];

    try {
        ImageLoader loader;
        cv::Mat img = loader.loadFromFile(image_path);
        
        FaceDetector detector;
        if (!detector.loadLandmarkModel(model_path)) {
            std::cerr << "[Critical] Failed to load model!" << std::endl;
            return -1;
        }

        // 1. 检测人脸 (Detect)
        auto faces = detector.detect(img);
        std::cout << ">>> Step 1: Detection Result: " << faces.size() << " faces found." << std::endl;

        if (faces.empty()) {
            std::cout << "[Warning] No faces detected! Check if image is too dark or faces are too small." << std::endl;
        }

        // 2. 遍历结果 (Iterate)
        for (size_t i = 0; i < faces.size(); ++i) {
            std::cout << "   Processing Face " << i << ": " << faces[i] << std::endl;

            // 无论如何，先画出人脸框 (蓝色)
            // 如果你能看到蓝框，说明检测器没问题
            cv::rectangle(img, faces[i], cv::Scalar(255, 0, 0), 2);

            // 3. 提取关键点 (Landmark)
            auto landmarks = detector.getLandmarks(img, faces[i]);
            
            if (landmarks.empty()) {
                std::cout << "      -> [Warning] Landmark extraction failed (maybe face too small)." << std::endl;
                // 继续下一个循环，不要 continue 跳过画图步骤
            } else {
                std::cout << "      -> [Success] " << landmarks.size() << " points extracted." << std::endl;

                // 画所有点 (绿色)
                for (const auto& p : landmarks) {
                    cv::circle(img, p, 2, cv::Scalar(0, 255, 0), -1);
                }

                // 高亮嘴部 (红色, 48-67)
                // Outer Lip
                for (int k = 48; k < 59; ++k) 
                    cv::line(img, landmarks[k], landmarks[k+1], cv::Scalar(0, 0, 255), 2);
                cv::line(img, landmarks[59], landmarks[48], cv::Scalar(0, 0, 255), 2);

                // Inner Lip
                for (int k = 60; k < 67; ++k) 
                    cv::line(img, landmarks[k], landmarks[k+1], cv::Scalar(0, 0, 255), 2);
                cv::line(img, landmarks[67], landmarks[60], cv::Scalar(0, 0, 255), 2);
            }
        }

        // 保存结果
        std::string out_name = "landmark_debug_result.jpg";
        cv::imwrite(out_name, img);
        std::cout << "[Success] Visualization saved to: " << out_name << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Fatal Error] " << e.what() << std::endl;
        return -1;
    }

    return 0;
}