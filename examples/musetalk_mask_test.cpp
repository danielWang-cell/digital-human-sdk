#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "core/image_loader.h"
#include "core/face_detector.h"
#include "core/face_aligner.h"
#include "core/face_mask_generator.h"
#include "core/musetalk_engine.h"
#include <cmath>

using namespace DigitalHuman::Core;

// 辅助函数，计算中心点
cv::Point2f get_center(const std::vector<cv::Point>& points) {
    float x = 0, y = 0;
    for (const auto& p : points) {
        x += p.x;
        y += p.y;
    }
    return cv::Point2f(x / points.size(), y / points.size());
}

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: MuseTalk Mask Integration Test ===" << std::endl;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <model_dir>" << std::endl;
        return -1;
    }
    std::string img_path = argv[1];
    std::string model_dir = argv[2];
    std::string dlib_model = model_dir + "/shape_predictor_68_face_landmarks.dat";

    try {
        // 1. 初始化
        MuseTalkEngine engine;
        if (!engine.init(model_dir)) {
            std::cerr << "[Error] Failed to initialize engine." << std::endl;
            return -1;
        }
        
        FaceDetector detector;
        if (!detector.loadLandmarkModel(dlib_model)) {
            std::cerr << "[Error] Failed to load dlib model." << std::endl;
            return -1;
        }

        FaceAligner aligner;
        FaceMaskGenerator mask_gen;
        ImageLoader loader;

        // 2. 加载原图
        cv::Mat raw_img = loader.loadFromFile(img_path);
        if (raw_img.empty()) {
            std::cerr << "[Error] Failed to load image: " << img_path << std::endl;
            return -1;
        }
        
        // 3. 第一次检测 (在原图上)
        std::cout << "[Step] Detecting face on raw image..." << std::endl;
        auto faces = detector.detect(raw_img);
        if (faces.empty()) { 
            std::cerr << "No face found." << std::endl; 
            return -1; 
        }
        
        auto landmarks_raw = detector.getLandmarks(raw_img, faces[0]);
        if (landmarks_raw.empty()) { 
            std::cerr << "No landmarks." << std::endl; 
            return -1; 
        }

        // 4. 执行对齐
        cv::Mat aligned_float = aligner.align(raw_img, landmarks_raw, 256);
        cv::Mat aligned_face;
        aligned_float.convertTo(aligned_face, CV_8UC3, 127.5, 127.5);

        // 5. 关键点坐标映射
        std::cout << "[Step] Mapping landmarks to aligned space..." << std::endl;
        std::vector<cv::Point> left_eye, right_eye;
        for (int i=36; i<=41; ++i) left_eye.push_back(landmarks_raw[i]);
        for (int i=42; i<=47; ++i) right_eye.push_back(landmarks_raw[i]);
        cv::Point2f l_c = get_center(left_eye);
        cv::Point2f r_c = get_center(right_eye);

        float dy = r_c.y - l_c.y;
        float dx = r_c.x - l_c.x;
        double angle = std::atan2(dy, dx) * 180.0 / CV_PI;
        double scale = (256 * 0.4) / std::sqrt(dx*dx + dy*dy);
        cv::Point2f eyes_center = (l_c + r_c) * 0.5f;

        cv::Mat M = cv::getRotationMatrix2D(eyes_center, angle, scale);
        M.at<double>(0, 2) += (256*0.5 - eyes_center.x);
        M.at<double>(1, 2) += (256*0.4 - eyes_center.y);

        std::vector<cv::Point> landmarks_aligned;
        for(const auto& p : landmarks_raw) {
            double x = M.at<double>(0,0)*p.x + M.at<double>(0,1)*p.y + M.at<double>(0,2);
            double y = M.at<double>(1,0)*p.x + M.at<double>(1,1)*p.y + M.at<double>(1,2);
            landmarks_aligned.emplace_back(std::round(x), std::round(y));
        }

        // 6. 生成掩码
        std::cout << "[Step] Generating mask..." << std::endl;
        cv::Mat mask_float = mask_gen.generateMouthMask(aligned_face.size(), landmarks_aligned, 5, 15);
        cv::Mat mask;
        mask_float.convertTo(mask, CV_8UC1, 255.0);

        // 7. 准备音频特征
        cv::Mat dummy_audio(50, 384, CV_32F);
        cv::randu(dummy_audio, cv::Scalar(0.0f), cv::Scalar(1.0f));

        // 8. 推理
        std::cout << "[Step] Running MuseTalk..." << std::endl;
        cv::Mat result = engine.run(aligned_face, dummy_audio, mask);

        if (result.empty()) {
            std::cerr << "[Error] MuseTalk inference failed (result is empty). Check logs above." << std::endl;
            return -1;
        }

        cv::imwrite("step3_final_result.jpg", result);
        std::cout << "[Success] Result saved to step3_final_result.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}