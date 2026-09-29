#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "core/musetalk_engine.h"

using namespace DigitalHuman::Core;

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: MuseTalk Validity Check ===" << std::endl;

    // 1. 设置模型路径
    std::string model_dir = "/workspace/models";
    if (argc > 1) model_dir = argv[1];

    std::cout << "[Info] Initializing engine with models in: " << model_dir << std::endl;

    // 2. 初始化引擎
    MuseTalkEngine engine;
    if (!engine.init(model_dir)) {
        std::cerr << "[Error] Initialization failed! Check model paths." << std::endl;
        return -1;
    }
    std::cout << "[Success] Engine initialized." << std::endl;

    // 3. 构造测试数据
    std::cout << "[Info] Preparing dummy data..." << std::endl;

    // A. 构造一张假的人脸图 (256x256, 灰色背景)
    cv::Mat dummy_face(256, 256, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::circle(dummy_face, cv::Point(128, 128), 100, cv::Scalar(255, 255, 255), -1);
    
    // B. 构造假的音频特征 (50帧, 384维)
    cv::Mat dummy_audio(50, 384, CV_32F);
    cv::randu(dummy_audio, cv::Scalar(0.0f), cv::Scalar(1.0f));

    // C. 构造假的嘴部掩码 (Mask) 
    // 接口要求传入 cv::Mat mask，而不是 cv::Rect
    cv::Mat mask = cv::Mat::zeros(256, 256, CV_8UC1); // 初始化全黑
    cv::Rect mouth_roi(100, 150, 60, 40);             // 定义矩形区域
    cv::rectangle(mask, mouth_roi, cv::Scalar(255), -1); // 将矩形区域涂白 (255)

    // 4. 运行推理
    std::cout << "[Info] Running inference (VAEEnc -> UNet -> VAEDec)..." << std::endl;
    try {
        // 【修正】传入 mask 而不是 mouth_roi
        cv::Mat result = engine.run(dummy_face, dummy_audio, mask);
        
        if (result.empty()) {
            std::cerr << "[Error] Inference returned empty result." << std::endl;
            return -1;
        }

        std::cout << "[Success] Inference complete." << std::endl;
        std::cout << "   Output size: " << result.cols << "x" << result.rows << std::endl;

        // 5. 保存结果
        std::string out_path = "musetalk_validity_test.jpg";
        cv::imwrite(out_path, result);
        std::cout << "[Success] Result saved to: " << out_path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Fatal Error] Exception during inference: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}