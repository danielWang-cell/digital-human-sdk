#include <iostream>
#include <vector>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp> 
#include "model/output_processor.h"

using namespace DigitalHuman::Model;

int main() {
    std::cout << "=== Digital Human SDK: Output Processor Test ===" << std::endl;

    OutputProcessor processor;

    // 1. 构造模拟数据
    int w = 96, h = 96;
    
    // A. 模拟模型输出 Tensor ([-1, 1])
    // 假设生成了一个纯红色的嘴巴区域
    ncnn::Mat dummy_tensor(w, h, 3);
    float* ptr_r = dummy_tensor.channel(0);
    float* ptr_g = dummy_tensor.channel(1);
    float* ptr_b = dummy_tensor.channel(2);
    for(int i=0; i<w*h; ++i) {
        ptr_r[i] = 1.0f;  // R=255
        ptr_g[i] = -1.0f; // G=0
        ptr_b[i] = -1.0f; // B=0
    }

    // B. 模拟原始人脸 (背景)
    // 纯灰色背景
    cv::Mat original_face(w, h, CV_8UC3, cv::Scalar(128, 128, 128));

    // C. 模拟掩码 (Mask)
    // 中间一个圆形的洞是白色的 (表示生成的嘴巴)，周围是黑色的 (表示保留原图)
    cv::Mat mask = cv::Mat::zeros(w, h, CV_32FC1);
    cv::circle(mask, cv::Point(48, 48), 30, cv::Scalar(1.0f), -1);
    // 做一点羽化
    cv::GaussianBlur(mask, mask, cv::Size(15, 15), 0);

    // 2. 执行处理
    std::cout << "[Step] Processing output..." << std::endl;
    cv::Mat result = processor.process(dummy_tensor, original_face, mask);

    // 3. 验证结果
    if (result.empty()) {
        std::cerr << "[Error] Processing failed!" << std::endl;
        return -1;
    }
    std::cout << "[Success] Result image created: " << result.cols << "x" << result.rows << std::endl;

    // 4. 质量验证
    QualityReport report = processor.validateLastOutput();
    std::cout << "   Quality Valid: " << (report.is_valid ? "Yes" : "No") << std::endl;
    std::cout << "   Sharpness Score: " << report.sharpness_score << std::endl;

    // 5. 保存图像以便人工检查
    // 预期效果：灰色背景，中间有个红色光晕，边缘平滑过渡
    cv::imwrite("output_blend_test.jpg", result);
    std::cout << "[Info] Check 'output_blend_test.jpg' for visual verification." << std::endl;

    // 6. 保存原始生成图对比 (不融合)
    cv::Mat raw_gen = processor.process(dummy_tensor, cv::Mat(), cv::Mat());
    cv::imwrite("output_raw_test.jpg", raw_gen);

    return 0;
}