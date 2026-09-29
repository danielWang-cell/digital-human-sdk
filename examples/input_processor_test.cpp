#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2/highgui.hpp>

#include "model/input_processor.h"

using namespace DigitalHuman::Model;

void check_value_range(const ncnn::Mat& m, const std::string& name) {
    float min_v = 1000.0f, max_v = -1000.0f;
    for (int c=0; c<m.c; c++) {
        const float* ptr = m.channel(c);
        for (int i=0; i<m.w*m.h; i++) {
            if (ptr[i] < min_v) min_v = ptr[i];
            if (ptr[i] > max_v) max_v = ptr[i];
        }
    }
    std::cout << "   " << name << " Range: [" << min_v << ", " << max_v << "]";
    if (min_v >= -1.05 && max_v <= 1.05) std::cout << " -> PASS" << std::endl;
    else std::cout << " -> FAIL (Out of range [-1, 1])" << std::endl;
}

int main() {
    std::cout << "=== Digital Human SDK: Input Processor Test ===" << std::endl;

    InputProcessor processor;

    // ------------------------------------------------
    // Test 1: 图像输入处理
    // ------------------------------------------------
    std::cout << "\n[Test 1] Image Processing (6-channel construction)..." << std::endl;
    // 创建一个纯白图像 (255)
    cv::Mat white_img(96, 96, CV_8UC3, cv::Scalar(255, 255, 255));
    // 嘴部 ROI (中间一块)
    cv::Rect mouth_roi(32, 60, 32, 20);

    ncnn::Mat img_tensor = processor.processImage(white_img, mouth_roi);

    // 验证维度
    if (processor.validateTensor(img_tensor, 6, 96, 96)) {
        std::cout << "   Dimension Check: PASS" << std::endl;
    } else {
        std::cout << "   Dimension Check: FAIL" << std::endl;
    }

    // 验证数值归一化
    // 白点(255) -> (255-127.5)/127.5 = 1.0
    // 遮挡点(0) -> (0-127.5)/127.5 = -1.0
    check_value_range(img_tensor, "Image Tensor");

    // 验证 Mask 是否生效
    // 检查第 4 通道 (Masked Blue) 的中心点，应该是 -1.0 (黑)
    float center_val = img_tensor.channel(3).row(70)[48]; 
    std::cout << "   Masked Region Value: " << center_val << " (Expected -1.0)" << std::endl;

    // ------------------------------------------------
    // Test 2: 音频输入处理
    // ------------------------------------------------
    std::cout << "\n[Test 2] Audio Processing..." << std::endl;
    // 构造模拟 Mel 数据 (80x16)
    std::vector<float> mel_data(80 * 16, 0.5f);
    
    ncnn::Mat audio_tensor = processor.processAudio(mel_data);
    
    if (processor.validateTensor(audio_tensor, 1, 80, 16)) { // c=1, h=80, w=16
        std::cout << "   Dimension Check: PASS" << std::endl;
    } else {
        std::cout << "   Dimension Check: FAIL" << std::endl;
    }

    // ------------------------------------------------
    // Test 3: 异常输入测试
    // ------------------------------------------------
    std::cout << "\n[Test 3] Error Handling..." << std::endl;
    std::vector<float> bad_audio(100); // 数据不够
    ncnn::Mat bad_tensor = processor.processAudio(bad_audio);
    if (bad_tensor.empty()) std::cout << "   Bad Audio Check: PASS" << std::endl;
    else std::cout << "   Bad Audio Check: FAIL" << std::endl;

    return 0;
}