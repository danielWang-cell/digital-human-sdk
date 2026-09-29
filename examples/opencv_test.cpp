#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// 辅助函数：画一个简单的 C++ 房子
void drawCppHouse(cv::Mat& img) {
    // 1. 绘制地面
    cv::line(img, cv::Point(0, 550), cv::Point(600, 550), cv::Scalar(50, 200, 50), 5);

    // 2. 绘制房子主体
    cv::Rect house_body(150, 300, 300, 250);
    cv::rectangle(img, house_body, cv::Scalar(100, 100, 100), -1); // 填充灰色

    // 3. 绘制屋顶
    std::vector<cv::Point> roof_points;
    roof_points.push_back(cv::Point(130, 300));
    roof_points.push_back(cv::Point(470, 300));
    roof_points.push_back(cv::Point(300, 150));
    cv::fillPoly(img, roof_points, cv::Scalar(50, 50, 200)); // 红色

    // 4. 写字 "C++"
    cv::putText(img, "C", cv::Point(275, 280), cv::FONT_HERSHEY_DUPLEX, 3.0, cv::Scalar(255, 255, 255), 4);
    cv::putText(img, "+", cv::Point(200, 450), cv::FONT_HERSHEY_DUPLEX, 4.0, cv::Scalar(0, 255, 255), 5);
    cv::putText(img, "+", cv::Point(340, 450), cv::FONT_HERSHEY_DUPLEX, 4.0, cv::Scalar(0, 255, 255), 5);
}

int main() {
    std::cout << "\n==========================================" << std::endl;
    std::cout << "   Digital Human SDK: OpenCV Test" << std::endl;
    std::cout << "==========================================\n" << std::endl;

    // ---------------------------------------------------------
    // 1. 验证 OpenCV 安装
    // ---------------------------------------------------------
    std::cout << "[1] Checking OpenCV Installation..." << std::endl;
    std::cout << "   -> OpenCV Version: " << CV_VERSION << std::endl;

    // ---------------------------------------------------------
    // 2. 验证基本写功能 (Image Write) - 生成素材
    // ---------------------------------------------------------
    std::cout << "[2] Generating test image in memory..." << std::endl;
    cv::Mat source_img = cv::Mat::zeros(600, 600, CV_8UC3);
    drawCppHouse(source_img);
    
    std::string src_filename = "test_src.jpg";
    if (cv::imwrite(src_filename, source_img)) {
        std::cout << "   -> [SUCCESS] Generated and saved: " << src_filename << std::endl;
    } else {
        std::cerr << "   -> [FAILED] Could not save initial image!" << std::endl;
        return -1;
    }

    // ---------------------------------------------------------
    // 3. 验证图像加载功能 (Image Load)
    // ---------------------------------------------------------
    std::cout << "[3] Testing imread (Loading image from disk)..." << std::endl;
    cv::Mat loaded_img = cv::imread(src_filename);

    if (loaded_img.empty()) {
        std::cerr << "   -> [FAILED] Could not open or find the image: " << src_filename << std::endl;
        return -1;
    }
    
    std::cout << "   -> [SUCCESS] Image loaded." << std::endl;
    std::cout << "   -> Resolution: " << loaded_img.cols << "x" << loaded_img.rows << std::endl;
    std::cout << "   -> Channels: " << loaded_img.channels() << std::endl;

    // ---------------------------------------------------------
    // 4. 验证颜色空间转换 (Color Conversion)
    // ---------------------------------------------------------
    std::cout << "[4] Testing color space conversion (BGR -> Grayscale)..." << std::endl;
    cv::Mat gray_img;
    
    // 调用核心处理函数
    try {
        cv::cvtColor(loaded_img, gray_img, cv::COLOR_BGR2GRAY);
        std::cout << "   -> [SUCCESS] Converted to Grayscale." << std::endl;
        std::cout << "   -> New Channels: " << gray_img.channels() << std::endl;
    } catch (const cv::Exception& e) {
        std::cerr << "   -> [FAILED] OpenCV Exception: " << e.what() << std::endl;
        return -1;
    }

    // ---------------------------------------------------------
    // 5. 验证处理结果输出 (Result Write)
    // ---------------------------------------------------------
    std::cout << "[5] Saving processed image..." << std::endl;
    std::string dst_filename = "test_result_gray.jpg";
    if (cv::imwrite(dst_filename, gray_img)) {
        std::cout << "   -> [SUCCESS] Saved grayscale image to: " << dst_filename << std::endl;
    } else {
        std::cerr << "   -> [FAILED] Could not save grayscale image!" << std::endl;
        return -1;
    }


    std::cout << "\n==========================================" << std::endl;
    std::cout << "   ALL TESTS PASSED SUCCESSFULLY" << std::endl;
    std::cout << "==========================================\n" << std::endl;

    return 0;
}