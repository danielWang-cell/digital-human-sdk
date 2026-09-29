#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "core/image_loader.h"

using namespace DigitalHuman::Core;

/**
 * @brief 辅助函数，将文件读取为二进制 buffer
 * 
 * @param fileName 文件路径
 * @return std::vector<unsigned char> 二进制 buffer
 */
 std::vector<unsigned char> readFileToBuffer(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        return {};
    }

    // 移动指针到末尾获取大小
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 读取文件内容
    std::vector<unsigned char> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    return buffer;
}

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: ImageLoader Test (Pimpl) ===" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    std::string test_path = argv[1];

    // 实例化对象
    ImageLoader loader;

    // ------------------------------------------
    // 1: 从文件加载
    // ------------------------------------------
    std::cout << "\n[1] 从硬盘文件加载..." << std::endl;
    try {
        cv::Mat img1 = loader.loadFromFile(test_path);
        std::cout << "SUCCESS: 加载成功, 尺寸: " << img1.cols << "x" << img1.rows << std::endl;
    } catch (const DigitalHuman::Core::ImageLoaderException& e) {
        std::cerr << "FAILED: " << e.what() << std::endl;
    }

    // ------------------------------------------
    // 2: 从内存加载 (模拟网络接收)
    // ------------------------------------------
    std::cout << "\n[2] 从内存 Buffer 加载..." << std::endl;
    // 模拟：先读成二进制流
    std::vector<unsigned char> mem_buffer = readFileToBuffer(test_path);
    if (mem_buffer.empty()) {
        std::cerr << "无法读取文件到内存用于测试" << std::endl;
    } else {
        std::cout << "模拟接收到 " << mem_buffer.size() << " 字节的数据包" << std::endl;
        try {
            cv::Mat img2 = loader.loadFromMemory(mem_buffer);
            std::cout << "SUCCESS: 内存解码成功, 尺寸: " << img2.cols << "x" << img2.rows << std::endl;
        } catch (const DigitalHuman::Core::ImageLoaderException& e) {
            std::cerr << "FAILED: " << e.what() << std::endl;
        }
    }

    // ------------------------------------------
    // 3: 异常处理 (故意加载不存在的文件)
    // ------------------------------------------
    std::cout << "\n[3] 测试异常处理..." << std::endl;
    try {
        cv::Mat bad = loader.loadFromFile("not_exist_file.jpg");
    } catch (const DigitalHuman::Core::ImageLoaderException& e) {
        std::cout << "SUCCESS (捕获到预期异常): " << e.what() << std::endl;
    }

    // ------------------------------------------
    // 4: 批量加载
    // ------------------------------------------
    std::cout << "\n[4] 批量加载..." << std::endl;
    std::vector<std::string> batch_paths = {
        test_path,              // 正常
        "fake_path_123.jpg",    // 错误
        test_path               // 正常
    };
    
    auto batch_imgs = loader.loadBatch(batch_paths);
    std::cout << "批量结果: 成功加载 " << batch_imgs.size() << " 张图片 (预期应为 2 张)" << std::endl;

    return 0;
}