#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>

#include "core/image_loader.h"

namespace fs = std::filesystem;

namespace DigitalHuman {
namespace Core {

// Pimpl的内部实现类，里面包含了3种方法
struct ImageLoader::Impl {
    cv::Mat loadFromFile(const std::string& filePath) {
        // 首先需要检查文件是否存在
        if (!fs::exists(filePath)) {
            throw ImageLoaderException("File does not exist: " + filePath);
        }

        // 加载图像，默认加载为彩色 BGR
        cv::Mat image = cv::imread(filePath);
        if (image.empty()) {
            throw ImageLoaderException("Failed to decode image: " + filePath);
        } else {
            return image;
        }
    }

    cv::Mat loadFromMemory(const std::vector<unsigned char>& buffer) {
        // 首先判断 Buffer 是否为空
        if (buffer.empty()) {
            throw ImageLoaderException("Buffer is empty");
        }

        // Buffer 不为空，使用 cv::imdecode 从内存解码
        cv::Mat image;

        try {
            image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        } catch (const cv::Exception& e) {
            throw ImageLoaderException("Failed to decode image from memory: " + std::string(e.what()));
        }
        
        if (image.empty()) {
            throw ImageLoaderException("Failed to decode image from memory");
        } 

        return image;
    }


    std::vector<cv::Mat> loadBatch(const std::vector<std::string>& filePaths) {
        std::vector<cv::Mat> images;
        images.reserve(filePaths.size());  // 内存预分配

        for (const auto& path : filePaths) {
            try {
                cv::Mat image = loadFromFile(path);
                images.push_back(image);
            } catch (const ImageLoaderException& e) {
                std::cerr << "\033[31m[Batch Load Error] " << e.what() << "\033[0m" << std::endl;

                // 可选择存入一个空 Mat 保持索引对齐
                // images.push_back(cv::Mat());
                // 或者选择跳过当前文件
                continue;
            }
        }

        return images;
    }
    
};


ImageLoader::ImageLoader() : pImpl(std::make_unique<Impl>()){}

ImageLoader::~ImageLoader() = default;

// 移动构造和赋值
ImageLoader::ImageLoader(ImageLoader&&) noexcept = default;
ImageLoader& ImageLoader::operator=(ImageLoader&&) noexcept = default;

cv::Mat ImageLoader::loadFromFile(const std::string& filePath) {
    return pImpl->loadFromFile(filePath);
}

cv::Mat ImageLoader::loadFromMemory(const std::vector<unsigned char>& buffer) {
    return pImpl->loadFromMemory(buffer);
}

std::vector<cv::Mat> ImageLoader::loadBatch(const std::vector<std::string>& filepaths) {
    return pImpl->loadBatch(filepaths);
}

} // namespace Core
} // namespace DigitalHuman