#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>

#include "core/image_loader.h"

// 定义命名空间
namespace fs = std::filesystem;

// 实现 ImageLoader 类
namespace DigitalHuman {
namespace Core {

// Pimpl 的内部实现类，里面包含了 3 种方法
struct ImageLoader::Impl {
    cv::Mat loadFromFile(const std::string& filepath) {
        // 首先需要检查文件是否存在
        if (!fs::exists(filepath)) {
            throw ImageLoaderException("File does not exist: " + filepath);
        }
        // 加载图像，默认加载为彩色 BGR
        cv::Mat image = cv::imread(filepath);
        if (image.empty()) {
            throw ImageLoaderException("Failed to decode image: " + filepath);
        } else {
            return image;
        }
    }

    cv::Mat loadFromMemory(const std::vector<unsigned char>& buffer) {
        // 首先判断 buffer 是否为空
        if (buffer.empty()) {
            throw ImageLoaderException("Buffer is empty");
        }

        // buffer 不为空，使用 cv::imread 从内存解码
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

    std::vector<cv::Mat> loadBatch(const std::vector<std::string>& filepaths) {
        std::vector<cv::Mat> images;
        images.reserve(filepaths.size());

        for (const auto& path : filepaths) {
            try {
                cv::Mat image = loadFromFile(path);
                images.push_back(image);
            } catch (const ImageLoaderException& e) {
                std::cerr << "\033[31m[Batch Load Error] " << e.what() << "033[0m" << std::endl;
                // images.push_back(cv::Mat());
                continue;
            }
        }

        return images;
    }
};

// ImageLoader 的外部类，提供访问内部类的方法
// 构造与析构函数
ImageLoader::ImageLoader() : pImpl(std::make_unique<Impl>()){}

ImageLoader::~ImageLoader() = default;

// 移动构造和赋值
ImageLoader::ImageLoader(ImageLoader&&) noexcept = default;
ImageLoader& ImageLoader::operator=(ImageLoader&&) noexcept = default;

cv::Mat ImageLoader::loadFromFile(const std::string& filepath) {
    return pImpl->loadFromFile(filepath);
}

cv::Mat ImageLoader::loadFromMemory(const std::vector<unsigned char>& buffer) {
    return pImpl->loadFromMemory(buffer);
}

std::vector<cv::Mat> ImageLoader::loadBatch(const std::vector<std::string>& filepaths) {
    return pImpl->loadBatch(filepaths);
}

}
}
