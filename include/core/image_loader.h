#pragma once

#include <string>
#include <memory>
#include <vector>
#include <stdexcept>
#include <opencv2/opencv.hpp>

namespace DigitalHuman{
namespace Core{
    /**
     * @brief 自定义的图像机载异常类
     *
     */
    class ImageLoaderException : public std::runtime_error{
    public:
        explicit ImageLoaderException(const std::string& message) : std::runtime_error(message){}
    };
    /**
     * @brief ImageLoader 类
     *
     */

    class ImageLoader {
    public:
        ImageLoader();
        ~ImageLoader(); // Pimpl 模式必须在 cpp 中定义析构

        ImageLoader(const ImageLoader&) = delete;
        ImageLoader& operator=(const ImageLoader&) = delete;

        ImageLoader(ImageLoader&&) noexcept;
        ImageLoader& operator=(ImageLoader&&) noexcept;


        /**
         * @brief 从文件加载图像
         * @param filepath 图像文件路径，支持绝对路径和相对路径
         * @return cv::Mat 加载后的 OpenCV 矩阵 (BGR 格式)
         * @throw ImageLoaderException 如果文件不存在或无法打开，抛出异常
         */
        cv::Mat loadFromFile(const std::string& filepath);


        /**
         * @brief 从内存加载图像
         * @param std::vector<unsigned char> buffer 
         * @return cv::Mat 加载后的 OpenCV 矩阵 (BGR 格式)
         * @throw ImageLoaderException 如果内存不存在或无法打开，抛出异常
         */
        cv::Mat loadFromMemory(const std::vector<unsigned char>& buffer);


        /**
         * @brief 批量加载
         * @param filepaths 图像文件路径列表
         * @return std::vector<cv::Mat> 加载后的 OpenCV 矩阵列表 (BGR 格式)
         * @throw ImageLoaderException 如果列表中某一文件不存在或无法打开，填入空或者报出警告
         */
        std::vector<cv::Mat> loadBatch(const std::vector<std::string>& filepaths);
    
    private:
        // 前置声明实现的结构体
        struct Impl;

        // 使用智能指针管理实现类的生命周期
        std::unique_ptr<Impl> pImpl;
    };

} // namespace core
} // namespace DigitalHuman
