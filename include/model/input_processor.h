#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <ncnn/mat.h>

namespace DigitalHuman {
namespace Model {

/**
 * @brief 模型输入预处理器，负责将原始数据转换为 ncnn 模型所需的 Tensor格式
 */
class InputProcessor {

public:
    InputProcessor();
    ~InputProcessor();

    InputProcessor(const InputProcessor&) = delete;
    InputProcessor& operator=(const InputProcessor&) = delete;

    /**
     * @brief 处理图像输入（Wav2Lip 专用）
     *          将图像处理为 [1, 6, 96, 96] 的 Tensor，包含 Reference 和 Masked 两部分
     * @param image 原始图像，OpenCV BGR 
     * @param mouth_roi 嘴部区域，用于生成下半脸 Mask
     * @return ncnn::Mat 6 通道张量，值域 [-1.0, 1.0]
     */
    ncnn::Mat processImage(const cv::Mat& image, const cv::Rect& mouth_roi);

    /**
     * @brief 处理音频输入，将梅尔频谱数据转换为 [1, 1, 80, 16] 的 Tensor
     * @param mel_data 展平的梅尔频谱数据，大小应为 80 * 16 = 1280
     * @return ncnn::Mat 单通道张量
     */
    ncnn::Mat processAudio(const std::vector<float>& mel_data);

    /**
     @brief 验证张量是否有效
     */
    bool validateTensor(const ncnn::Mat& tensor, int expected_c, int expected_h, int expected_w);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;

};

} // namespace Model
} // namespace DigitalHuman