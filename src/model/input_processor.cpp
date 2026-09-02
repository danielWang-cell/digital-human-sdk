#include <iostream>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include "model/input_processor.h"

namespace DigitalHuman {
namespace Model {

struct InputProcessor::Impl {

    // 这是 Wav2Lip 固定输入尺寸
    const int TARGET_SIZE = 96;
    const int MEL_H = 80;
    const int MEL_W = 16;

    ncnn::Mat processImage(const cv::Mat& image, const cv::Rect& mouth_roi) {
        if (image.empty()) {
            spdlog::error("[InputProcessor] Empty input image.");
            return ncnn::Mat();
        }

        // 1. 首先缩放至 96 x 96
        cv::Mat resized_img;
        if (image.rows != TARGET_SIZE || image.cols != TARGET_SIZE) {
            cv::resize(image, resized_img, cv::Size(TARGET_SIZE, TARGET_SIZE));
        } else {
            resized_img = image;
        }

        // 2. 创建下半脸遮挡的 Masked Image
        cv::Mat masked_img = resized_img.clone();
        // 如果传入了 ROI，使用 ROI；否则默认遮挡下半部分，这里的 ROI 必须是相对于 96 x 96 图像的坐标
        cv::Rect roi = mouth_roi;

        // 如果 ROI 无效或未传入，默认遮挡下半脸
        if (roi.area() == 0) {
            roi = cv::Rect(0, TARGET_SIZE / 2, TARGET_SIZE, TARGET_SIZE / 2);
        }

        // 确保 ROI 不越界
        roi = roi & cv::Rect(0, 0, TARGET_SIZE, TARGET_SIZE);
        // 涂黑（置 0）
        cv::rectangle(masked_img, roi, cv::Scalar(0, 0, 0), -1);

        // 3. 归一化参数 ( 0 - 255 -> -1.0 - 1.0 )
        // mean = 127.5, norm = 1 / 127.5
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};

        // 4. 转换(为 ncnn::Mat ( HWC -> CHW, BGR -> RGB 视训练模型而定 )
        ncnn::Mat ncnn_ref = ncnn::Mat::from_pixels(
            resized_img.data,
            ncnn::Mat::PIXEL_BGR2RGB,
            TARGET_SIZE, TARGET_SIZE
        );
        ncnn_ref.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Mat ncnn_masked = ncnn::Mat::from_pixels(
            masked_img.data,
            ncnn::Mat::PIXEL_BGR2RGB,
            TARGET_SIZE, TARGET_SIZE
        );
        ncnn_masked.substract_mean_normalize(mean_vals, norm_vals);

        // 5. 拼接通道( 3ch + 3ch -> 6ch )
        // ncnn::Mat 是 Planar 存储的，可以直接按通道拷贝
        ncnn::Mat input_tensor(TARGET_SIZE, TARGET_SIZE, 6, (size_t)4u); // 4 bytes float

        // 拷贝 Reference，前 3 层
        for (int c = 0; c < 3; c++) {
            const float* src = ncnn_ref.channel(c);
            float* dst = input_tensor.channel(c);
            memcpy(dst, src, TARGET_SIZE * TARGET_SIZE * sizeof(float));
        }

        // 拷贝 Masked，后 3 层
        for (int c = 0; c < 3; c++) {
            const float* src = ncnn_masked.channel(c);
            float* dst = input_tensor.channel(c + 3);
            memcpy(dst, src, TARGET_SIZE * TARGET_SIZE * sizeof(float));
        }

        // // 5. 拼接通道( 3ch + 3ch -> 6ch )
        // ncnn::Mat input_tensor;
        // std::vector<ncnn::Mat> mats = {ncnn_ref, ncnn_masked};
        // ncnn::concat(mats, input_tensor, 0); // 0 表示沿着 channel 拼接

        return input_tensor;
    }

    ncnn::Mat processAudio(const std::vector<float>& mel_data) {
        // 检查数据长度
        if (mel_data.size() != static_cast<size_t>(MEL_H * MEL_W)) {
            spdlog::error("[InputProcessor] Invalid audio data size. Expected {}, got {}",
                          MEL_H * MEL_W, mel_data.size());
            return ncnn::Mat();  
        }

        // 构造 ncnn::Mat，注意 ncnn 的维度顺序：w(inner), h, c(outer)
        // pytorch [Batch, 1, 80, 16] -> H = 80, W = 16
        // 在 ncnn 中，我们要创建一个 w = 16，h = 80，c = 1 的矩阵
        ncnn::Mat audio_tensor(MEL_W, MEL_H, 1, (size_t)4u);

        // 拷贝数据
        memcpy(audio_tensor.data, mel_data.data(), mel_data.size() * sizeof(float));

        return audio_tensor;
    }
};

InputProcessor::InputProcessor() : pImpl(std::make_unique<Impl>()) {}
InputProcessor::~InputProcessor() = default;

ncnn::Mat InputProcessor::processImage(const cv::Mat& image, const cv::Rect& mouth_roi) {
    return pImpl->processImage(image, mouth_roi);
}

ncnn::Mat InputProcessor::processAudio(const std::vector<float>& mel_data) {
    return pImpl->processAudio(mel_data);
}

bool InputProcessor::validateTensor(const ncnn::Mat& tensor, int expected_c, int expected_h, int expected_w) {
    if (tensor.empty()) {
        return false;
    }
    if (tensor.c != expected_c || tensor.h != expected_h || tensor.w != expected_w) {
        spdlog::error("[InputProcessor] Validation Failed! Expected {}x{}x{}, got {}x{}x{}",
                      expected_w, expected_h, expected_c, tensor.w, tensor.h, tensor.c);
        return false;
    }
    return true;
}

} // namespace Model
} // namespace DigitalHuman