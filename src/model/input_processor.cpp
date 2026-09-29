#include <iostream>
#include <algorithm>
#include <cstring>
#include <opencv2/imgproc.hpp>

#include "model/input_processor.h"

namespace DigitalHuman {
namespace Model {

struct InputProcessor::Impl {

    const int TARGET_SIZE = 96;
    const int MEL_H = 80;
    const int MEL_W = 16;

    ncnn::Mat processImage(const cv::Mat& image, const cv::Rect& mouth_roi) {
        if (image.empty()) {
            return ncnn::Mat();
        }

        cv::Mat resized_img;
        if (image.rows != TARGET_SIZE || image.cols != TARGET_SIZE) {
            cv::resize(image, resized_img, cv::Size(TARGET_SIZE, TARGET_SIZE));
        } else {
            resized_img = image.clone();
        }

        if (resized_img.type() != CV_8UC3) {
            resized_img.convertTo(resized_img, CV_8UC3);
        }

        cv::Mat masked_img = resized_img.clone();

        cv::Rect roi = mouth_roi;
        if (roi.width <= 0 || roi.height <= 0) {
            roi = cv::Rect(0, TARGET_SIZE / 2, TARGET_SIZE, TARGET_SIZE / 2);
        }

        roi &= cv::Rect(0, 0, TARGET_SIZE, TARGET_SIZE);
        if (roi.width > 0 && roi.height > 0) {
            masked_img(roi).setTo(cv::Scalar(0, 0, 0));
        }

        // 保持 BGR，不做 BGR2RGB
        ncnn::Mat ncnn_masked = ncnn::Mat::from_pixels(
            masked_img.data, ncnn::Mat::PIXEL_BGR, TARGET_SIZE, TARGET_SIZE);

        ncnn::Mat ncnn_org = ncnn::Mat::from_pixels(
            resized_img.data, ncnn::Mat::PIXEL_BGR, TARGET_SIZE, TARGET_SIZE);

        const float norm_vals[3] = {
            1.0f / 255.0f,
            1.0f / 255.0f,
            1.0f / 255.0f
        };

        ncnn_masked.substract_mean_normalize(nullptr, norm_vals);
        ncnn_org.substract_mean_normalize(nullptr, norm_vals);

        ncnn::Mat input_tensor(TARGET_SIZE, TARGET_SIZE, 6, (size_t)4u);

        // 官方顺序：masked face first
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_tensor.channel(c),
                        ncnn_masked.channel(c),
                        TARGET_SIZE * TARGET_SIZE * sizeof(float));
        }

        // original face second
        for (int c = 0; c < 3; ++c) {
            std::memcpy(input_tensor.channel(c + 3),
                        ncnn_org.channel(c),
                        TARGET_SIZE * TARGET_SIZE * sizeof(float));
        }

        return input_tensor;
    }

    ncnn::Mat processAudio(const std::vector<float>& mel_features) {
        if (mel_features.size() != 16 * 80) {
            std::cerr << "[InputProcessor] Invalid mel size: "
                    << mel_features.size() << ", expected 1280." << std::endl;
            return ncnn::Mat();
        }

        ncnn::Mat audio_tensor(16, 80, 1, (size_t)4u);

        for (int freq = 0; freq < 80; ++freq) {
            float* row_ptr = audio_tensor.row(freq);
            for (int time = 0; time < 16; ++time) {
                row_ptr[time] = mel_features[freq * 16 + time];
            }
        }

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
        std::cerr << "[InputProcessor] Validation Failed! Expected: "
                  << expected_w << "x" << expected_h << "x" << expected_c
                  << ", Got: "
                  << tensor.w << "x" << tensor.h << "x" << tensor.c << std::endl;
        return false;
    }
    return true;
}

} // namespace Model
} // namespace DigitalHuman