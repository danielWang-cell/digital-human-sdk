#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/imgproc.hpp>

#include "model/output_processor.h"

namespace DigitalHuman {
namespace Model {

struct OutputProcessor::Impl {
    cv::Mat last_result;
    QualityReport last_report;

    cv::Mat tensorToImage(const ncnn::Mat& tensor) {
        if (tensor.empty()) {
            return cv::Mat();
        }

        if (tensor.w != 96 || tensor.h != 96 || tensor.c != 3) {
            std::cerr << "[OutputProcessor] Unexpected tensor shape: "
                    << tensor.w << "x" << tensor.h << "x" << tensor.c << std::endl;
            return cv::Mat();
        }

        cv::Mat img(96, 96, CV_8UC3);

        const float* ch0 = tensor.channel(0); // B
        const float* ch1 = tensor.channel(1); // G
        const float* ch2 = tensor.channel(2); // R

        for (int y = 0; y < 96; ++y) {
            cv::Vec3b* row = img.ptr<cv::Vec3b>(y);
            for (int x = 0; x < 96; ++x) {
                int idx = y * 96 + x;

                float b = std::min(std::max(ch0[idx], 0.0f), 1.0f);
                float g = std::min(std::max(ch1[idx], 0.0f), 1.0f);
                float r = std::min(std::max(ch2[idx], 0.0f), 1.0f);

                row[x] = cv::Vec3b(
                    static_cast<unsigned char>(b * 255.0f + 0.5f),
                    static_cast<unsigned char>(g * 255.0f + 0.5f),
                    static_cast<unsigned char>(r * 255.0f + 0.5f)
                );
            }
        }

        return img;
    }

    void sharpenImage(cv::Mat& img) {
        if (img.empty()) {
            return;
        }
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            0, -1,  0,
           -1,  5, -1,
            0, -1,  0);
        cv::filter2D(img, img, img.depth(), kernel);
    }

    cv::Mat blendImages(const cv::Mat& gen, const cv::Mat& org, const cv::Mat& mask) {
        if (gen.empty()) {
            return cv::Mat();
        }

        if (org.empty() || mask.empty()) {
            return gen.clone();
        }

        cv::Mat mask_f;
        if (mask.type() != CV_32FC1) {
            mask.convertTo(mask_f, CV_32F, 1.0 / 255.0);
        } else {
            mask_f = mask;
        }

        cv::Mat mask_3c;
        cv::cvtColor(mask_f, mask_3c, cv::COLOR_GRAY2BGR);

        cv::Mat gen_f, org_f;
        gen.convertTo(gen_f, CV_32F);
        org.convertTo(org_f, CV_32F);

        cv::Mat part1 = gen_f.mul(mask_3c);
        cv::Mat part2 = org_f.mul(cv::Scalar::all(1.0f) - mask_3c);

        cv::Mat result_f;
        cv::add(part1, part2, result_f);

        cv::Mat result;
        result_f.convertTo(result, CV_8UC3);
        return result;
    }

    void checkQuality(const cv::Mat& img) {
        last_report.is_valid = true;
        last_report.message = "OK";
        last_report.sharpness_score = 0.0;

        if (img.empty()) {
            last_report.is_valid = false;
            last_report.message = "Empty Output";
            return;
        }

        cv::Mat gray, lap;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::Laplacian(gray, lap, CV_64F);

        cv::Scalar mu, sigma;
        cv::meanStdDev(lap, mu, sigma);
        double variance = sigma.val[0] * sigma.val[0];
        last_report.sharpness_score = variance;

        if (variance < 50.0) {
            last_report.message = "Low Sharpness (Blurry)";
        }
    }
};

OutputProcessor::OutputProcessor() : pImpl(std::make_unique<Impl>()) {}
OutputProcessor::~OutputProcessor() = default;

cv::Mat OutputProcessor::process(const ncnn::Mat& tensor,
                                 const cv::Mat& original_face,
                                 const cv::Mat& mask) {
    cv::Mat generated = pImpl->tensorToImage(tensor);
    if (generated.empty()) {
        return cv::Mat();
    }

    // 先关掉锐化，避免把异常输出进一步放大
    // pImpl->sharpenImage(generated);

    if (!original_face.empty() && !mask.empty()) {
        pImpl->last_result = pImpl->blendImages(generated, original_face, mask);
    } else {
        pImpl->last_result = generated;
    }

    pImpl->checkQuality(pImpl->last_result);
    return pImpl->last_result;
}

QualityReport OutputProcessor::validateLastOutput() const {
    return pImpl->last_report;
}

} // namespace Model
} // namespace DigitalHuman