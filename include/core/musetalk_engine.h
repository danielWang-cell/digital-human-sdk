#pragma once

#include <opencv2/core.hpp>
#include <memory>
#include <string>

namespace DigitalHuman {
namespace Core {

class MuseTalkEngine {
public:
    MuseTalkEngine();
    ~MuseTalkEngine();

    MuseTalkEngine(const MuseTalkEngine&) = delete;
    MuseTalkEngine& operator=(const MuseTalkEngine&) = delete;

    /**
     * @param model_dir  包含：
     *   vae_encoder.ncnn.param/.bin
     *   vae_decoder.ncnn.param/.bin
     *   musetalk_unet.ncnn.param/.bin
     */
    bool init(const std::string& model_dir);

    /**
     * @param aligned_face 256x256 BGR, CV_8UC3（已对齐人脸）
     * @param audio_feat   50x384, CV_32F（MuseTalk audio embedding）
     * @param mouth_mask   256x256, CV_8UC1（嘴部 255，其余 0）
     */
    cv::Mat run(const cv::Mat& aligned_face,
                const cv::Mat& audio_feat,
                const cv::Mat& mouth_mask);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Core
} // namespace DigitalHuman
