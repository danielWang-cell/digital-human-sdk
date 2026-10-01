#include "model/wav2lip_backend.h"

#include <chrono>

namespace DigitalHuman::Model {

Wav2LipBackend::Wav2LipBackend(ncnn::Net* net) : net_(net) {
    inference_.bindModel(net_);
}

bool Wav2LipBackend::available() const {
    return net_ != nullptr;
}

void Wav2LipBackend::setConfig(const InferenceConfig& config) {
    inference_.setConfig(config);
}

LatencyStats Wav2LipBackend::latencyStats() const {
    return inference_.getLatencyStats();
}

bool Wav2LipBackend::process(const AudioFeature& audio,
                             const FaceInput& face,
                             InferenceResult& result) {
    result = InferenceResult{};
    if (!available() || face.aligned_face.empty() || audio.values.size() != 1280) {
        return false;
    }

    const cv::Rect mouth_roi = face.context.mouth_roi.width > 0
        ? face.context.mouth_roi
        : cv::Rect(0, face.aligned_face.rows / 2,
                   face.aligned_face.cols, face.aligned_face.rows / 2);
    const ncnn::Mat face_tensor = input_processor_.processImage(face.aligned_face, mouth_roi);
    const ncnn::Mat audio_tensor = input_processor_.processAudio(audio.values);
    if (face_tensor.empty() || audio_tensor.empty()) {
        return false;
    }

    ncnn::Mat output_tensor;
    const auto start = std::chrono::steady_clock::now();
    if (inference_.infer(audio_tensor, face_tensor, output_tensor) != 0) {
        return false;
    }
    result.generated_face = output_processor_.process(output_tensor, cv::Mat(), cv::Mat());
    result.latency_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    result.success = !result.generated_face.empty();
    return result.success;
}

} // namespace DigitalHuman::Model
