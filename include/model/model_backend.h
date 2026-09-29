#pragma once

#include "core/frame_context.h"
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace DigitalHuman::Model {

struct AudioFeature {
    std::vector<float> values;
    double pts_ms = 0.0;
};

struct FaceInput {
    cv::Mat aligned_face;
    Core::FrameContext context;
};

struct InferenceResult {
    cv::Mat generated_face;
    double latency_ms = 0.0;
    bool success = false;
};

class ModelBackend {
public:
    virtual ~ModelBackend() = default;
    virtual const char* name() const = 0;
    virtual bool available() const = 0;
    virtual bool process(const AudioFeature&, const FaceInput&, InferenceResult&) = 0;
};

} // namespace DigitalHuman::Model
