#pragma once

#include <memory>
#include <ncnn/net.h>
#include "model/model_backend.h"
#include "model/model_inference.h"
#include "model/input_processor.h"
#include "model/output_processor.h"

namespace DigitalHuman::Model {

class Wav2LipBackend final : public ModelBackend {
public:
    explicit Wav2LipBackend(ncnn::Net* net);

    const char* name() const override { return "Wav2Lip"; }
    bool available() const override;
    bool process(const AudioFeature&, const FaceInput&, InferenceResult&) override;

    void setConfig(const InferenceConfig& config);
    LatencyStats latencyStats() const;

private:
    ncnn::Net* net_ = nullptr;
    ModelInference inference_;
    InputProcessor input_processor_;
    OutputProcessor output_processor_;
};

} // namespace DigitalHuman::Model
