#pragma once

namespace DigitalHuman::Core {

struct PipelineConfig {
    int inference_threads = 4;
    bool use_vulkan = false;
    bool use_fp16 = false;
    bool light_mode = true;
    bool enable_sharpen = false;
    double sharpen_strength = 0.25;
    bool enable_debug_logging = true;
};

} // namespace DigitalHuman::Core
