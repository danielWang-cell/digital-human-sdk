#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "model/model_loader.h"
#include "model/model_inference.h"

using namespace DigitalHuman::Model;

int main(int argc, char** argv) {
    const std::string model_dir = argc > 1 ? argv[1] : "../models";
    const int iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 50;
    const std::string model_path = model_dir + "/wav2lip.ncnn.param";

    ModelLoader loader;
    if (!loader.load(model_path)) {
        std::cerr << "benchmark_error=model_load_failed path=" << model_path << '\n';
        return 2;
    }

    ModelInference engine;
    engine.bindModel(loader.getNet());
    InferenceConfig config;
    config.num_threads = 4;
    config.use_fp16 = false;
    config.light_mode = true;
    engine.setConfig(config);

    ncnn::Mat audio(16, 80, 1, static_cast<size_t>(4u));
    ncnn::Mat face(96, 96, 6, static_cast<size_t>(4u));
    audio.fill(0.5f);
    face.fill(0.0f);
    ncnn::Mat output;

    if (engine.infer(audio, face, output) != 0) {
        std::cerr << "benchmark_error=warmup_failed\n";
        return 3;
    }
    engine.resetLatencyStats();
    for (int i = 0; i < iterations; ++i) {
        if (engine.infer(audio, face, output) != 0) {
            std::cerr << "benchmark_error=inference_failed iteration=" << i << '\n';
            return 4;
        }
    }

    const LatencyStats stats = engine.getLatencyStats();
    std::cout << std::fixed << std::setprecision(3)
              << "model=wav2lip\n"
              << "iterations=" << stats.sample_count << '\n'
              << "average_ms=" << stats.averageMs() << '\n'
              << "min_ms=" << stats.min_ms << '\n'
              << "max_ms=" << stats.max_ms << '\n'
              << "fps=" << (stats.averageMs() > 0.0f ? 1000.0f / stats.averageMs() : 0.0f) << '\n'
              << "output_shape=" << output.w << "x" << output.h << "x" << output.c << '\n';
    return 0;
}
