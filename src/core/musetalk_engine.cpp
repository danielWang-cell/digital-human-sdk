#include "core/musetalk_engine.h"

#include <ncnn/net.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <random>

namespace DigitalHuman {
namespace Core {

class MuseTalkEngine::Impl {
public:
    bool init(const std::string& model_dir);
    cv::Mat run(const cv::Mat& image,
                const cv::Mat& mask,
                const cv::Mat& audio_feat);

private:
    void parse_unet_inputs(const std::string& param_path);

private:
    ncnn::Net unet_;
    std::vector<std::string> unet_input_names_;
};

/* ========================= init ========================= */

bool MuseTalkEngine::Impl::init(const std::string& model_dir)
{
    const std::string param_path = model_dir + "/musetalk_unet.ncnn.param";
    const std::string bin_path   = model_dir + "/musetalk_unet.ncnn.bin";

    // 1. parse UNet input names
    parse_unet_inputs(param_path);
    if (unet_input_names_.size() != 3) {
        std::cerr << "[MuseTalk] ERROR: expect 3 UNet inputs, got "
                  << unet_input_names_.size() << std::endl;
        return false;
    }

    // 2. load UNet
    unet_.opt.use_vulkan_compute = false;
    unet_.opt.num_threads = 4;
    unet_.opt.use_fp16_storage = false;
    unet_.opt.use_fp16_arithmetic = false;

    if (unet_.load_param(param_path.c_str()) != 0 ||
        unet_.load_model(bin_path.c_str()) != 0) {
        std::cerr << "[MuseTalk] ERROR: failed to load UNet model\n";
        return false;
    }

    std::cout << "[MuseTalk] UNet loaded successfully\n";
    std::cout << "[MuseTalk] UNet inputs:\n";
    for (size_t i = 0; i < unet_input_names_.size(); ++i) {
        std::cout << "  [" << i << "] " << unet_input_names_[i] << "\n";
    }

    return true;
}

/* ========================= parse_unet_inputs ========================= */

void MuseTalkEngine::Impl::parse_unet_inputs(const std::string& param_path)
{
    std::ifstream fin(param_path);
    if (!fin.is_open()) {
        std::cerr << "[MuseTalk] ERROR: cannot open param file: "
                  << param_path << std::endl;
        return;
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        std::string layer_type, layer_name;
        int bottom_count, top_count;

        ss >> layer_type >> layer_name >> bottom_count >> top_count;

        if (layer_type == "Input") {
            std::string blob_name;
            ss >> blob_name;
            unet_input_names_.push_back(blob_name);
        }
    }

    fin.close();
}

/* ========================= run ========================= */

cv::Mat MuseTalkEngine::Impl::run(const cv::Mat& image,
                                 const cv::Mat& /*mask*/,
                                 const cv::Mat& /*audio_feat*/)
{
    // ------------------------------
    // 1. latent: [1, 8, 32, 32]
    // ------------------------------
    ncnn::Mat latent(32, 32, 8, sizeof(float));

    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.f, 1.f);

    for (int c = 0; c < latent.c; c++) {
        float* ptr = latent.channel(c);
        for (int i = 0; i < 32 * 32; i++) {
            ptr[i] = dist(rng);
        }
    }

    // ------------------------------
    // 2. timestep (int32!)
    // ------------------------------
    ncnn::Mat timestep(1, (size_t)4u); // int32
    reinterpret_cast<int*>(timestep.data)[0] = 500;

    // ------------------------------
    // 3. audio embedding (dummy, shape OK)
    // [1, 50, 384]
    // ------------------------------
    ncnn::Mat audio_emb(384, 50, 1, sizeof(float));
    audio_emb.fill(0.f);

    // ------------------------------
    // 4. UNet forward
    // ------------------------------
    ncnn::Extractor ex = unet_.create_extractor();

    ex.input(unet_input_names_[0].c_str(), latent);
    ex.input(unet_input_names_[1].c_str(), timestep);
    ex.input(unet_input_names_[2].c_str(), audio_emb);

    ncnn::Mat out;
    ex.extract("out0", out);

    if (out.empty()) {
        std::cerr << "[MuseTalk] UNet output EMPTY\n";
        return cv::Mat();
    }

    // ------------------------------
    // 5. 临时阶段：直接返回原图
    // （后续这里接 VAE decoder）
    // ------------------------------
    return image.clone();
}

/* ========================= Public API ========================= */

MuseTalkEngine::MuseTalkEngine()
{
    impl_ = std::make_unique<Impl>();
}

MuseTalkEngine::~MuseTalkEngine() = default;

bool MuseTalkEngine::init(const std::string& model_dir)
{
    return impl_->init(model_dir);
}

cv::Mat MuseTalkEngine::run(const cv::Mat& image,
                            const cv::Mat& mask,
                            const cv::Mat& audio_feat)
{
    return impl_->run(image, mask, audio_feat);
}

} // namespace Core
} // namespace DigitalHuman
