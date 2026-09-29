#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp> 

#include "model/model_loader.h"
#include "model/input_processor.h"
#include "model/model_inference.h"

using namespace DigitalHuman::Model;

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: Inference Module Test ===" << std::endl;

    if (argc < 2) {
        std::cerr << "Usage: ./inference_test <model_path>" << std::endl;
        return -1;
    }
    std::string model_path = argv[1];

    // 1. 加载模型 (依赖 ModelLoader)
    ModelLoader loader;
    if (!loader.load(model_path)) {
        return -1;
    }
    std::cout << "[Step 1] Model Loaded." << std::endl;

    // 2. 准备推理引擎
    ModelInference engine;
    engine.bindModel(loader.getNet());
    
    // 配置参数
    InferenceConfig config;
    config.num_threads = 4; // 使用 4 线程（匹配 4 vCPU 环境）
    config.use_fp16 = true;
    engine.setConfig(config);
    
    std::cout << "[Step 2] Inference Engine Ready." << std::endl;

    // 3. 准备 Dummy 数据 (模拟 InputProcessor 的输出)
    // 真实场景下，这里应该调用 InputProcessor
    // Audio: [1, 1, 80, 16]
    ncnn::Mat audio_in(16, 80, 1, (size_t)4u);
    audio_in.fill(0.5f); // 填充随机值

    // Face: [1, 6, 96, 96]
    ncnn::Mat face_in(96, 96, 6, (size_t)4u);
    face_in.fill(0.0f); // 填充0 (相当于全黑图)

    // 4. 执行推理 (热身)
    ncnn::Mat out;
    if (engine.infer(audio_in, face_in, out) != 0) {
        std::cerr << "Inference Failed!" << std::endl;
        return -1;
    }
    std::cout << "[Step 3] Warm-up Inference Done." << std::endl;

    // 5. 性能测试 (Loop 50次)
    std::cout << "[Step 4] Benchmarking (50 iters)..." << std::endl;
    float total_time = 0;
    for (int i = 0; i < 50; ++i) {
        engine.infer(audio_in, face_in, out);
        total_time += engine.getLastLatency();
    }
    float avg_time = total_time / 50.0f;
    
    std::cout << "   Average Latency: " << avg_time << " ms" << std::endl;
    
    // 验收标准检查
    if (avg_time < 50.0f) {
        std::cout << "   -> PASS (Real-time requirement met < 50ms)" << std::endl;
    } else {
        std::cout << "   -> WARNING (Performance might be low)" << std::endl;
    }

    // 6. 检查输出维度
    std::cout << "[Step 5] Checking Output..." << std::endl;
    std::cout << "   Output Shape: " << out.w << "x" << out.h << "x" << out.c << std::endl;
    
    if (out.w == 96 && out.h == 96 && out.c == 3) {
        std::cout << "   -> PASS (Dimension correct)" << std::endl;
    } else {
        std::cerr << "   -> FAIL (Dimension mismatch)" << std::endl;
    }

    return 0;
}
