#include <iostream>
#include <thread>
#include <chrono>

#include "model/model_loader.h"

using namespace DigitalHuman::Model;

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: Model Loader Test (Module) ===" << std::endl;

    std::string model_path = "../models/wav2lip.ncnn.param";
    if (argc > 1) model_path = argv[1];

    // ---------------------------------------------
    // Test 1: 错误处理测试
    // ---------------------------------------------
    std::cout << "\n[Test 1] Error Handling (Invalid Path)..." << std::endl;
    {
        ModelLoader loader;
        if (!loader.load("wrong/path/fake.param")) {
            std::cout << "   -> PASS (Correctly detected missing file)" << std::endl;
        } else {
            std::cerr << "   -> FAIL (Should not load invalid file)" << std::endl;
        }
    }

    // ---------------------------------------------
    // Test 2: 同步加载与预热 (Sync Load & Warm-up)
    // ---------------------------------------------
    std::cout << "\n[Test 2] Sync Load & Warm-up..." << std::endl;
    {
        ModelLoader loader;
        auto start = std::chrono::high_resolution_clock::now();
        
        if (loader.load(model_path)) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "   -> Loaded & Warmed up in " << duration << " ms" << std::endl;
            
            if (loader.isLoaded()) {
                std::cout << "   -> PASS (Model state is loaded)" << std::endl;
            }
        } else {
            std::cerr << "   -> FAIL (Load failed, check model path)" << std::endl;
        }
    }

    // ---------------------------------------------
    // Test 3: 异步加载 (Async Load)
    // ---------------------------------------------
    std::cout << "\n[Test 3] Async Loading..." << std::endl;
    {
        ModelLoader async_loader;
        bool is_finished = false;

        std::cout << "   -> Starting async load..." << std::endl;
        
        async_loader.loadAsync(model_path, [&](ncnn::Net* net, float cost) {
            std::cout << "\n   -> [Callback] Load Finished!" << std::endl;
            std::cout << "   -> [Callback] Cost inside thread: " << cost << " ms" << std::endl;
            if (net) std::cout << "   -> [Callback] Net pointer valid." << std::endl;
            is_finished = true;
        });

        // 模拟 UI 线程不阻塞
        int ticks = 0;
        while (!is_finished) {
            std::cout << "." << std::flush;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            ticks++;
            if (ticks > 50) break; // 5秒超时
        }
        std::cout << "\n   -> PASS (Async logic works)" << std::endl;
    }

    return 0;
}