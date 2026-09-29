#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>

#include "core/pipeline.h"

using namespace DigitalHuman::Core;

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "  Digital Human SDK: Pipeline Core Test  " << std::endl;
    std::cout << "===========================================" << std::endl;

    // 实例化主控管线
    Pipeline pipeline;

    // 启动流水线
    if (!pipeline.start("../models/", false, "")) {
        std::cerr << "Failed to start pipeline!" << std::endl;
        return -1;
    }

    std::cout << "\n[Test] Pipeline is running. Simulating data input...\n" << std::endl;

    // 模拟一张 640x480 的人脸底图
    cv::Mat dummy_face = cv::Mat::zeros(480, 640, CV_8UC3);
    cv::putText(dummy_face, "Test Face", cv::Point(200, 240), 
                cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
                
    // 模拟一段音频 Mel 频谱特征数据
    std::vector<float> dummy_mel(80 * 16, 0.5f); 

    // 模拟主循环 ，运行 5 秒钟
    const double target_fps = 25.0;
    const double frame_duration_ms = 1000.0 / target_fps; // 每帧 40ms
    int frame_count = 0;

    auto start_time = std::chrono::steady_clock::now();

    while (true) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - start_time).count();

        // 运行 5 秒后退出测试
        if (elapsed_ms > 5000.0) {
            break;
        }

        // 检查系统是否健康（看门狗是否触发了死锁报警，或抛出了异常）
        if (!pipeline.isHealthy()) {
            std::cerr << "[Test] Pipeline reported unhealthy state! Aborting loop." << std::endl;
            break;
        }

        // 构造推理任务
        InferenceTask task;
        task.pts_ms = frame_count * frame_duration_ms; // 赋予时间戳 (0, 40, 80, 120...)
        task.base_face = dummy_face;
        task.audio_feature = dummy_mel;

        // 推入流水线 (如果队列满了，这里会稍微阻塞)
        bool success = pipeline.pushTask(task);
        if (success) {
            if (frame_count % 25 == 0) {
                // 每隔一秒打印一次进度，避免刷屏
                std::cout << "[Test] Successfully pushed 25 frames. Current PTS: " 
                          << task.pts_ms << " ms" << std::endl;
            }
            frame_count++;
        }

        // 模拟摄像头捕获耗时 (~40ms 产生一帧)
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }

    std::cout << "\n[Test] 5 seconds passed. Stopping pipeline..." << std::endl;

    // 退出流水线
    pipeline.stop();

    std::cout << "===========================================" << std::endl;
    std::cout << "  Test Finished Successfully!  " << std::endl;
    std::cout << "===========================================" << std::endl;

    return 0;
}