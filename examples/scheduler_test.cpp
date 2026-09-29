#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <string>

#include <opencv2/imgproc.hpp> 
#include <opencv2/core.hpp>

#include "core/frame_scheduler.h"
#include "video/video_frame.h" 

using namespace DigitalHuman::Core;
using DigitalHuman::Video::VideoFrame; 

int main() {
    std::cout << "=== Digital Human SDK: Frame Scheduler Test ===" << std::endl;

    FrameScheduler scheduler(25.0); // 25 FPS -> 40ms/frame

    // 1. 模拟推理线程：提前生成 15 帧数据放入队列 (增加到 15 帧以支持后续测试)
    // PTS: 0, 40, 80, ..., 560
    std::cout << "[Step 1] Pushing 15 frames to buffer..." << std::endl;
    for (int i = 0; i < 15; ++i) {
        VideoFrame frame;
        // 造一个带数字的图像方便辨认
        frame.image = cv::Mat::zeros(100, 100, CV_8UC3);
        
        cv::putText(frame.image, std::to_string(i), cv::Point(30, 50), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 255), 2);
        
        frame.pts = i * 40.0;
        frame.index = i;
        scheduler.pushFrame(frame);
    }
    std::cout << "   Buffer Size: " << scheduler.getQueueSize() << std::endl;

    // 2. 模拟渲染循环 (音频时钟驱动)
    struct TestCase {
        double audio_time;
        std::string expected_action;
    };

    std::vector<TestCase> cases = {
        {0.0,   "Render Frame 0 (Diff=0)"},
        {35.0,  "Render Frame 1 (Diff=5, Sync)"},
        //Frame 9 (360ms, Diff=-41) 被丢弃，直接渲染 Frame 10 (400ms, Diff=-1)
        {401.0, "Should DROP frames 2..9, Render Frame 10"},
        // Frame 10 已被消费，队首是 Frame 11 (440ms)
        {445.0, "Render Frame 11 (440ms) (Diff=-5)"},
        // 队首 Frame 12 (480ms). Audio 485ms. Diff -5. Sync.
        {485.0, "Render Frame 12 (480ms)"},
        // 队首 Frame 13 (520ms). Audio 470ms. Diff +50 > 40. Wait.
        {470.0, "Wait (Frame 13 is too fast, Diff=50)"}
    };

    std::cout << "\n[Step 2] Simulating Render Loop..." << std::endl;

    for (const auto& test : cases) {
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Audio Time: " << test.audio_time << " ms | Expect: " << test.expected_action << std::endl;
        
        cv::Mat result = scheduler.getFrameForRender(test.audio_time);
        
        std::cout << "   Current Queue Size: " << scheduler.getQueueSize() << std::endl;
        std::cout << "   Total Dropped: " << scheduler.getDroppedCount() << std::endl;
        
        // 可选：保存图片验证
        // if (!result.empty()) {
        //      cv::imwrite("sched_" + std::to_string((int)test.audio_time) + ".jpg", result);
        // }
    }

    // 3. 验证掉帧逻辑
    // Frame 0, 1 正常消费。
    // 在 401ms 时，Frame 2(80)..9(360) 共8帧应该被丢弃，Frame 10 被渲染。
    // 所以预期掉帧数应为 8。
    if (scheduler.getDroppedCount() >= 8) {
        std::cout << "\n[Result] PASS: Frame dropping logic works correctly." << std::endl;
    } else {
        std::cout << "\n[Result] FAIL: Did not drop enough frames. Count: " << scheduler.getDroppedCount() << std::endl;
    }

    return 0;
}