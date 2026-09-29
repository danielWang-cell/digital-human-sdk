#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "video/render_process.h"
#include "utils/thread_safe_queue.h"
#include "video/video_frame.h"

using namespace DigitalHuman::Video;

void pushMockFrames(::ThreadSafeQueue<VideoFrame>& render_queue, std::atomic<double>& mock_audio_pts, 
                    const cv::Mat& bg_image, const cv::Rect& roi, int num_frames) {
    for (int i = 0; i < num_frames; ++i) {
        VideoFrame frame;
        frame.pts = i * 40.0; 
        
        cv::Mat current_mouth(96, 96, CV_8UC3, cv::Scalar(0, 0, 255));
        int mouth_open_height = 5 + 20 * std::abs(std::sin(i * 0.3)); 
        cv::ellipse(current_mouth, cv::Point(48, 48), cv::Size(30, mouth_open_height), 0, 0, 360, cv::Scalar(255, 255, 255), -1);

        frame.image = current_mouth;
        frame.background = bg_image.clone();
        frame.roi = roi;

        render_queue.push(std::move(frame));
        mock_audio_pts.store(i * 40.0);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
    }
}

int main() {
    std::cout << "========== 视觉综合测试 (Headless & GUI) ==========" << std::endl;

    std::atomic<double> mock_audio_pts{0.0};
    auto audio_clock_cb = [&]() -> double { 
        return mock_audio_pts.load(); 
    };

    ::ThreadSafeQueue<VideoFrame> render_queue(50);
    RenderProcessor renderer(render_queue, audio_clock_cb);

    cv::Mat bg_image(512, 512, CV_8UC3, cv::Scalar(50, 50, 50));
    cv::Rect roi(208, 208, 96, 96); 

    // ==========================================
    // 测试阶段 A：无界面模式 (Headless)
    // ==========================================
    std::cout << "\n[Test Phase A] 启动无界面 (Headless) 静默渲染测试..." << std::endl;
    mock_audio_pts.store(0.0); 
    
    // 调用 start(true) 强制开启后台渲染模式
    if (!renderer.start(true)) { 
        std::cerr << "[Test] 渲染线程启动失败！" << std::endl;
        return -1;
    }

    std::cout << "[Test Phase A] 正在后台静默处理 50 帧数据..." << std::endl;
    pushMockFrames(render_queue, mock_audio_pts, bg_image, roi, 50);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200)); 
    std::cout << "[Test Phase A] 停止无界面渲染线程...\n" << std::endl;
    renderer.stop(true);

    // ==========================================
    // 测试阶段 B：有界面模式 (GUI)
    // ==========================================
    std::cout << "\n[Test Phase B] 启动有界面 (GUI) 可视渲染测试..." << std::endl;
    mock_audio_pts.store(0.0); 
    
    // 调用 start(false) 要求弹窗显示
    if (!renderer.start(false)) { 
        std::cerr << "[Test] 渲染线程启动失败！" << std::endl;
        return -1;
    }

    std::cout << "[Test Phase B] 正在输出可视化动画 (100帧)..." << std::endl;
    pushMockFrames(render_queue, mock_audio_pts, bg_image, roi, 100);

    std::cout << "\n========== 所有压测任务投递完毕 ==========" << std::endl;
    std::cout << "\033[32m[Observation] 窗口将为您保留 4 秒钟以供观察...\033[0m" << std::endl;
    
    for(int wait = 4; wait > 0; wait--) {
        std::cout << "倒计时关闭: " << wait << " 秒..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\n[Test Phase End] 正在触发优雅退出机制..." << std::endl;
    renderer.stop(true);
    std::cout << "[Success] 综合测试圆满结束！" << std::endl;

    return 0;
}