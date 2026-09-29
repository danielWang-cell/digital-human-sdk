#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <atomic>
#include <chrono>
#include <iomanip>
#include "core/timestamp_manager.h"

using namespace DigitalHuman::Core;

// 辅助函数：打印带颜色的状态日志
void printStatus(int frame_idx, double v_pts, double diff, SyncAction action) {
    std::cout << std::fixed << std::setprecision(1)
              << "Frame " << std::setw(2) << frame_idx 
              << " | V_PTS: " << std::setw(6) << v_pts 
              << " | Diff: " << std::setw(6) << diff << "ms | Action: ";
    
    switch (action) {
        case SyncAction::Render: 
            std::cout << "\033[32m[Render]\033[0m (Sync OK)"; 
            break;
        case SyncAction::Drop:   
            std::cout << "\033[31m[Drop]\033[0m   (Video Too Slow)"; 
            break;
        case SyncAction::Wait:   
            std::cout << "\033[33m[Wait]\033[0m   (Video Too Fast)"; 
            break;
        case SyncAction::Reset:  
            std::cout << "\033[35m[Reset]\033[0m  (Sync Lost)"; 
            break;
    }
    std::cout << std::endl;
}

/**
 * @brief 运行一个测试场景
 * @param name 场景名称
 * @param render_cost_ms 模拟每帧渲染耗时(ms)
 * @param inject_drift 是否注入严重的时间漂移(模拟音频突变)
 */
void runTest(const std::string& name, int render_cost_ms, bool inject_drift = false) {
    std::cout << "\n========================================================" << std::endl;
    std::cout << "[Test Scenario]: " << name << std::endl;
    std::cout << "  -> Simulating Render Cost: " << render_cost_ms << "ms per frame" << std::endl;
    std::cout << "========================================================" << std::endl;

    TimestampManager sync_mgr;
    sync_mgr.reset(); // 确保初始状态清零
    
    const int sample_rate = 16000;
    const double fps = 25.0; // 目标帧间隔 40ms
    std::atomic<bool> running{true};

    // --- 1. 启动模拟音频线程 (作为时钟源) ---
    // 它是绝对的参考系，匀速前进
    std::thread audio_thread([&]() {
        int64_t total_samples = 0;
        int tick = 0;
        while(running) {
            // 每 10ms 推进一次音频进度
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            
            // 正常推进 160 个采样点 (10ms)
            int samples_to_add = 160;

            // 特殊测试：在第 20 次 tick 时注入 1秒 的剧烈漂移
            if (inject_drift && tick == 20) {
                // 突然增加 16000 个采样点 (1秒)，模拟音频缓冲区跳变
                samples_to_add += 16000; 
                std::cout << "\n>>> [Injection] Audio timestamp jumped forward by 1000ms! <<<\n" << std::endl;
            }

            total_samples += samples_to_add;
            sync_mgr.updateAudioTime(total_samples, sample_rate);
            tick++;
        }
    });

    // --- 2. 模拟视频渲染主循环 ---
    // 模拟运行 15 帧
    for (int i = 0; i < 15; ++i) {
        // 模拟渲染耗时
        std::this_thread::sleep_for(std::chrono::milliseconds(render_cost_ms));
        
        // 获取当前帧应有的时间戳
        double v_pts = sync_mgr.getNextVideoPTS(fps);
        
        // 检查同步状态
        SyncAction action = sync_mgr.checkSync(v_pts);
        double diff = sync_mgr.getDiff(); // Diff = Video - Audio

        printStatus(i, v_pts, diff, action);

        // 如果建议等待，模拟等待时间以修正偏差
        if (action == SyncAction::Wait) {
            // 简单模拟：如果快了，就多等一会儿，让音频追上来
            // 实际代码中这里会是真正的 sleep
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    // 停止音频线程
    running = false;
    if (audio_thread.joinable()) {
        audio_thread.join();
    }
}

int main() {
    // ---------------------------------------------------------
    // 测试 1: 正常情况 (Normal)
    // 渲染耗时 35ms (略小于帧间隔 40ms)，应该是完美同步的
    // ---------------------------------------------------------
    runTest("Normal Synchronization", 35);

    // ---------------------------------------------------------
    // 测试 2: 视频滞后 (Lagging -> Drop)
    // 渲染耗时 70ms (远大于 40ms)，视频会越来越慢，直到触发丢帧
    // ---------------------------------------------------------
    runTest("Video Lagging (Drop Strategy)", 70);

    // ---------------------------------------------------------
    // 测试 3: 视频超前 (Leading -> Wait)
    // 渲染耗时 10ms (极快)，视频跑得太快了，需要等待音频
    // ---------------------------------------------------------
    runTest("Video Leading (Wait Strategy)", 10);
    
    // ---------------------------------------------------------
    // 测试 4: 严重漂移 (Hard Reset)
    // 渲染正常，但音频突然跳变 1秒，触发强制重置
    // ---------------------------------------------------------
    runTest("Severe Drift (Reset Strategy)", 35, true);

    return 0;
}