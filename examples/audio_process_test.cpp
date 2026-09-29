#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <chrono>
#include <atomic>
#include <cassert>

#include "audio/audio_process.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace DigitalHuman::Audio;

/**
 * @brief 辅助函数：生成一段标准正弦波音频（模拟真实的声音输入）
 */
void generateSineWave(std::vector<float>& pcm, int sample_rate, float frequency, float duration_sec) {
    int num_samples = static_cast<int>(sample_rate * duration_sec);
    pcm.resize(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        pcm[i] = std::sin(2.0f * M_PI * frequency * i / sample_rate);
    }
}

int main() {
    std::cout << "========== 开始测试 AudioProcessor ==========" << std::endl;

    const int SAMPLE_RATE = 16000;
    std::atomic<int> callback_count{0};
    std::atomic<double> last_pts{-1.0};

    // 1. 定义特征输出回调函数 (Lambda 表达式)
    // 这个函数将运行在 AudioProcessor 的内部处理线程中
    AudioProcessor::FeatureCallback output_callback = [&](double pts_ms, std::vector<float> mel_feature) {
        callback_count++;
        last_pts = pts_ms;
        
        // 检查提取出的特征向量大小是否为 80（对应 n_mels = 80）
        assert(mel_feature.size() == 80 && "Mel feature size must be 80!");
        
        // 打印部分日志以确认进度，避免日志刷屏，每 20 帧打印一次
        if (callback_count % 20 == 0) {
            std::cout << "[Test Callback] 成功接收第 " << callback_count 
                      << " 帧特征，对应时间戳 PTS: " << pts_ms << " ms" << std::endl;
        }
    };

    // 2. 初始化音频处理器
    AudioProcessor processor(output_callback, SAMPLE_RATE);

    // 3. 启动后台线程
    std::cout << "[Test] 正在启动音频处理线程..." << std::endl;
    bool started = processor.start();
    assert(started && "Processor should start successfully!");

    // 4. 生成 1 秒钟的 440Hz 标准测试音频 (16000 个采样点)
    std::vector<float> full_pcm_data;
    generateSineWave(full_pcm_data, SAMPLE_RATE, 440.0f, 1.0f);
    std::cout << "[Test] 成功生成 1 秒测试音频数据，共 " << full_pcm_data.size() << " 个采样点。" << std::endl;

    // 5. 模拟实时音频流：将 1 秒的数据切分成小块 (如每次 512 个采样点) 推送进去
    const int CHUNK_SIZE = 512;
    double current_pts = 0.0;

    std::cout << "[Test] 正在模拟实时音频回调，推送数据中..." << std::endl;
    for (size_t i = 0; i < full_pcm_data.size(); i += CHUNK_SIZE) {
        size_t end = std::min(i + CHUNK_SIZE, full_pcm_data.size());
        std::vector<float> chunk(full_pcm_data.begin() + i, full_pcm_data.begin() + end);
        
        // 调用我们的 pushRawAudio 接口
        processor.pushRawAudio(chunk, current_pts);
        
        // 更新下一块的起始时间戳：加上当前块的时长 (毫秒)
        current_pts += (chunk.size() / static_cast<double>(SAMPLE_RATE)) * 1000.0;
        
        // 稍微休眠，模拟真实的硬件耗时，也可以注释掉以测试极限吞吐量
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // 6. 等待后台线程处理完所有积压的数据
    std::cout << "[Test] 数据推送完毕，等待队列消费..." << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // 7. 停止后台线程
    std::cout << "[Test] 正在停止音频处理线程..." << std::endl;
    processor.stop();

    // 8. 验证结果正确性
    int total_frames = callback_count.load();
    std::cout << "\n========== 测试结果汇总 ==========" << std::endl;
    std::cout << "总共提取了 " << total_frames << " 帧 Mel 特征。" << std::endl;
    std::cout << "最后一帧时间戳: " << last_pts.load() << " ms" << std::endl;

    /*
     * 理论计算：
     * 1 秒的数据 = 1000ms
     * 帧长 25ms，帧移 10ms
     * 提取的帧数 = (1000 - 25) / 10 + 1 = 98 帧 (大约)
     */
    assert(total_frames > 90 && total_frames < 105 && "Total frames should be around 98!");
    
    std::cout << "\033[32m[Success] AudioProcessor 模块全部功能测试通过！\033[0m" << std::endl;

    return 0;
}