#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <thread>
#include <vector>
#include <cstdio>
#include <mutex>

namespace DigitalHuman {
namespace Audio {

/**
 * @brief 基于 FFmpeg + PulseAudio 的实时麦克风采集类
 *
 * 使用场景：
 *   当前项目运行在 WSL2 + Docker 环境中，容器内没有真实 ALSA 声卡设备，
 *   因此不能稳定使用 PortAudio/ALSA 直接打开麦克风。
 *   但是 WSLg 会通过 PulseAudio 提供 Windows 麦克风输入源，例如： RDPSource
 * 
 *因此本类通过启动 FFmpeg 子进程：
 *
 *      ffmpeg -f pulse -i RDPSource -ar 16000 -ac 1 -f s16le -
 *
 *   从 PulseAudio 中读取实时麦克风数据，并把 FFmpeg 输出的裸 PCM 数据 转换为 float PCM，回调给上层 AudioProcessor。
 *
 * 输出数据格式：
 *   - sample_rate: 16000 Hz
 *   - channels: 1
 *   - sample format: float32，范围约 [-1.0, 1.0]
 *
 * 典型链路：
 *
 *   Windows 麦克风
 *       ↓
 *   WSLg / PulseAudio / RDPSource
 *       ↓
 *   Docker 容器中的 FFmpeg
 *       ↓
 *   FFmpegPulseCapture
 *       ↓
 *   AudioProcessor::pushRawAudio()
 *       ↓
 *   Wav2Lip 实时推理
 */
class FFmpegPulseCapture {
public:

    /**
     * @brief PCM 数据回调函数类型
     * @param pcm    一段 float PCM 数据，单声道，范围约 [-1.0, 1.0]
     * @param pts_ms 当前 PCM 块对应的起始时间戳，单位 ms
     * @note pts_ms 是根据已经读取的采样点数量计算出来的逻辑时间戳：pts_ms = total_samples / sample_rate * 1000
     *       是音频流内部的连续时间轴
     */
    using PcmCallback = std::function<void(const std::vector<float>& pcm, double pts_ms)>;

     /**
     * @brief 构造函数
     * @param source_name   PulseAudio 输入源名称， "RDPSource"
     * @param sample_rate   目标采样率，Wav2Lip 默认使用 16000
     * @param chunk_samples 每次回调的采样点数量，例如 512。
     *                      16000 Hz 下 512 samples ≈ 32 ms
     * @param cb            采集到 PCM 后的回调函数
     */
    FFmpegPulseCapture(const std::string& source_name,
                       int sample_rate,
                       int chunk_samples,
                       PcmCallback cb);

    /**
     * @brief 析构时自动停止采集线程和 FFmpeg 子进程
     */
    ~FFmpegPulseCapture();

    FFmpegPulseCapture(const FFmpegPulseCapture&) = delete;
    FFmpegPulseCapture& operator=(const FFmpegPulseCapture&) = delete;

    /**
     * @brief 启动采集线程
     *        内部会启动一个后台线程，在该线程中通过 popen 启动 FFmpeg，并持续从 FFmpeg stdout 读取裸 PCM 数据。
     * @return true 表示线程启动成功；false 表示启动失败
     */
    bool start();

    /**
     * @brief 停止采集
     * 会关闭 FFmpeg pipe，并等待后台采集线程退出。
     */
    void stop();

    bool isRunning() const {
        return running_.load();
    }

private:
/**
     * @brief 采集线程主循环
     * 主要流程：
     *   1. 拼接 FFmpeg 命令
     *   2. popen 启动 FFmpeg
     *   3. 按 chunk_samples_ 读取 s16le PCM
     *   4. int16 转 float
     *   5. 计算 pts_ms
     *   6. 调用 callback_
     */
    void captureLoop();

private:
    std::string source_name_;
    int sample_rate_ = 16000;
    int chunk_samples_ = 512;    // 每次从 FFmpeg pipe 中读取的采样点数

    PcmCallback callback_;

    std::atomic<bool> running_{false}; 
    std::mutex pipe_mtx_;
    std::thread worker_;        // 后台采集线程

    /**
     * @brief FFmpeg 子进程 stdout pipe
     * popen 返回的 FILE*
     *   pipe_ 只在 captureLoop 所在线程中读取
     *   stop() 中会调用 pclose(pipe_) 以结束子进程
     */
    FILE* pipe_ = nullptr;
};

} // namespace Audio
} // namespace DigitalHuman