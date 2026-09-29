#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <sstream>
#include <cmath>

#include "audio/ffmpeg_pulse_capture.h"

namespace DigitalHuman {
namespace Audio {
    
    FFmpegPulseCapture::FFmpegPulseCapture( const std::string& source_name,
                                            int sample_rates,
                                            int chunk_samples,
                                            PcmCallback cb)
            : source_name_(source_name),
            sample_rate_(sample_rates),
            chunk_samples_(chunk_samples),
            callback_(std::move(cb)){}

    FFmpegPulseCapture::~FFmpegPulseCapture() {
        stop();
    }

    bool FFmpegPulseCapture::start() {
        if (running_.load(std::memory_order_acquire)) {
            return true;            
        }

        running_.store(true, std::memory_order_release);

        // 启动后台线程，这里不能再主线程中读取 FFmpeg pipe，否则会阻塞主流程
        worker_ = std::thread(&FFmpegPulseCapture::captureLoop, this);

        std::cout << "[FFmpegPulseCapture] started. source=" << source_name_
              << ", sr=" << sample_rate_
              << ", chunk_samples=" << chunk_samples_
              << std::endl;

        return true;
    }

    void FFmpegPulseCapture::stop() {
        if (!running_.exchange(false, std::memory_order_acq_rel)) {
            return;
        }

        FILE* local_pipe = nullptr;
        {
            std::lock_guard<std::mutex> lock(pipe_mtx_);
            local_pipe = pipe_;
            pipe_ = nullptr;
        }
        // 这里直接 pclose(pipe_)原因是 captureLoop 可能正在 fread() 等待FFmpeg 输出数据，此时 pclose 可以关闭 pipe，使 fread返回，从而让线程退出
        // ruguo  captureLoop 已经退出，则会将pipe_置空，所以在这里需要空校验
        if (local_pipe) {
            pclose(local_pipe);
        }

        if (worker_.joinable()) {
            worker_.join();
        }

        std::cout << "[FFmpegPulseCapture] stopped." << std::endl;
    }

    void FFmpegPulseCapture::captureLoop() {
        std::ostringstream cmd;

        cmd << "ffmpeg -hide_banner -loglevel warning "
            << "-f pulse "                                      // 设置输入设备类型为 PulseAudio
            << "-i " << source_name_ << " "                     // 指定 PulseAudio 为输入源，这里是 RDFSource
            << "-af volume=-4dB "                               // 对输入音频做衰减
            << "-ar " << sample_rate_ << " "                    // 重采样到 16000 HZ
            << "-ac 1 "                                         // 转成单声道
            << "-f s16le - "                                    // 输出裸 PCM，格式为 signed 16-bit little-endian , - 表示输出到 stdout
            << "2>/tmp/ffmpeg_pulse_capture.log";               // 把 FFmpeg 日志写到文件中，

        std::cout << "[FFmpegPulseCapture] command: " << cmd.str() << std::endl;

        // 以只读的方式启动FFmpeg子进程
        FILE* local_pipe = popen(cmd.str().c_str(), "r");
        if (!local_pipe) {
            std::cerr << "[FFmpegPulseCapture] popen failed: "
                    << std::strerror(errno) << std::endl;
            running_.store(false, std::memory_order_release);
            return;
        }

        {
            std::lock_guard<std::mutex> lock(pipe_mtx_);
            pipe_ = local_pipe;
        }

        const size_t bytes_per_samples = sizeof(int16_t);
        const size_t target_read_bytes = static_cast<size_t>(chunk_samples_) * bytes_per_samples;

        std::vector<int16_t> pcm16(chunk_samples_);
        std::vector<float> pcm_float(chunk_samples_);

        // 构造连续音频时间轴
        int64_t total_samples = 0;

        while (running_.load(std::memory_order_acquire)) {
             /**
             * 从 FFmpeg stdout 中读取一块 PCM
             * 理想情况下，每次读到：
             *   chunk_samples_ * sizeof(int16_t)
             * 但 pipe 读取可能返回不足一个完整 chunk 的字节数，所以下面根据实际 n 计算 samples_read。
             */
            size_t n = fread(pcm16.data(), 1, target_read_bytes, pipe_);

            if (n == 0) {
                if (feof(pipe_)) {
                    std::cerr << "[FFmpegPulseCapture] ffmpeg pipe EOF." << std::endl;
                    break;
                }

                if (ferror(pipe_)) {
                    std::cerr << "[FFmpegPulseCapture] pipe read error." << std::endl;
                    break;
                }

                continue;
            }

            size_t samples_read = n / bytes_per_samples;
            if (samples_read == 0) {
                continue;
            }

            /**
             * 如果 fread 返回了不足 chunk_samples_ 的数据， pcm_float 的长度也应该和实际的 samples_read保持一致
             */
            pcm_float.resize(samples_read);

            //  int16 PCM 转 float PCM
            for (size_t i = 0; i  < samples_read; ++i) {
                pcm_float[i] = static_cast<float>(pcm16[i]) / 32768.0f;
            }

            // 当前的 PCM 块的起始时间戳
            // 先用 total_samples 计算当前块的起始 PTS， 再把 samples_read 累加进去
            double pts_ms = static_cast<double>(total_samples) / static_cast<double>(sample_rate_) * 1000.0;

            total_samples += static_cast<int64_t>(samples_read);


            // 调试部分
#if 0
            static int rms_count = 0;
            if (rms_count++ % 30 == 0) {
                double sum = 0.0;
                for (float v : pcm_float) {
                    sum += v * v;
                }
                double rms = std::sqrt(sum / pcm_float.size());
                std::cout << "[FFmpegPulseCapture] rms=" << rms
                        << ", pts=" << pts_ms
                        << ", samples=" << samples_read
                        << std::endl;
            } 
#endif
            if (callback_) {
                callback_(pcm_float, pts_ms);
            }
        }

        // 退出，这里是 FFmpeg 子进程自己结束，负责 pclose
         {
            std::lock_guard<std::mutex> lock(pipe_mtx_);
            if (pipe_ == local_pipe) {
                pipe_ = nullptr;
            }
        }

        if (local_pipe) {
            pclose(local_pipe);
        }
        running_.store(false, std::memory_order_release);

        std::cout << "[FFmpegPulseCapture] captureLoop exit." << std::endl;

    }

} // namespace Audio
} // namespace DigitalHuman