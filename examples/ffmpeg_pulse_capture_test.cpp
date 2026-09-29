#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <thread>
#include <vector>

#include "audio/ffmpeg_pulse_capture.h"

using DigitalHuman::Audio::FFmpegPulseCapture;

static std::atomic<bool> g_running{true};

void handleSignal(int) {
    g_running.store(false);
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handleSignal);

    std::string source = "RDPSource";
    if (argc >= 2) {
        source = argv[1];
    }

    constexpr int sample_rate = 16000;
    constexpr int chunk_samples = 512;

    std::atomic<int64_t> callback_count{0};
    std::atomic<int64_t> total_samples{0};

    FFmpegPulseCapture capture(
        source,
        sample_rate,
        chunk_samples,
        [&](const std::vector<float>& pcm, double pts_ms) {
            callback_count++;
            total_samples += static_cast<int64_t>(pcm.size());

            double sum = 0.0;
            float max_abs = 0.0f;

            for (float v : pcm) {
                sum += static_cast<double>(v) * static_cast<double>(v);
                max_abs = std::max(max_abs, std::abs(v));
            }

            double rms = pcm.empty() ? 0.0 : std::sqrt(sum / pcm.size());

            if (callback_count.load() % 30 == 0) {
                std::cout << "[MicTest] callback=" << callback_count.load()
                          << ", pts_ms=" << pts_ms
                          << ", samples=" << pcm.size()
                          << ", rms=" << rms
                          << ", max_abs=" << max_abs
                          << std::endl;
            }
        }
    );

    if (!capture.start()) {
        std::cerr << "[MicTest] failed to start capture." << std::endl;
        return -1;
    }

    std::cout << "[MicTest] started. Speak to microphone. Press Ctrl+C to stop.\n";

    auto start_time = std::chrono::steady_clock::now();

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto now = std::chrono::steady_clock::now();
        auto sec = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (sec >= 10) {
            break;
        }
    }

    capture.stop();

    std::cout << "[MicTest] total callbacks=" << callback_count.load()
              << ", total_samples=" << total_samples.load()
              << ", total_seconds="
              << static_cast<double>(total_samples.load()) / sample_rate
              << std::endl;

    return 0;
}