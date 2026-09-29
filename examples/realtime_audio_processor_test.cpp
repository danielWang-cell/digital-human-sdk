#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <iostream>
#include <thread>
#include <vector>

#include "audio/audio_process.h"
#include "audio/ffmpeg_pulse_capture.h"

using DigitalHuman::Audio::AudioProcessor;
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

    std::atomic<int> mel_count{0};

    AudioProcessor audio_processor(
        [&](double pts_ms, std::vector<float> mel_features) {
            int idx = mel_count.fetch_add(1);

            if (idx % 25 == 0) {
                float min_v = mel_features.empty() ? 0.0f : mel_features[0];
                float max_v = mel_features.empty() ? 0.0f : mel_features[0];
                double sum = 0.0;

                for (float v : mel_features) {
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                    sum += v;
                }

                double mean = mel_features.empty() ? 0.0 : sum / mel_features.size();

                std::cout << "[MelTest] idx=" << idx
                          << ", pts_ms=" << pts_ms
                          << ", size=" << mel_features.size()
                          << ", min=" << min_v
                          << ", max=" << max_v
                          << ", mean=" << mean
                          << std::endl;
            }
        },
        sample_rate
    );

    audio_processor.setRealtimeMode(true);
    audio_processor.start();

    FFmpegPulseCapture capture(
        source,
        sample_rate,
        chunk_samples,
        [&](const std::vector<float>& pcm, double pts_ms) {
            audio_processor.pushRawAudio(pcm, pts_ms);
        }
    );

    if (!capture.start()) {
        std::cerr << "[MelTest] failed to start capture." << std::endl;
        audio_processor.stop();
        return -1;
    }

    std::cout << "[MelTest] started. Speak to microphone. Press Ctrl+C to stop.\n";

    auto start_time = std::chrono::steady_clock::now();

    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        auto now = std::chrono::steady_clock::now();
        auto sec = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        if (sec >= 15) {
            break;
        }
    }

    capture.stop();

    audio_processor.markInputFinished();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    audio_processor.stop();

    std::cout << "[MelTest] total mel chunks=" << mel_count.load() << std::endl;

    return 0;
}