#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "audio/audio_process.h"
#include "audio/ffmpeg_pulse_capture.h"
#include "core/pipeline.h"
#include "core/face_detector.h"

using namespace DigitalHuman::Audio;
using namespace DigitalHuman::Core;

static std::atomic<bool> g_running{true};

void handleSignal(int) {
    g_running.store(false);
}

int main(int argc, char** argv) {
    std::cout << "========== Digital Human Realtime PulseAudio Test ==========\n";

    if (argc < 3) {
        std::cout << "用法:\n";
        std::cout << "  ./bin/digital_human_realtime_pulse_app <face.jpg> <models_dir> [source]\n\n";
        std::cout << "示例:\n";
        std::cout << "  ./bin/digital_human_realtime_pulse_app ../face_new.jpg ../models RDPSource\n";
        return -1;
    }

    const std::string face_path = argv[1];
    const std::string model_dir = argv[2];
    const std::string source_name = (argc >= 4) ? argv[3] : "RDPSource";

    constexpr int SAMPLE_RATE = 16000;
    constexpr int CHUNK_SAMPLES = 512;

    // 初始实时视觉延迟。后续根据嘴型跟随效果微调。
    constexpr double REALTIME_VISUAL_DELAY_MS = 320.0;

    std::signal(SIGINT, handleSignal);

    cv::Mat base_face = cv::imread(face_path);
    if (base_face.empty()) {
        std::cerr << "[Realtime] failed to load face image: " << face_path << std::endl;
        return -1;
    }

    {
        FaceDetector detector;
        std::string landmark_model =
            model_dir + "/shape_predictor_68_face_landmarks.dat";

        if (detector.loadLandmarkModel(landmark_model)) {
            auto faces = detector.detect(base_face);
            if (faces.empty()) {
                std::cerr << "[Realtime] no face detected in static image." << std::endl;
                return -1;
            }

            std::cout << "[Realtime] face detected: " << faces[0] << std::endl;
        } else {
            std::cerr << "[Realtime] warning: landmark model load failed." << std::endl;
        }
    }

    Pipeline pipeline;

    // 开启实时预览模式
    pipeline.setUseAudioPlayerClock(false);
    pipeline.setRealtimePreviewMode(true);
    pipeline.setStaticFaceCacheEnabled(true);

    // 实时模式：不传输出视频路径，直接窗口预览。
    if (!pipeline.start(model_dir, false, "")) {
        std::cerr << "[Realtime] pipeline start failed." << std::endl;
        return -1;
    }

    AudioProcessor audio_processor(
        [&](double pts_ms, std::vector<float> mel_features) {
            // 实时模式限流，防止推理队列无限积压。
            if (pipeline.getTaskQueueSize() >= 5) {
                return;
            }

            InferenceTask task;
            task.pts_ms = pts_ms + REALTIME_VISUAL_DELAY_MS;
            task.audio_feature = std::move(mel_features);

            // 静态图片实时驱动，浅拷贝即可。
            task.base_face = base_face;

            pipeline.pushTask(task);
        },
        SAMPLE_RATE
    );

    audio_processor.setRealtimeMode(true);
    audio_processor.start();

    FFmpegPulseCapture mic_capture(
        source_name,
        SAMPLE_RATE,
        CHUNK_SAMPLES,
        [&](const std::vector<float>& pcm, double pts_ms) {
            audio_processor.pushRawAudio(pcm, pts_ms);
        }
    );

    if (!mic_capture.start()) {
        std::cerr << "[Realtime] mic capture start failed." << std::endl;
        audio_processor.stop();
        pipeline.stop();
        return -1;
    }

    std::cout << "\n[Realtime] started.\n";
    std::cout << "source=" << source_name << "\n";
    std::cout << "Press Ctrl+C to stop.\n";

    auto last_log = std::chrono::steady_clock::now();

    while (g_running.load() && pipeline.isHealthy()) {
        auto now = std::chrono::steady_clock::now();

        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log).count() > 1000) {
            std::cout << "[Realtime] task_queue="
                      << pipeline.getTaskQueueSize()
                      << ", render_queue="
                      << pipeline.getRenderQueueSize()
                      << std::endl;

            last_log = now;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::cout << "\n[Realtime] stopping...\n";

    mic_capture.stop();

    audio_processor.markInputFinished();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    audio_processor.stop();
    pipeline.stop();

    std::cout << "[Realtime] done.\n";
    return 0;
}