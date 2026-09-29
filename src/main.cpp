#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <iomanip>

#include <opencv2/opencv.hpp>
#include <portaudio.h>

#include "core/pipeline.h"
#include "audio/audio_process.h"
#include "core/face_detector.h" // 引入人脸诊断

using namespace DigitalHuman::Core;
using namespace DigitalHuman::Audio;

// 计时器辅助类
struct Timer {
    std::chrono::steady_clock::time_point start;
    std::string name;

    Timer(std::string n) : name(n), start(std::chrono::steady_clock::now()) {
    }

    void stop() {
        auto end = std::chrono::steady_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << "[Timer] " << name << " 耗时: " << diff << " ms" << std::endl;
    }
};

std::vector<float> parseWavToFloat(const std::string& filepath) {
    Timer t("音频解析 (WAV -> Float)");
    std::ifstream file(filepath, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "无法打开音频文件: " << filepath << std::endl;
        return {};
    }

    // 简单跳过 WAV header，适用于标准 PCM WAV
    file.seekg(44, std::ios::beg);
    std::vector<int16_t> pcm16;
    int16_t sample = 0;

    while (file.read(reinterpret_cast<char*>(&sample), sizeof(int16_t))) {
        pcm16.push_back(sample);
    }

    std::vector<float> pcm_float(pcm16.size());
    for (size_t i = 0; i < pcm16.size(); ++i) {
        pcm_float[i] = static_cast<float>(pcm16[i]) / 32768.0f;
    }

    t.stop();
    return pcm_float;
}

int main(int argc, char** argv) {
    std::cout << "========== Digital Human 诊断与推理引擎 ==========\n" << std::endl;

    if (argc < 4) {
        std::cout << "用法: ./bin/digital-human-app <图片> <音频> <输出.mp4> [模型目录]\n";
        return -1;
    }

    const std::string face_path    = argv[1];
    const std::string audio_input  = argv[2];
    const std::string video_output = argv[3];
    // 因为在 build 下运行，默认上一级目录寻找 models
    const std::string model_dir    = (argc >= 5) ? argv[4] : "../models"; 

    // 1. 加载图片并进行人脸诊断
    cv::Mat base_face = cv::imread(face_path);

    if (base_face.empty()) {
        std::cerr << "致命错误: 无法加载底图 " << face_path << std::endl;
        return -1;
    }

    std::cout << "[Step 1] 正在进行人脸特征点诊断..." << std::endl;
    {
        FaceDetector detector;
        std::string landmark_model = model_dir + "/shape_predictor_68_face_landmarks.dat";

        if (!detector.loadLandmarkModel(landmark_model)) {
            std::cerr << "警告: 无法加载关键点模型: " << landmark_model << std::endl;
        } else {
            auto rects = detector.detect(base_face);

            if (rects.empty()) {
                std::cerr << "致命错误: 未在图中检测到人脸！请检查图片内容。" << std::endl;
                return -1;
            }

            cv::Mat debug_img = base_face.clone();
            auto landmarks = detector.getLandmarks(base_face, rects[0]);
            
            // 绘制诊断信息
            cv::rectangle(debug_img, rects[0], cv::Scalar(0, 255, 0), 2);
            for (int i = 0; i < landmarks.size(); ++i) {
                // 嘴部点 (48-67) 用红色高亮，其他用绿色
                cv::Scalar color = (i >= 48 && i <= 67) ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
                cv::circle(debug_img, landmarks[i], 2, color, -1);

                if (i == 48 || i == 54 || i == 51 || i == 57) {
                    cv::putText(debug_img, std::to_string(i), landmarks[i], cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
                }
            }

            // 直接存放在当前工作目录 
            std::string debug_path = "debug_face_detection.jpg";
            cv::imwrite(debug_path, debug_img);
            std::cout << "[Success] 诊断图已生成: " << debug_path << " (请检查红色点是否准确覆盖嘴唇)" << std::endl;
        }
    }

    // 2. 启动流水线
    Timer pipeline_timer("全链路推理与渲染");
    Pipeline pipeline;

    if (const char* cache_env = std::getenv("DH_STATIC_FACE_CACHE")) {
        pipeline.setStaticFaceCacheEnabled(std::string(cache_env) != "0");
    }
    
    // 使用 avi 格式，并直接存放在当前工作目录 (build 目录下)
    const std::string temp_silent_video = "temp_silent.avi";

    if (!pipeline.start(model_dir, true, temp_silent_video)) {
        std::cerr << "无法启动 Pipeline" << std::endl;
        return -1;
    }

    // 3. 配置音频处理
    AudioProcessor audio_processor([&](double pts_ms, std::vector<float> mel_features) {
        InferenceTask task;
        task.pts_ms = pts_ms;
        task.audio_feature = std::move(mel_features);
        task.base_face = base_face;   // 静态图片采用浅拷贝就行
        pipeline.pushTask(task);
    }, 16000);

    audio_processor.start();

    // 4. 读取音频并推送数据
    std::vector<float> full_pcm_data = parseWavToFloat(audio_input);

    if (!full_pcm_data.empty()) {
        const int CHUNK_SIZE = 512;
        const int AUDIO_SAMPLE_RATE = 16000;

        double current_pts = 0.0;

        for (size_t i = 0; i < full_pcm_data.size(); i += CHUNK_SIZE) {
            size_t end = std::min(i + CHUNK_SIZE, full_pcm_data.size());

            std::vector<float> chunk(
                full_pcm_data.begin() + i,
                full_pcm_data.begin() + end
            );

            audio_processor.pushRawAudio(chunk, current_pts);

            current_pts +=
                (static_cast<double>(chunk.size()) / AUDIO_SAMPLE_RATE) * 1000.0;
        }
    }

    // 通知 AudioProcessor 音频已经全部推送完毕。
    // 这样 audio_process.cpp 才会执行 flushRemainingVideoFrames()，
    // 对尾部不足 16 帧 Mel 的部分做 padding，补齐最后 3 帧。
    audio_processor.markInputFinished();

    // 5. 循环监控进度
    std::cout << "\n[System] 正在执行 AI 推理..." << std::endl;

    while (pipeline.isHealthy()) {
        size_t t_size = pipeline.getTaskQueueSize();
        size_t r_size = pipeline.getRenderQueueSize();
        bool drained = audio_processor.isDrained();

        std::cout << "\r[Progress] 推理中: " << std::setw(3) << t_size 
                  << " | 渲染中: " << std::setw(3) << r_size 
                  << " | 音频完毕: " << (drained ? "是" : "否") << "    " << std::flush;

        if (t_size == 0 && r_size == 0 && drained) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    pipeline_timer.stop();

    audio_processor.stop();
    pipeline.stop();

    // 6. 最终合并
    std::cout << "\n[Step 3] 正在合并音视频..." << std::endl;
    Timer merge_timer("FFmpeg 音视频合并");
    
    // 使用 libx264 编码，确保最终视频兼容性，不报绿屏
    const std::string ffmpeg_cmd =
        "ffmpeg -y -i \"" + temp_silent_video + "\" -i \"" + audio_input +
        "\" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest \"" +
        video_output + "\" > /dev/null 2>&1";
    
    int ret = std::system(ffmpeg_cmd.c_str());
    merge_timer.stop();

    if (ret == 0) {
        // 安全清理产生的临时 AVI 文件
        std::remove(temp_silent_video.c_str());
        std::cout << "\n\033[32m[DONE] 运行成功！请查看: " << video_output << "\033[0m\n" << std::endl;
    } else {
        std::cerr << "\n\033[31m[ERROR] 合并失败，请检查是否安装了 ffmpeg\033[0m\n" << std::endl;
    }

    return 0;
}
