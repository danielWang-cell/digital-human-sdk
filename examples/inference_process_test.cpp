#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cassert>
#include <atomic>
#include <string>

#include <opencv2/imgproc.hpp>

// 包含项目头文件
#include "model/inference_process.h"
#include "core/pipeline.h"
#include "utils/thread_safe_queue.h"
#include "core/frame_scheduler.h"
#include "video/video_frame.h"

// 包含 ncnn 库
#include <ncnn/net.h>

using namespace DigitalHuman::Core;
using namespace DigitalHuman::Model;
using namespace DigitalHuman::Video;

/**
 * @brief 模拟渲染调度器
 * 用于接收推理线程产生的真实结果并打印信息
 */
class MockFrameScheduler : public DigitalHuman::Core::FrameScheduler {
public:
    std::atomic<int> received_frames{0};
    double last_received_pts{-1.0};

    void pushFrame(const DigitalHuman::Video::VideoFrame& frame) override {
        received_frames++;
        last_received_pts = frame.pts;
        
        // 如果图像不为空，打印其真实尺寸
        std::string status = frame.image.empty() ? "空图像 (Fallback)" : "有效图像";
        std::cout << "[MockScheduler] 收到帧! PTS: " << frame.pts 
                  << " ms, 状态: " << status 
                  << " [" << frame.image.cols << "x" << frame.image.rows << "]" << std::endl;
    }
};

int main(int argc, char** argv) {
    std::cout << "========== Wav2Lip 模型集成测试开始 ==========" << std::endl;

    // 1. 设置模型路径 
    std::string model_dir = "../models/";
    if (argc > 1) {
        model_dir = argv[1]; 
    }

    std::string param_path = model_dir + "wav2lip.ncnn.param";
    std::string bin_path = model_dir + "wav2lip.ncnn.bin";

    std::cout << "[Test] 正在加载模型参数: " << param_path << std::endl;
    std::cout << "[Test] 正在加载模型权重: " << bin_path << std::endl;

    // 2. 初始化 ncnn::Net 并加载真实模型
    ncnn::Net wav2lip_net;
    
    // 开启 Vulkan 加速 (如果编译时支持且有 GPU)
    wav2lip_net.opt.use_vulkan_compute = true; 
    wav2lip_net.opt.num_threads = 4;

    int ret_param = wav2lip_net.load_param(param_path.c_str());
    int ret_bin = wav2lip_net.load_model(bin_path.c_str());

    if (ret_param != 0 || ret_bin != 0) {
        std::cerr << "\033[31m[Test] 致命错误: 无法加载 Wav2Lip 模型文件！请检查路径是否正确。\033[0m" << std::endl;
        return -1;
    }
    std::cout << "\033[32m[Test] Wav2Lip 模型加载成功！\033[0m" << std::endl;

    // 3. 初始化通信基础设施
    ::ThreadSafeQueue<DigitalHuman::Core::InferenceTask> input_queue(20);
    MockFrameScheduler scheduler;

    // 4. 创建并初始化推理处理器
    InferenceProcessor processor(input_queue, scheduler);
    
    // 将加载好的 net 传递给处理器
    if (!processor.initWav2Lip(&wav2lip_net, true)) {
        std::cerr << "[Test] 处理器初始化失败！" << std::endl;
        return -1;
    }

    // 5. 启动推理线程
    processor.setModelType(InferenceModelType::Wav2Lip);
    if (!processor.start()) {
        std::cerr << "[Test] 无法启动推理线程！" << std::endl;
        return -1;
    }

    // 6. 模拟推送推理任务 (模拟真实的 Wav2Lip 输入数据)
    std::cout << "[Test] 开始投递推理任务..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        InferenceTask task;
        task.pts_ms = i * 40.0;
        
        // 构造 Wav2Lip 标准输入特征 (1280 维)
        task.audio_feature.assign(1280, 0.01f * (i + 1)); 
        
        // 模拟一帧 256x256 的底图 (BGR 格式)
        task.base_face = cv::Mat(256, 256, CV_8UC3, cv::Scalar(100, 100, 100));
        // 在脸上画个圆，方便区分原始帧
        cv::circle(task.base_face, cv::Point(128, 128), 50, cv::Scalar(255, 0, 0), -1);

        input_queue.push(std::move(task));
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    // 7. 等待推理完成 (真实推理耗时较长，预留 2 秒)
    std::cout << "[Test] 任务推送完毕，等待模型执行..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // 8. 优雅停止
    std::cout << "[Test] 停止处理器..." << std::endl;
    processor.stop();

    // 9. 结果分析
    std::cout << "\n========== 测试结果分析 ==========" << std::endl;
    std::cout << "预期帧数: 10, 实际收到: " << scheduler.received_frames.load() << std::endl;
    
    if (scheduler.received_frames == 10) {
        std::cout << "\033[32m[Success] 推理线程模型集成测试圆满成功！\033[0m" << std::endl;
    } else {
        std::cerr << "\033[31m[Failed] 帧数不匹配，请检查推理过程中的 ERROR 日志。\033[0m" << std::endl;
    }

    return 0;
}
