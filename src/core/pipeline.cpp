#include "core/pipeline.h"
#include "utils/thread_safe_queue.h"
#include "audio/audio_video_Synchronous.h"
#include "core/frame_scheduler.h"
#include "video/video_frame.h"

#include "model/inference_process.h"
#include "video/render_process.h"

#include <ncnn/net.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <filesystem>

namespace DigitalHuman {
namespace Core {

// 实时模式下，如果将所有的推理结果都塞进 render_queue_，会导致队列满载
class PipelineFrameScheduler : public DigitalHuman::Core::FrameScheduler {
public:
    explicit PipelineFrameScheduler(::ThreadSafeQueue<Video::VideoFrame>& q)
        : render_queue_(q) {}

    void setRealtimeDropOldFrames(bool enabled) {
        realtime_drop_old_frames_.store(enabled, std::memory_order_release);
    }

    void pushFrame(const Video::VideoFrame& frame) override {
        // 实时预览只保留最新帧
        if (realtime_drop_old_frames_.load(std::memory_order_acquire)) {
            render_queue_.clear();
        }
        render_queue_.push(frame);
    }

private:
    ::ThreadSafeQueue<Video::VideoFrame>& render_queue_;
    std::atomic<bool> realtime_drop_old_frames_{false};
};

struct Pipeline::Impl {
    std::atomic<bool> is_running_{false};
    std::atomic<bool> has_error_{false};
    std::string error_message_;

    ::ThreadSafeQueue<InferenceTask> task_queue_{50};
    ::ThreadSafeQueue<Video::VideoFrame> render_queue_{50};

    Audio::AudioPlayer audio_player_;
    ncnn::Net wav2lip_net_;
    PipelineFrameScheduler scheduler_adapter_;

    std::unique_ptr<Model::InferenceProcessor> inference_engine_;
    std::unique_ptr<Video::RenderProcessor> render_engine_;
    std::thread watchdog_thread_;

    bool is_offline_mode_ = false;          // 是否为离线模式
    bool static_face_cache_enabled_ = true; // 是否开启人脸缓存，默认为 true 因为静态图+实时语音也需要
    bool realtime_preview_mode_ = false;    // 是否开启预览模式，实时预览时，渲染端不等 AudioPlayer PTS ，队列丢弃旧帧
    bool use_audio_player_clock_ = false;   // 是否使用 AudioPlayer ，在 PulseAudio FFmpeg 实时采集模式时应该为 false
    PipelineConfig config_{};

    Impl() : scheduler_adapter_(render_queue_) {
        inference_engine_ = std::make_unique<Model::InferenceProcessor>(task_queue_, scheduler_adapter_);

        auto audio_cb = [this]() -> double {
            return audio_player_.getCurrentTime();
        };
        render_engine_ = std::make_unique<Video::RenderProcessor>(render_queue_, audio_cb);
    }

    ~Impl() {
        stop();
    }

    void triggerEmergencyStop(const std::string& reason) {
        if (!has_error_.exchange(true)) {
            error_message_ = reason;
            std::cerr << "\033[31m[CRITICAL] Emergency Stop: "
                      << reason << "\033[0m" << std::endl;
        }
    }

    bool start(const std::string& model_dir, bool headless, const std::string& output_video_path) {
        if (is_running_.load()) {
            return true;
        }

        
        task_queue_.clear();
        render_queue_.clear();
        task_queue_.reset();
        render_queue_.reset();

        // 设置 scheduler 的实时丢帧模式
        scheduler_adapter_.setRealtimeDropOldFrames(realtime_preview_mode_);

        has_error_.store(false);
        error_message_.clear();

        // 这是判断是离线模式还是在线实时模式
        // 如果是离线模式则使用静态人脸缓存，如果是实时模式则关闭缓存
        is_offline_mode_ = !output_video_path.empty();

        // Support both the packaged layout and the repository's flat ncnn
        // model filenames, so callers only need to pass the models directory.
        std::string param_path = model_dir + "/wav2lip/wav2lip.param";
        std::string bin_path   = model_dir + "/wav2lip/wav2lip.bin";
        if (!std::filesystem::exists(param_path) || !std::filesystem::exists(bin_path)) {
            const std::string flat_param = model_dir + "/wav2lip.ncnn.param";
            const std::string flat_bin   = model_dir + "/wav2lip.ncnn.bin";
            if (std::filesystem::exists(flat_param) && std::filesystem::exists(flat_bin)) {
                param_path = flat_param;
                bin_path = flat_bin;
            }
        }

        wav2lip_net_.clear();

        // 首轮使用 CPU + FP32，与已经验证成功的 test_wav2lip_ncnn 保持一致
        wav2lip_net_.opt.use_vulkan_compute = false;
        wav2lip_net_.opt.num_threads = 4;
        wav2lip_net_.opt.use_fp16_packed = false;
        wav2lip_net_.opt.use_fp16_storage = false;
        wav2lip_net_.opt.use_fp16_arithmetic = false;
        wav2lip_net_.opt.lightmode = true;

        if (wav2lip_net_.load_param(param_path.c_str()) != 0 ||
            wav2lip_net_.load_model(bin_path.c_str()) != 0) {
            triggerEmergencyStop("无法加载 Wav2Lip 模型，请检查路径: " + model_dir);
            return false;
        }

        inference_engine_->setInferenceThreads(config_.inference_threads);
        inference_engine_->setInferenceOptions(config_.use_vulkan, config_.use_fp16, config_.light_mode);
        inference_engine_->setSharpening(config_.enable_sharpen, config_.sharpen_strength);
        if (!inference_engine_->initWav2Lip(&wav2lip_net_, false)) {
            triggerEmergencyStop("InferenceProcessor 初始化失败。");
            return false;
        }

        // 这里设置模型的类型与工作模式
        // 由 is_offline_mode_ 控制是离线模式还是在线模式
        inference_engine_->setModelType(Model::InferenceModelType::Wav2Lip);
        inference_engine_->setInferenceThreads(config_.inference_threads);
        inference_engine_->setSharpening(config_.enable_sharpen, config_.sharpen_strength);
        inference_engine_->setStaticFaceCacheEnabled(static_face_cache_enabled_);

        //------------------调试日志-------------
        std::cout << "[Pipeline] mode: "
                << (is_offline_mode_ ? "offline" : "realtime")
                << ", static_face_cache="
                << (static_face_cache_enabled_ ? "ON" : "OFF")
                << ", realtime_preview="
                << (realtime_preview_mode_ ? "ON" : "OFF")
                << ", use_audio_player_clock="
                << (use_audio_player_clock_ ? "ON" : "OFF")
                << std::endl;

        // -------------------------------------

        if (!is_offline_mode_ && use_audio_player_clock_) {
            if (!audio_player_.open(16000, 1, 512)) {
                triggerEmergencyStop("音频设备初始化失败。");
                return false;
            }
            audio_player_.play();

            std::cout << "[Pipeline] AudioPlayer enabled." << std::endl;
        } else {
            std::cout << "[Pipeline] AudioPlayer disabled. "
                    << "Render will not use AudioPlayer clock." << std::endl;
        }

        // 设置实时预览模式
        // realtime_preview_mode_ = true时，RenderProcessor应该
        //          1、不去等待 AudioPlayer PTS
        //          2、拿到推理帧之后尽快显示
        //          3、配合 scheduler 丢弃旧渲染帧
        if (render_engine_) {
            render_engine_->setRealtimePreviewMode(realtime_preview_mode_);
        }

        if (!inference_engine_->start()) {
            triggerEmergencyStop("推理线程启动失败。");
            return false;
        }

        if (!render_engine_->start(headless, output_video_path)) {
            inference_engine_->stop();
            triggerEmergencyStop("渲染线程启动失败。");
            return false;
        }

        is_running_.store(true);
        watchdog_thread_ = std::thread(&Impl::watchdogLoop, this);
        return true;
    }

    void stop() {
        if (!is_running_.exchange(false)) {
            return;
        }

        has_error_.store(false);

        // 先关闭输入队列，唤醒可能阻塞的推理线程
        task_queue_.shutdown();

        if (inference_engine_) {
            inference_engine_->stop();
        }

        if (render_engine_) {
            render_engine_->stop(true);
        }

        if (!is_offline_mode_) {
            audio_player_.stop();
        }

        if (watchdog_thread_.joinable()) {
            watchdog_thread_.join();
        }
    }

    bool pushTask(const InferenceTask& task) {
        if (!is_running_.load() || has_error_.load()) {
            return false;
        }
        return task_queue_.push(task);
    }

    void pushAudioData(const std::vector<float>& pcm_data) {
        if (is_running_.load() && !is_offline_mode_) {
            audio_player_.pushData(pcm_data);
        }
    }

    bool isHealthy() const {
        return is_running_.load() && !has_error_.load();
    }

    // 设置人脸缓存配置函数
    void setStaticFaceCacheEnabled(bool enabled) {
        static_face_cache_enabled_ = enabled;

        if (inference_engine_) {
            inference_engine_->setStaticFaceCacheEnabled(enabled);
        }

        std::cout << "[Pipeline] setStaticFaceCacheEnabled: "
              << (enabled ? "ON" : "OFF") << std::endl;
    }

    // 设置一个预览模式
    void setRealtimePreviewMode(bool enabled) {
        realtime_preview_mode_ = enabled;

        scheduler_adapter_.setRealtimeDropOldFrames(enabled);

        if (render_engine_) {
            render_engine_->setRealtimePreviewMode(enabled);
        }

        std::cout << "[Pipeline] setRealtimePreviewMode: "
              << (enabled ? "ON" : "OFF") << std::endl;
    }

    void setUseAudioPlayerClock(bool enabled) {
        use_audio_player_clock_ = enabled;

        std::cout << "[Pipeline] setUseAudioPlayerClock: "
                << (enabled ? "ON" : "OFF") << std::endl;
    }

    void setConfig(const PipelineConfig& config) {
        config_ = config;
        if (inference_engine_) {
            inference_engine_->setInferenceThreads(config_.inference_threads);
            inference_engine_->setInferenceOptions(config_.use_vulkan, config_.use_fp16, config_.light_mode);
            inference_engine_->setSharpening(config_.enable_sharpen, config_.sharpen_strength);
        }
    }

    PipelineConfig getConfig() const { return config_; }

    size_t getTaskQueueSize() const {
        return task_queue_.size();
    }

    size_t getRenderQueueSize() const {
        return render_queue_.size();
    }

    void watchdogLoop() {
        while (is_running_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::seconds(2));

            if (task_queue_.size() >= 48) {
                std::cerr << "\033[33m[Watchdog] 警告: 推理队列积压满载！\033[0m" << std::endl;
            }
            if (render_queue_.size() >= 48) {
                std::cerr << "\033[33m[Watchdog] 警告: 渲染队列积压满载！\033[0m" << std::endl;
            }
        }
    }
};

Pipeline::Pipeline() : pImpl(std::make_unique<Impl>()) {}
Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&&) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&&) noexcept = default;

bool Pipeline::start(const std::string& model_dir, bool headless, const std::string& output_video_path) {
    return pImpl->start(model_dir, headless, output_video_path);
}

void Pipeline::stop() {
    pImpl->stop();
}

bool Pipeline::pushTask(const InferenceTask& task) {
    return pImpl->pushTask(task);
}

void Pipeline::pushAudioData(const std::vector<float>& pcm_data) {
    pImpl->pushAudioData(pcm_data);
}

bool Pipeline::isHealthy() const {
    return pImpl->isHealthy();
}

size_t Pipeline::getTaskQueueSize() const {
    return pImpl->getTaskQueueSize();
}

size_t Pipeline::getRenderQueueSize() const {
    return pImpl->getRenderQueueSize();
}

void Pipeline::setStaticFaceCacheEnabled(bool enabled) {
    pImpl->setStaticFaceCacheEnabled(enabled);
}

void Pipeline::setRealtimePreviewMode(bool enabled) {
    pImpl->setRealtimePreviewMode(enabled);
}

void Pipeline::setUseAudioPlayerClock(bool enabled) {
    pImpl->setUseAudioPlayerClock(enabled);
}

void Pipeline::setConfig(const PipelineConfig& config) {
    pImpl->setConfig(config);
}

PipelineConfig Pipeline::getConfig() const {
    return pImpl->getConfig();
}

} // namespace Core
} // namespace DigitalHuman
