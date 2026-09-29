#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>

#include "video/render_process.h"
#include "utils/thread_safe_queue.h"

namespace DigitalHuman {
namespace Video {

struct RenderProcessor::Impl {
    ::ThreadSafeQueue<VideoFrame>& render_queue_;
    std::function<double()> audio_clock_cb_;

    std::thread worker_thread_;
    std::atomic<bool> is_running_{false};
    std::atomic<bool> flush_on_exit_{true};
    std::atomic<bool> is_headless_{false};
    std::atomic<bool> realtime_preview_mode_{false};        // 预览模式

    std::string output_video_path_;
    bool is_offline_mode_ = false;
    cv::VideoWriter video_writer_;

    const double SYNC_THRESHOLD_DROP_MS = -60.0;

    int frame_count_ = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> fps_start_time_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    bool first_frame_logged_ = false;

    Impl(::ThreadSafeQueue<VideoFrame>& queue, std::function<double()> clock_cb)
        : render_queue_(queue), audio_clock_cb_(std::move(clock_cb)) {}

    ~Impl() { stop(false); }

    bool start(bool headless, const std::string& output_video_path) {
        if (is_running_.load()) {
            return true;
        }

        is_headless_.store(headless);
        output_video_path_ = output_video_path;
        is_offline_mode_ = !output_video_path_.empty();

        if (!is_offline_mode_ && !audio_clock_cb_) {
            std::cerr << "[Render] 错误: 实时模式下未绑定音频时钟回调！" << std::endl;
            return false;
        }

        is_running_.store(true);
        flush_on_exit_.store(true);
        fps_start_time_ = std::chrono::high_resolution_clock::now();
        start_time_ = std::chrono::steady_clock::now();
        first_frame_logged_ = false;
        worker_thread_ = std::thread(&Impl::processLoop, this);

        std::cout << "[RenderProcessor] 启动成功. 模式: "
                  << (is_offline_mode_ ? "离线视频导出" : "实时流媒体") << std::endl;
        return true;
    }

    void stop(bool flush) {
        if (!is_running_.exchange(false)) {
            return;
        }

        flush_on_exit_.store(flush);
        render_queue_.shutdown();

        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

        if (video_writer_.isOpened()) {
            video_writer_.release();
        }

        if (!is_headless_.load()) {
            try {
                cv::destroyAllWindows();
            } catch (...) {
            }
        }
        std::cout << "[RenderProcessor] 渲染线程已安全退出。" << std::endl;
    }

    void setRealtimePreviewMode(bool enabled) {
        realtime_preview_mode_.store(enabled, std::memory_order_release);

        std::cout << "[RenderProcessor] realtime_preview_mode="
              << (enabled ? "ON" : "OFF") << std::endl;
    }
    cv::Mat blendImagesFast(const cv::Mat& background, const cv::Mat& mouth, const cv::Rect& roi) {
        if (background.empty() || mouth.empty() || roi.width <= 0 || roi.height <= 0) {
            return background.clone();
        }

        cv::Mat result = background.clone();
        cv::Rect safe_roi = roi & cv::Rect(0, 0, background.cols, background.rows);
        if (safe_roi.width <= 0 || safe_roi.height <= 0) {
            return result;
        }

        if (safe_roi.width != mouth.cols || safe_roi.height != mouth.rows) {
            cv::Mat resized_mouth;
            cv::resize(mouth, resized_mouth, safe_roi.size());
            resized_mouth.copyTo(result(safe_roi));
            return result;
        }

        cv::Mat mask = cv::Mat::zeros(mouth.size(), CV_32FC3);
        cv::ellipse(mask, cv::Point(mouth.cols / 2, mouth.rows / 2),
                    cv::Size(std::max(1, mouth.cols / 2 - 2), std::max(1, mouth.rows / 2 - 2)),
                    0, 0, 360, cv::Scalar(1.0, 1.0, 1.0), -1, cv::LINE_AA);
        cv::GaussianBlur(mask, mask, cv::Size(11, 11), 5.0);

        cv::Mat bg_roi = result(safe_roi);
        cv::Mat mouth_f, bg_f;
        mouth.convertTo(mouth_f, CV_32FC3);
        bg_roi.convertTo(bg_f, CV_32FC3);

        cv::Mat out_f;
        cv::multiply(mouth_f, mask, mouth_f);
        cv::multiply(bg_f, cv::Scalar(1.0, 1.0, 1.0) - mask, bg_f);
        cv::add(mouth_f, bg_f, out_f);
        out_f.convertTo(bg_roi, CV_8UC3);
        return result;
    }

    void processLoop() {
        while (true) {
            if (!is_running_.load() && !flush_on_exit_.load()) {
                break;
            }

            auto frame_opt = render_queue_.pop(20);
            if (!frame_opt.has_value()) {
                if (!is_running_.load()) {
                    break;
                }
                continue;
            }

            VideoFrame frame = std::move(frame_opt.value());
            if (frame.image.empty()) {
                continue;
            }

            // 实时模式的同步逻辑
            // 如果离线模式 is_offline_mode_ = true时不走这
            const bool realtime_preview = realtime_preview_mode_.load(std::memory_order_acquire);


            if (!is_offline_mode_ && !realtime_preview) {
                const double current_audio_pts =  audio_clock_cb_ ? audio_clock_cb_() : 0.0;
                const double pts_diff = frame.pts - current_audio_pts;
                if (pts_diff > 5.0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(pts_diff)));
                } else if (pts_diff < SYNC_THRESHOLD_DROP_MS && is_running_.load()) {
                    continue;
                }
            }

            // 合成最终帧
            cv::Mat final_display = frame.background.empty()
                ? frame.image
                : blendImagesFast(frame.background, frame.image, frame.roi);

            if (final_display.empty()) {
                continue;
            }

            if (!first_frame_logged_) {
                const auto first_frame_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start_time_).count();
                std::cout << "[Render] First frame latency: " << first_frame_ms << " ms" << std::endl;
                first_frame_logged_ = true;
            }

            // 离线模式
            if (is_offline_mode_) {
                if (!video_writer_.isOpened()) {
                    const int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                    if (!video_writer_.open(output_video_path_, codec, 25.0,
                                            cv::Size(final_display.cols, final_display.rows))) {
                        std::cerr << "[Render] Error: 无法打开输出视频文件: " << output_video_path_ << std::endl;
                    }
                }
                if (video_writer_.isOpened()) {
                    video_writer_.write(final_display);
                }
            }

            // 显示窗口
            // 实时预览模式下会直接显示最新推理帧
            if (!is_headless_.load()) {
                try {
                    cv::imshow("Digital Human Generating...", final_display);
                    int key = cv::waitKey(1);

                    // ESC 退出渲染线程
                    if (key == 27) {
                        is_running_.store(false);
                        break;
                    }
                } catch (...) {
                    std::cerr << "[Render] Warning: cv::imshow failed, switch to headless mode."
                          << std::endl;
                    is_headless_.store(true);
                }
            }

            // FPS 日志
            ++frame_count_;
            const auto now = std::chrono::high_resolution_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - fps_start_time_).count();
            if (elapsed >= 1) {
                if (is_offline_mode_) {
                    std::cout << "[Render] 离线渲染速度: "
                            << frame_count_
                            << " FPS (已处理PTS: "
                            << frame.pts
                            << " ms)"
                            << std::endl;
                } else if (realtime_preview) {
                    std::cout << "[Render] 实时预览 FPS: "
                            << frame_count_
                            << " | RenderQueue: "
                            << render_queue_.size()
                            << " | Frame PTS: "
                            << frame.pts
                            << " ms"
                            << std::endl;
                } else {
                    const double current_audio_pts =
                        audio_clock_cb_ ? audio_clock_cb_() : 0.0;

                    std::cout << "[Render] 实时 FPS: "
                            << frame_count_
                            << " | Audio PTS: "
                            << current_audio_pts
                            << " ms"
                            << std::endl;
                }

                frame_count_ = 0;
                fps_start_time_ = now;
            }
        }
        std::cout << "[Render] processLoop exit." << std::endl;
    }
};

RenderProcessor::RenderProcessor(::ThreadSafeQueue<VideoFrame>& render_queue,
                                 std::function<double()> audio_clock_cb)
    : pImpl(std::make_unique<Impl>(render_queue, std::move(audio_clock_cb))) {}

RenderProcessor::~RenderProcessor() = default;

bool RenderProcessor::start(bool headless, const std::string& output_video_path) {
    return pImpl->start(headless, output_video_path);
}

void RenderProcessor::stop(bool flush_queue) {
    pImpl->stop(flush_queue);
}

void RenderProcessor::setRealtimePreviewMode(bool enabled) {
    pImpl->setRealtimePreviewMode(enabled);
}

} // namespace Video
} // namespace DigitalHuman
