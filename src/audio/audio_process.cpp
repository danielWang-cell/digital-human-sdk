#include <iostream>
#include <cmath>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <functional>
#include <cstring>
#include <algorithm>

#include "audio/audio_process.h"
#include "audio/audio_preprocessor.h"
#include "audio/audio_mel_feature_extract.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace DigitalHuman {
namespace Audio {

struct RawAudioBlock {
    std::vector<float> data;
    double start_pts;
};

struct AudioProcessor::Impl {
    AudioProcessor::FeatureCallback output_cb_;
    int sample_rate_;
    int frame_size_;
    int stride_size_;

    std::unique_ptr<AudioPreprocessor> preprocessor_;
    std::unique_ptr<MelFeatureExtractor> mel_extractor_;
    std::vector<float> hamming_window_;

    std::thread worker_thread_;
    std::atomic<bool> is_running_{false};
    std::atomic<bool> is_processing_{false};
    mutable std::mutex mtx_;
    std::condition_variable cv_;

    std::deque<RawAudioBlock> input_queue_;
    std::vector<float> pcm_stream_cache_;
    double current_stream_pts_ = 0.0;
    std::deque<std::vector<float>> mel_frame_buffer_;

    int video_frame_index_ = 0;   
    int total_mel_frames_ = 0; 

    // 离线输入结束标志
    std::atomic<bool> input_finished_{false};

    // 尾部 flush 是否已经执行过，防止重复补帧
    bool tail_flushed_ = false;

    // 累计输入音频采样点数，用于精确计算目标视频的帧数
    int64_t total_input_samples_ = 0;

    // 实时模式变量
    bool realtime_mode_ = false;
    size_t realtime_max_queue_blocks_ = 20;

    Impl(AudioProcessor::FeatureCallback cb, int sr)
        : output_cb_(std::move(cb)), sample_rate_(sr) {
        frame_size_ = static_cast<int>(sr * 50.0 / 1000.0);
        stride_size_ = static_cast<int>(sr * 12.5 / 1000.0);

        preprocessor_ = std::make_unique<AudioPreprocessor>();
        mel_extractor_ = std::make_unique<MelFeatureExtractor>(sr, 800, 80);

        hamming_window_.resize(frame_size_);
        for (int i = 0; i < frame_size_; ++i) {
            hamming_window_[i] = 0.54f - 0.46f * cosf(2.0f * M_PI * i / (frame_size_ - 1));
        }

        pcm_stream_cache_.reserve(sr * 2);
    }

    ~Impl() { 
        stop(); 
    }

    void markInputFinished() {
        std::cout << "[AudioProcessor] markInputFinished called." << std::endl;
        input_finished_.store(true, std::memory_order_release);
        cv_.notify_all();
    }

    bool start() {
        if (is_running_.load()) {
            return true;
        }

        {
            std::lock_guard<std::mutex> lock(mtx_);
            input_queue_.clear();
            pcm_stream_cache_.clear();
            mel_frame_buffer_.clear();
            
            // 重置时间轴变量
            video_frame_index_ = 0;   
            total_mel_frames_ = 0;    
            current_stream_pts_ = 0.0;
            input_finished_.store(false);
            tail_flushed_ = false;
            total_input_samples_ = 0;
        }

        is_processing_.store(false);
        is_running_.store(true);
        worker_thread_ = std::thread(&Impl::processLoop, this);
        std::cout << "[AudioProcess] Start running!" << std::endl;
        return true;
    }

    void stop() {
        if (!is_running_.exchange(false)) {
            return;
        }

        cv_.notify_all();
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
        std::cout << "[AudioProcess] Stop!" << std::endl;
    }

    void setRealtimeMode(bool enabled) {
        std::lock_guard<std::mutex> lock(mtx_);

        realtime_mode_ = enabled;

        // 切换模式时清理一下过长的队列，防止历史积压影响实时测试
        if (realtime_mode_) {
            while (input_queue_.size() > realtime_max_queue_blocks_) {
                input_queue_.pop_front();
            }
        }

        std::cout << "[AudioProcessor] realtime_mode="
              << (enabled ? "ON" : "OFF")
              << ", max_queue_blocks=" << realtime_max_queue_blocks_
              << std::endl;
    }

    void pushRawAudio(const std::vector<float>& pcm_data, double start_pts_ms) {
        if (!is_running_.load() || pcm_data.empty()) {
            return;
        }

        {
            std::lock_guard<std::mutex> lock(mtx_);

            // 实时模式：允许丢旧音频块，防止延迟累积
            // 离线模式：不能丢，否则会造成视频截断/少帧
            if (realtime_mode_) {
                while (input_queue_.size() >= realtime_max_queue_blocks_) {
                    input_queue_.pop_front();
                }
            }
            input_queue_.push_back({pcm_data, start_pts_ms});
        }
        
        cv_.notify_one();
    }

    bool isDrained() const {
        std::lock_guard<std::mutex> lock(mtx_);
        // 主线程只有在输入已结束 + 尾部补帧已执行 + 队列清空 + 当前没有处理时才认为音频处理完毕
        return input_finished_.load(std::memory_order_acquire) &&
                tail_flushed_ &&
                input_queue_.empty() &&
                !is_processing_.load();
    }

    // 80×16 chunk 生成逻辑
    std::vector<float> makeWav2LipChunkFromBuffer(int relative_idx, bool allow_padding) {
        std::vector<float> wav2lip_chunk;
        wav2lip_chunk.reserve(80 * 16);

        if (mel_frame_buffer_.empty()) {
            return wav2lip_chunk;
        }

        const int buffer_size = static_cast<int>(mel_frame_buffer_.size());

        for (int freq = 0; freq < 80; ++freq) {
            for (int time = 0; time < 16; ++time) {
                int idx = relative_idx + time;

                if (idx < 0) {
                    idx = 0;
                }

                if (idx >= buffer_size) {
                    if (!allow_padding) {
                        wav2lip_chunk.clear();
                        return wav2lip_chunk;
                    }

                    // 尾部 padding：复用最后一帧 Mel
                    idx = buffer_size - 1;
                }

                wav2lip_chunk.push_back(mel_frame_buffer_[idx][freq]);
            }
        }

        return wav2lip_chunk;
    }

    // 视频帧刷新逻辑
    void flushRemainingVideoFrames() {
        // 添加实时模式
        if (realtime_mode_) {
            tail_flushed_ = true;
            std::cout << "[AudioProcessor] realtime mode: skip tail flush." << std::endl;
            return;
        }

        std::cout << "[AudioProcessor] flushRemainingVideoFrames called. "
          << "mel_buffer=" << mel_frame_buffer_.size()
          << ", total_input_samples=" << total_input_samples_
          << ", video_frame_index=" << video_frame_index_
          << ", input_finished=" << input_finished_.load()
          << std::endl;

        if (tail_flushed_) {
            return;
        }

        if (mel_frame_buffer_.empty() || total_input_samples_ <= 0) {
            std::cout << "[AudioProcessor] Flush skipped: empty mel buffer or no input samples."
                    << std::endl;
            return;
        }

        const double video_frame_ms = 40.0; // 25 FPS

        double audio_duration_ms =
            static_cast<double>(total_input_samples_) / sample_rate_ * 1000.0;

        int target_total_video_frames =
            static_cast<int>(std::ceil(audio_duration_ms / video_frame_ms));

        int current_buffer_head_idx =
            total_mel_frames_ - static_cast<int>(mel_frame_buffer_.size());

        std::cout << "[AudioProcessor] Flush tail frames: "
                << "audio_duration_ms=" << audio_duration_ms
                << ", current_video_frames=" << video_frame_index_
                << ", target_total_video_frames=" << target_total_video_frames
                << ", total_mel_frames=" << total_mel_frames_
                << ", buffer_size=" << mel_frame_buffer_.size()
                << std::endl;

        while (video_frame_index_ < target_total_video_frames) {
            int expected_mel_start =
                static_cast<int>(video_frame_index_ * 3.2);

            int relative_idx = expected_mel_start - current_buffer_head_idx;

            if (relative_idx < 0) {
                relative_idx = 0;
            }

            if (relative_idx >= static_cast<int>(mel_frame_buffer_.size())) {
                relative_idx = static_cast<int>(mel_frame_buffer_.size()) - 1;
            }

            std::vector<float> wav2lip_chunk =
                makeWav2LipChunkFromBuffer(relative_idx, true);

            if (wav2lip_chunk.size() != 1280) {
                std::cerr << "[AudioProcessor] Flush failed: invalid chunk size="
                        << wav2lip_chunk.size() << std::endl;
                break;
            }

            double pts_ms = video_frame_index_ * video_frame_ms;

            if (output_cb_) {
                output_cb_(pts_ms, std::move(wav2lip_chunk));
            }

            std::cout << "[AudioProcessor] Flush video frame idx="
                    << video_frame_index_
                    << ", pts=" << pts_ms << " ms"
                    << std::endl;

            video_frame_index_++;
        }

        tail_flushed_ = true;
    }
    
    void processLoop() {
        while (is_running_.load(std::memory_order_acquire)) {
            std::vector<RawAudioBlock> blocks_to_process;
            {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait_for(lock, std::chrono::milliseconds(50), [this] {
                    return !input_queue_.empty() 
                            || input_finished_.load(std::memory_order_acquire)
                            || !is_running_.load();
                });

                if (!is_running_.load()) {
                    break;
                }
                if (input_queue_.empty()) {
                    if (input_finished_.load(std::memory_order_acquire)) {
                        break;
                    }
                    continue;
                }

                while (!input_queue_.empty()) {
                    blocks_to_process.push_back(std::move(input_queue_.front()));
                    input_queue_.pop_front();
                }
                is_processing_.store(true);
            }

            for (const auto& block : blocks_to_process) {
                if (pcm_stream_cache_.empty()) {
                    current_stream_pts_ = block.start_pts;
                }

                total_input_samples_ += static_cast<int64_t>(block.data.size());

                pcm_stream_cache_.insert(pcm_stream_cache_.end(), block.data.begin(), block.data.end());
            }

            while (pcm_stream_cache_.size() >= static_cast<size_t>(frame_size_)) {
                std::vector<float> current_frame(
                    pcm_stream_cache_.begin(),
                    pcm_stream_cache_.begin() + frame_size_);

                preprocessor_->preEmphasize(current_frame, 0.97f);

                for (int i = 0; i < frame_size_; ++i) {
                    current_frame[i] *= hamming_window_[i];
                }

                try {
                    cv::Mat mel_feature = mel_extractor_->extract(current_frame);
                    if (!mel_feature.empty()) {
                        std::vector<float> feature_vec(
                            mel_feature.begin<float>(),
                            mel_feature.end<float>());

                        // ================== 基于视频帧率的严格映射算法 ==================
                        mel_frame_buffer_.push_back(std::move(feature_vec));
                        total_mel_frames_++;  // 累加生成的总特征帧数

                        // 视频为 25FPS(40ms)，音频Hop=12.5ms。严格比例为 40 / 12.5 = 3.2
                        while (true) {
                            // 计算当前视频帧需要从哪个绝对 Mel 索引开始
                            int expected_mel_start = static_cast<int>(video_frame_index_ * 3.2);
                            
                            // 换算到当前双端队列 (deque) 中的相对索引
                            int current_buffer_head_idx = total_mel_frames_ - mel_frame_buffer_.size();
                            int relative_idx = expected_mel_start - current_buffer_head_idx;

                            // 如果当前队列中拥有该起始点，并且向后有足够的 16 帧
                            if (relative_idx >= 0 && relative_idx + 16 <= static_cast<int>(mel_frame_buffer_.size())) {
                                std::vector<float> wav2lip_chunk =
                                    makeWav2LipChunkFromBuffer(relative_idx, false);

                                if (wav2lip_chunk.size() != 1280) {
                                    break;
                                }

                                static int debug_mel_count = 0;
                                if (debug_mel_count < 10) {
                                    auto minmax = std::minmax_element(wav2lip_chunk.begin(), wav2lip_chunk.end());
                                    double sum = 0.0;
                                    for (float v : wav2lip_chunk) {
                                        sum += v;
                                    }

                                    std::cout << "[DEBUG MEL CHUNK] idx=" << debug_mel_count
                                            << " size=" << wav2lip_chunk.size()
                                            << " min=" << *minmax.first
                                            << " max=" << *minmax.second
                                            << " mean=" << sum / wav2lip_chunk.size()
                                            << std::endl;

                                    debug_mel_count++;
                                }

                                double pts_ms = video_frame_index_ * 40.0;
                                if (output_cb_) {
                                    output_cb_(pts_ms, std::move(wav2lip_chunk));
                                }

                                video_frame_index_++;
                            } else {
                                // 队列里数据不够 16 帧，或者还没达到视频需要的起始点，跳出等待下一轮音频流
                                break;
                            }
                        }

                        // 安全清理不再需要的旧数据，保证尾部 padding 时至少还有一个 mel 可用
                        int next_mel_start = static_cast<int>(video_frame_index_ * 3.2);
                        int current_buf_head = total_mel_frames_ - mel_frame_buffer_.size();
                        int pop_count = next_mel_start - current_buf_head;
                        
                        if (pop_count > 0) {
                            // 至少保留一帧，供尾部 padding 使用
                            int max_pop = std::max(0, static_cast<int>(mel_frame_buffer_.size()) - 1);
                            pop_count = std::min(pop_count, max_pop);

                            for (int i = 0; i < pop_count; i++) {
                                mel_frame_buffer_.pop_front();
                            }
                        }
                        // =================================================================================
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[AudioProcessor] Error extracting Mel feature: "
                              << e.what() << std::endl;
                }

                pcm_stream_cache_.erase(
                    pcm_stream_cache_.begin(),
                    pcm_stream_cache_.begin() + stride_size_);

                current_stream_pts_ +=
                    (static_cast<double>(stride_size_) / sample_rate_) * 1000.0;
            }


            static int status_count = 0;
            if (++status_count % 100 == 0) {
                std::cout << "[AudioProcessor] status. "
                        << "input_finished=" << input_finished_.load()
                        << ", input_queue=" << input_queue_.size()
                        << ", pcm_cache=" << pcm_stream_cache_.size()
                        << ", mel_buffer=" << mel_frame_buffer_.size()
                        << ", total_input_samples=" << total_input_samples_
                        << ", video_frame_index=" << video_frame_index_
                        << std::endl;
            }

            is_processing_.store(false);
        }
        flushRemainingVideoFrames();
        is_processing_.store(false);
    }
};

AudioProcessor::AudioProcessor(AudioProcessor::FeatureCallback output_cb, int sample_rate)
    : pImpl(std::make_unique<Impl>(std::move(output_cb), sample_rate)) {}

AudioProcessor::~AudioProcessor() = default;

bool AudioProcessor::start() {
    return pImpl->start();
}

void AudioProcessor::stop() {
    pImpl->stop();
}

void AudioProcessor::pushRawAudio(const std::vector<float>& pcm_data, double start_pts_ms) {
    pImpl->pushRawAudio(pcm_data, start_pts_ms);
}

bool AudioProcessor::isDrained() const {
    return pImpl->isDrained();
}

void AudioProcessor::markInputFinished() {
    pImpl->markInputFinished();
}

void AudioProcessor::setRealtimeMode(bool enabled) {
    pImpl->setRealtimeMode(enabled);
}

} // namespace Audio
} // namespace DigitalHuman