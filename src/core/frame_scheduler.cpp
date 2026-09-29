#include <iostream>
#include <cmath>
#include <deque>
#include <mutex>

#include "core/frame_scheduler.h"

namespace DigitalHuman {
namespace Core {

using DigitalHuman::Video::VideoFrame;

struct FrameScheduler::Impl {
    // 配置参数
    double target_fps;
    // 抖动缓冲阈值
    size_t min_buffer_size;
    // 同步容忍度，设置为1帧的时间
    const double SYN_THRESHOLD = 40.0;

    // 状态信息
    std::deque<VideoFrame> queue;
    cv::Mat last_frame;             // 上一帧缓存
    bool is_buffering = true;       // 判断当前是否处于缓冲状态

    int dropped_count = 0;
    mutable std::mutex mtx;

    Impl(double fps, size_t buf_size) : target_fps(fps), min_buffer_size(buf_size) {
        // 这里初始化为一个黑色底图，防止一开始就getFrame导致空指针崩溃
        last_frame = cv::Mat::zeros(96, 96, CV_8UC3);
    }

    void pushFrame(const VideoFrame& frame) {
        // 加锁
        std::lock_guard<std::mutex> lock(mtx);

        // 防止推理过快
        if (queue.size() > 100) {
            queue.pop_front();
            dropped_count++;
        }

        // 压入队列
        queue.push_back(frame);

        // 如果处于缓冲状态，检查空间是否足够
        if (is_buffering && queue.size() >= min_buffer_size) {
            is_buffering = false;   // 可以开始播放了，不需要再缓冲了
            std::cout << "[Scheduler] Jitter buffer full. Start playing." << std::endl;
        }
    }

    cv::Mat getFrameForRender(double audio_time_ms) {
        std::lock_guard<std::mutex> lock(mtx);

        // 缓冲策略
        if (queue.empty()) {
            is_buffering = true;
            return last_frame;  // 如果缓冲队列空了，返回上一帧的数据，避免黑屏
        }

        // 同步策略
        while (!queue.empty()) {
            VideoFrame& head = queue.front();

            // Diff = 视频的计划时间 - 音频的实际时间
            // > 0 代表视频快乐
            // < 0 代表视频慢了
            double diff = head.pts - audio_time_ms;

            // 策略1 滞后处理
            if (diff < -SYN_THRESHOLD) {
                queue.pop_front();  // 马上丢弃这一帧，马上检查下一帧是否能够追上
                dropped_count++;
                continue;
            }

            // 策略2 超前处理
            if (diff > SYN_THRESHOLD) {
                // 保持当前状态，即保持上一帧，不消耗队列
                return last_frame;
            }

            //策略3 正常处理
            last_frame = head.image.clone();    // 更新上一帧的缓存
            queue.pop_front();                  // 消费当前帧

            return last_frame;
        }

        return last_frame;  // 如果在同步时，所有帧都过期了，那么也是返回上一帧
    }

    void reset() {
        // 重置状态
        std::lock_guard<std::mutex> lock(mtx);
        queue.clear();
        dropped_count = 0;
        is_buffering = true;
        last_frame = cv::Mat::zeros(96, 96, CV_8UC3);
    }

    // 统计接口
    size_t getQueueSize() const {
        std::lock_guard<std::mutex> lock(mtx);
        return queue.size();
    }

    int getDroppedCount() const {
        std::lock_guard<std::mutex> lock(mtx);
        return dropped_count;
    }

    bool isBuffering() const {
        std::lock_guard<std::mutex> lock(mtx);
        return is_buffering;
    }

};


FrameScheduler::FrameScheduler(double fps, size_t buf_size) 
    : pImpl(std::make_unique<Impl>(fps, buf_size)){}

FrameScheduler::~FrameScheduler() = default;

void FrameScheduler::pushFrame(const VideoFrame& frame) {
    pImpl->pushFrame(frame);
}

cv::Mat FrameScheduler::getFrameForRender(double audio_time_ms) {
    return pImpl->getFrameForRender(audio_time_ms);
}

void FrameScheduler::reset() {
    pImpl->reset();
}

size_t FrameScheduler::getQueueSize() const { 
    return pImpl->getQueueSize(); 
}

int FrameScheduler::getDroppedCount() const { 
    return pImpl->getDroppedCount(); 
}

bool FrameScheduler::isBuffering() const { 
    return pImpl->isBuffering(); 
}

} // namespace Core
} // namespace DigitalHuman