#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include "video/video_frame.h"

namespace DigitalHuman {
namespace Core {

using DigitalHuman::Video::VideoFrame;

/**
 * @brief 帧调度器，根据音频时钟，从队列中选出最合适的视频帧进行渲染
 * 实现音画同步、掉帧处理(Drop)、等待处理(Wait)、抖动缓冲(Jitter Buffer)
 */
class FrameScheduler {
public:
    /**
     * @brief 构造函数
     * @param target_fps 目标帧率，默认25fps
     * @param buffer_size 抖动缓冲大小，默认是3帧
     */
    explicit FrameScheduler(double target_fps = 25.0, size_t buffer_size = 3);
    ~FrameScheduler();

    FrameScheduler(const FrameScheduler&) = delete;
    FrameScheduler& operator=(const FrameScheduler&) = delete;

    /**
     * @brief 推入新生成的推理帧，作为生产者
     * @param frame 包含图像和PTS的视频帧
     */
    virtual void pushFrame(const VideoFrame& frame);

    /**
     * @brief 获取当前应该显示的帧
     * @param audio_time_ms 当前音频播放的时间，这是基准时间
     * @param cv::Mat 要显示的图像，可能是新帧，也可能是上一帧的缓存
     */
    cv::Mat getFrameForRender(double audio_time_ms);

    /**
     * @brief 重置状态，清空队列，重置计数器
     */
    void reset();

    /**
     * @brief 一些常见的状态函数
     */
    // 查看当前积压了多少帧
    size_t getQueueSize() const;
    // 累计丢帧数
    int getDroppedCount() const;
    // 是否处于缓冲状态
    bool isBuffering() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Core
} // namespace DigitalHuman