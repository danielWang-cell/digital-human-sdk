#pragma once

#include <memory>
#include <atomic>
#include <functional>
#include <opencv2/opencv.hpp>
#include "video/video_frame.h"

template <typename T>
class ThreadSafeQueue;

namespace DigitalHuman {
namespace Video {

/**
 * @class RenderProcessor
 * @brief 数字人渲染与音画同步引擎
 * @note 该类核心职责
 *      1、从推理队列提取带有 PTS (时间戳) 的口型画面
 *      2、这是是一个从设备，对齐 Audio Clock
 *      3、执行图像融合，将口型贴回底图
 *      4、驱动 OpenCV 的 GUI 刷新进行屏幕显示
 */
class RenderProcessor {
public:
    /**
     * @brief 构造渲染处理器
     * @param render_queue 存放待渲染帧的线程安全队列，因为推理线程是生产者队列
     * @param audio_clock_cb 音频主时钟回调
     */
    RenderProcessor(::ThreadSafeQueue<VideoFrame>& render_queue,
                    std::function<double()> audio_clock_cb);
    ~RenderProcessor();

    // 禁用拷贝与赋值操作
    RenderProcessor(const RenderProcessor&) = delete;
    RenderProcessor& operator=(const RenderProcessor&) = delete;

     /**
     * @brief 启动渲染后台线程
     * @param headless 是否开启无界面静默模式
     * @param output_video_path 若不为空，则进入离线视频生成模式（
     */
    bool start(bool headless = false, const std::string& output_video_path = "");

    /**
     * @brief 停止渲染线程并清理资源
     * @param flush_queue 是否等待队列中的剩余帧渲染完毕 
     */
    void stop(bool flush_queue = true);

    /**
     * @brief 设置预览模式
     */
    void setRealtimePreviewMode(bool enabled);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Video
} // namespace DigitalHuman