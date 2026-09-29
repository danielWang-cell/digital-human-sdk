#pragma once

#include <memory>
#include <chrono>

namespace DigitalHuman {
namespace Core {

/**
 * @brief 同步状态枚举
 */
enum class SyncAction {
    Render,     // 正常渲染
    Drop,       // 丢帧 ，视频滞后
    Wait,       // 等待 ，视频超前
    Reset       // 重置 ，偏差过大
};

/**
 * @brief 时间戳管理器，负责生成单调时间戳、计算偏差并提供同步建议
 */
class TimestampManager {
public:
    TimestampManager();
    ~TimestampManager();

    TimestampManager(const TimestampManager&) = delete;
    TimestampManager& operator=(const TimestampManager&) = delete;

    /**
     * @brief 重置时钟 
     */
    void reset();

    /**
     * @brief 更新音频播放进度 ，由音频回调调用
     * @param sample_count 已播放的样本数
     * @param sample_rate 采样率
     */
    void updateAudioTime(int64_t sample_count, int sample_rate);

    /**
     * @brief 获取下一个视频帧的目标时间戳
     * @param frame_rate 视频帧率 (如 25 fps)
     * @return 毫秒级时间戳
     */
    double getNextVideoPTS(double frame_rate);

    /**
     * @brief 计算当前同步状态
     * @param video_pts 当前准备渲染的视频帧 PTS (ms)
     * @return SyncAction 同步建议
     */
    SyncAction checkSync(double video_pts);

    /**
     * @brief 获取当前偏差统计 ，这里主要用来调试
     * @return 偏差值 ms (Video - Audio)
     */
    double getDiff() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Core
} // namespace DigitalHuman