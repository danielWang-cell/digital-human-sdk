#include <iostream>
#include <cmath>
#include <atomic>

#include "core/timestamp_manager.h"

namespace DigitalHuman {
namespace Core {

struct TimestampManager::Impl {
    // 使用原子变量
    std::atomic<double> audio_clock{0.0}; // 当前音频播放时间 (ms)
    std::atomic<double> video_clock{0.0}; // 当前视频生成时间 (ms)
    
    int64_t video_frame_count = 0;
    
    // 阈值配置
    const double SYNC_THRESHOLD = 40.0;    // 40ms 内认为同步 (人眼阈值)
    const double MAX_LAG = 500.0;          // 滞后超过 500ms 触发重置
    const double DROP_THRESHOLD = -40.0;   // 滞后超过 40ms 建议丢帧

    // 统计信息
    double current_diff = 0.0;

    void reset() {
        audio_clock = 0.0;
        video_clock = 0.0;
        video_frame_count = 0;
        current_diff = 0.0;
    }

    void updateAudioTime(int64_t sample_count, int sample_rate) {
        if (sample_rate <= 0) {
            return;
        }

        // 计算公式: (样本数 / 采样率) * 1000 = 毫秒
        audio_clock = (static_cast<double>(sample_count) / sample_rate) * 1000.0;
    }

    double getNextVideoPTS(double frame_rate) {
        if (frame_rate <= 0) {
            return 0.0;
        }

        // 简单递增：帧号 * 帧间隔
        // 更高级的实现可以用 PID 调节，但这里先用线性递增
        double pts = (static_cast<double>(video_frame_count) / frame_rate) * 1000.0;
        video_frame_count++;
        video_clock = pts;
        return pts;
    }

    SyncAction checkSync(double video_pts) {
        double audio_pts = audio_clock.load();
        
        // Diff = Video - Audio
        // 正值：视频快了 ，跑在音频前面
        // 负值：视频慢了 ，落在音频后面
        double diff = video_pts - audio_pts;
        current_diff = diff;

        if (std::abs(diff) <= SYNC_THRESHOLD) {
            return SyncAction::Render; // 在允许误差范围内，正常显示
        }

        if (diff > SYNC_THRESHOLD) {
            // 视频比音频快太多 -> 等待音频追上来
            return SyncAction::Wait;
        }

        if (diff < -MAX_LAG) {
            // 视频严重滞后 -> 重置或硬追赶
            return SyncAction::Reset; 
        }
        
        if (diff < DROP_THRESHOLD) {
            // 视频慢了 -> 丢掉这一帧，直接算下一帧
            return SyncAction::Drop;
        }

        return SyncAction::Render; // 默认
    }
};


TimestampManager::TimestampManager() : pImpl(std::make_unique<Impl>()) {}
TimestampManager::~TimestampManager() = default;

void TimestampManager::reset() {
    pImpl->reset();
}

void TimestampManager::updateAudioTime(int64_t sample_count, int sample_rate) {
    pImpl->updateAudioTime(sample_count, sample_rate);
}

double TimestampManager::getNextVideoPTS(double frame_rate) {
    return pImpl->getNextVideoPTS(frame_rate);
}

SyncAction TimestampManager::checkSync(double video_pts) {
    return pImpl->checkSync(video_pts);
}

double TimestampManager::getDiff() const {
    return pImpl->current_diff;
}

} // namespace Core
} // namespace DigitalHuman