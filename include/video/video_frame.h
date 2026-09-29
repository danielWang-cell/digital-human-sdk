#pragma once
#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Video {

/**
 * @brief 视频帧结构体，包含图像数据、显示时间戳和帧序号
 */
struct VideoFrame {
    cv::Mat image;      // 图像数据 (BGR)
    double pts = 0.0;   // 显示时间戳PTS
    int64_t index = 0;  // 帧序号 (从0开始递增)

    cv::Mat background;     // 原始高清底图 
    cv::Rect roi;           // 前景图在底图中的融合区域坐标
    
    // 帧是否有效
    bool isValid() const { 
        return !image.empty(); 
    }
    
};

} // namespace Video
} // namespace DigitalHuman