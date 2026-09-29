#pragma once

#include <cstdint>
#include <opencv2/core.hpp>
#include <vector>

namespace DigitalHuman::Core {

// Common per-frame data shared by Wav2Lip, MuseTalk and future trackers.
struct FrameContext {
    cv::Mat source_frame;
    std::int64_t frame_index = 0;
    double pts_ms = 0.0;
    cv::Rect face_roi;
    std::vector<cv::Point2f> landmarks;
    cv::Mat align_matrix;
    cv::Mat inverse_align_matrix;
    cv::Rect mouth_roi;
    cv::Mat mouth_mask;
    bool tracking_valid = false;
};

} // namespace DigitalHuman::Core
