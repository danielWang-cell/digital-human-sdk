#include "core/face_aligner.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>

namespace DigitalHuman {
namespace Core {

struct FaceAligner::Impl {
    // 计算两点间的中心
    cv::Point2f getCenter(const std::vector<cv::Point>& points) {
        float x = 0, y = 0;
        for (const auto& p : points) {
            x += p.x;
            y += p.y;
        }
        return cv::Point2f(x / points.size(), y / points.size());
    }

    // 执行对齐逻辑
    cv::Mat align(const cv::Mat& image, const std::vector<cv::Point>& landmarks, int target_size) {
        if (image.empty() || landmarks.size() != 68) {
            std::cerr << "[FaceAligner] Error: Invalid input." << std::endl;
            return cv::Mat();
        }

        // 1. 提取眼睛关键点索引
        // 左眼: 36-41, 右眼: 42-47
        std::vector<cv::Point> left_eye_pts, right_eye_pts;
        for (int i = 36; i <= 41; ++i) left_eye_pts.push_back(landmarks[i]);
        for (int i = 42; i <= 47; ++i) right_eye_pts.push_back(landmarks[i]);

        // 2. 计算左右眼中心
        cv::Point2f left_eye_center = getCenter(left_eye_pts);
        cv::Point2f right_eye_center = getCenter(right_eye_pts);

        // 3. 计算旋转角度 (让双眼连线水平)
        // dy, dx
        float dy = right_eye_center.y - left_eye_center.y;
        float dx = right_eye_center.x - left_eye_center.x;
        // 计算角度 (弧度 -> 角度)
        double angle = std::atan2(dy, dx) * 180.0 / CV_PI;

        // 4. 计算缩放比例
        double desired_dist = target_size * 0.4; // 目标眼距
        double current_dist = std::sqrt(dx*dx + dy*dy);
        double scale = desired_dist / current_dist;

        // 5. 计算变换中心，以双眼连线的中点为旋转中心
        cv::Point2f eyes_center((left_eye_center.x + right_eye_center.x) / 2.0f,
                                (left_eye_center.y + right_eye_center.y) / 2.0f);

        // 6. 获取旋转矩阵 (2x3)
        // getRotationMatrix2D 会生成一个绕 eyes_center 旋转并缩放的矩阵
        cv::Mat M = cv::getRotationMatrix2D(eyes_center, angle, scale);

        // 7. 调整平移量 (Translation)
        double tx = target_size * 0.5;
        double ty = target_size * 0.4;
        
        M.at<double>(0, 2) += (tx - eyes_center.x);
        M.at<double>(1, 2) += (ty - eyes_center.y);

        // 8. 应用仿射变换 (Crop + Resize + Rotate)
        cv::Mat aligned_face;
        cv::warpAffine(image, aligned_face, M, cv::Size(target_size, target_size), 
                       cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        // 9. 归一化 (0-255 -> -1.0-1.0)
        // 转换为浮点型
        cv::Mat normalized_face;
        
        return aligned_face;
    }
};

FaceAligner::FaceAligner() : pImpl(std::make_unique<Impl>()) {}
FaceAligner::~FaceAligner() = default;
FaceAligner::FaceAligner(FaceAligner&&) noexcept = default;
FaceAligner& FaceAligner::operator=(FaceAligner&&) noexcept = default;

cv::Mat FaceAligner::align(const cv::Mat& image, const std::vector<cv::Point>& landmarks, int target_size) {
    return pImpl->align(image, landmarks, target_size);
}

} // namespace Core
} // namespace DigitalHuman