#include "core/face_aligner.h"
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <ratio>
#include <sys/stat.h>
#include <system_error>

namespace DigitalHuman {
namespace Core {

/**
 * @brief 安全读取 cv::Mat 中的数值，并统一转换成 double
 * @param m 输入矩阵
 * @param r 行号
 * @param c 列号
 * @return double 类型数值 
 * 
 * 这个函数的作用是：
 *      OpenCV 的仿射矩阵可能是 CV_64F，也可能是 CV_32F
 *      如果矩阵是 CV_64F，需要用：
 *               m.at<double>(r, c)
 *      如果矩阵是 CV_32F，需要用：
 *               m.at<float>(r, c)
 *      直接写死 at<double>()，遇到 CV_32F 时可能读取错误
 *  本函数将两种情况统一封装，方便后续 transformLandmarks() 使用
 */
double matAtAsDouble(const cv::Mat& m, int r, int c) {
    return m.depth() == CV_64F ? m.at<double>(r, c) : static_cast<double>(m.at<float>(r, c));
}

struct FaceAligner::Impl {
    // 计算两点间的中心
    cv::Point2f getCenter(const std::vector<cv::Point>& points) {
        cv::Point2f center(0.0f, 0.0f);
        if (points.empty()) {
            return center;
        }

        for (const auto& p : points) {
            center.x += static_cast<float>(p.x);
            center.y += static_cast<float>(p.y);
        }

        center.x /= static_cast<float>(points.size());
        center.y /= static_cast<float>(points.size());
        return center;
    }
    // 基于左右眼进行人脸对齐逻辑
    cv::Mat alignByEyes(const cv::Mat& image, const std::vector<cv::Point>& landmarks, int target_size) {
        if (image.empty() || landmarks.size() != 68) {
            std::cerr << "[FaceAligner] Error: alignByEyes Invalid input." << std::endl;
            return cv::Mat();
        }

        // 1. 提取眼睛关键点索引
        //   左眼：36，37，38，39，40，41
        //   右眼：42，43，44，45，46，47
        std::vector<cv::Point> left_eye_pts, right_eye_pts;
        for (int i = 36; i <= 41; ++i) {
            left_eye_pts.push_back(landmarks[i]);
        }

        for (int i = 42; i <= 47; ++i) {
            right_eye_pts.push_back(landmarks[i]);
        }

        // 2. 计算左右眼中心
        cv::Point2f left_eye_center = getCenter(left_eye_pts);
        cv::Point2f right_eye_center = getCenter(right_eye_pts);

        // 3. 计算旋转角度（让双眼连线水平）
        // dx，dy
        float dy = right_eye_center.y - left_eye_center.y;
        float dx = right_eye_center.x - left_eye_center.x;
        // 计算角度（弧度 -> 角度）
        double angle = std::atan2(dy, dx) * 180.0 / CV_PI;

        // 4. 计算缩放比例
        double desired_dist = target_size * 0.4; //希望对齐后，两眼之间的距离大约占 target_size 的 40%
        double current_dist = std::sqrt(dx * dx + dy * dy);
        if (current_dist <= 1e-6) {
            return cv::Mat();
        }
        double scale = desired_dist / current_dist;

        // 5. 计算变换中心，以双眼连线的中点为旋转中心
        cv::Point2f eyes_center((left_eye_center.x + right_eye_center.x) * 0.5f,
                                (left_eye_center.y + right_eye_center.y) * 0.5f);
        
        // 6. 获取旋转矩阵（2 x 3）
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
        return aligned_face;
    }
    
    // 基于人脸框进行对齐，并返回完整的变换结果
    FaceAlignmentResult alignByRect(const cv::Mat& image, 
                                    const cv::Rect& face_rect,
                                    const std::vector<cv::Point>& landmarks,
                                    int target_size,
                                    double padding_ratio) {
    
        FaceAlignmentResult result;
        if (image.empty() || face_rect.width <= 0 || face_rect.height <= 0) {
            std::cerr << "[FaceAligner] Error: alignByRect Invalid input." << std::endl;
            return result;
        }

        // 保证人脸框不要越界，如果 face_rect 超出图像范围，使用 & 操作和整张图范围求交集
        cv::Rect safe_rect = face_rect & cv::Rect(0, 0, image.cols, image.rows);
        if (safe_rect.width <= 0 || safe_rect.height <= 0) {
            return result;
        }

        // 计算 padding
        int padding = static_cast<int>(safe_rect.width * padding_ratio);
        int side = std::max(safe_rect.width, safe_rect.height) + 2 * padding;
        side = std::max(side, 32);

        // 人脸框中心点
        double cx = safe_rect.x + safe_rect.width / 2.0;
        double cy = safe_rect.y + safe_rect.height / 2.0;
        double scale = static_cast<double>(target_size) / static_cast<double>(side);

        // 构造原图到对齐图的仿射矩阵 M
        cv::Mat M = cv::Mat::zeros(2, 3, CV_64F);
        M.at<double>(0, 0) = scale;
        M.at<double>(1, 1) = scale;
        M.at<double>(0, 2) = -(cx - side / 2.0) * scale;
        M.at<double>(1, 2) = -(cy - side / 2.0) * scale;

        // 计算逆仿射变换矩阵
        cv::Mat M_inv;
        cv::invertAffineTransform(M, M_inv);

        // 进行仿射变换
        cv::Mat aligned;
        cv::warpAffine(
            image,
            aligned,
            M,
            cv::Size(target_size, target_size),
            cv::INTER_CUBIC,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );

        // 组装结果
        result.valid = !aligned.empty();
        result.aligned_face = aligned;
        result.M = M;
        result.M_inv = M_inv;
        result.raw_rect = safe_rect;

        // 将原图 landmarks 同步变换到 aligned 坐标系
        result.landmarks_aligned = transformLandmarks(landmarks, M);

        return result;
    }

    // 使用仿射矩阵变换一组关键点，这里不会改变图像，只是改变关键点坐标
    std::vector<cv::Point2f> transformLandmarks(const std::vector<cv::Point>& landmarks, 
                                                const cv::Mat& affine) const {
        std::vector<cv::Point2f> out;
        out.reserve(landmarks.size());

        if (affine.empty() || affine.rows != 2 || affine.cols != 3) {
            return out;
        }

        for (const auto& p : landmarks) {
            double x = 
                matAtAsDouble(affine, 0, 0) * p.x +
                matAtAsDouble(affine, 0, 1) * p.y +
                matAtAsDouble(affine, 0, 2);


            double y = 
                matAtAsDouble(affine, 1, 0) * p.x +
                matAtAsDouble(affine, 1, 1) * p.y +
                matAtAsDouble(affine, 1, 2);
            
            out.emplace_back(static_cast<float>(x), static_cast<float>(y));

        }

        return out;
    }
   
};

FaceAligner::FaceAligner() : pImpl(std::make_unique<Impl>()) {}
FaceAligner::~FaceAligner() = default;
FaceAligner::FaceAligner(FaceAligner&&) noexcept = default;
FaceAligner& FaceAligner::operator=(FaceAligner&&) noexcept = default;

cv::Mat FaceAligner::align(const cv::Mat& image, const std::vector<cv::Point>& landmarks, int target_size) {
    return pImpl->alignByEyes(image, landmarks, target_size);
}

FaceAlignmentResult FaceAligner::alignByRect(const cv::Mat& image,
                                             const cv::Rect& face_rect,
                                             const std::vector<cv::Point>& landmarks,
                                             int target_size,
                                             double padding_ratio) {
    return pImpl->alignByRect(image, face_rect, landmarks, target_size, padding_ratio);
}

std::vector<cv::Point2f> FaceAligner::transformLandmarks(const std::vector<cv::Point>& landmarks,
                                                         const cv::Mat& affine) const {
    return pImpl->transformLandmarks(landmarks, affine);
}


} // namespace Core
} // namespace DigitalHuman