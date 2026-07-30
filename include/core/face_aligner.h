#include <type_traits>
#include <vector>
#include <memory>
#include <opencv2/core.hpp>

namespace DigitalHuman {
namespace Core {

/**
 * @brief 人脸对齐结果
 * 这个结构体保存一次人脸对齐需要的全部信息：
 *     1. aligned_face：裁剪并缩放后的 96 x 96 人脸图
 *     2. M：原图坐标 -> 96 x 96 对齐空间的仿射矩阵
 *     3. M_inv：96 x 96 对齐空间 -> 原图坐标的反向仿射矩阵
 *     4. landmarks_aligned：被映射到 96 x 96 空间中的 68 关键点
 */
struct FaceAlignmentResult {
    bool valid = false; // 本次对齐是否有效

    cv::Mat aligned_face; // 96 x 96 BGR 人脸图，给 Wav2Lip 使用
    cv::Mat M; // 原图 -> 96 x 96
    cv::Mat M_inv; // 96 x 96 -> 原图

    cv::Rect raw_rect; // 原始人脸检测框
    std::vector<cv::Point2f> landmarks_aligned; // 96 x 96 空间下的 68 点关键点
};

/**
 * @brief 人脸对齐与预处理模块
 *     主要负责的功能是：
 *         1. 根据人脸框裁剪人脸
 *         2. 将人脸缩放到模型所需要的大小
 *         3. 计算正向和反向仿射矩阵
 *         4. 把原图中的 landmarks 映射到对齐后的小图坐标中
 */
class FaceAligner {
public:
    FaceAligner();
    ~FaceAligner();

    FaceAligner(FaceAligner&&) noexcept;
    FaceAligner& operator=(FaceAligner&&) noexcept;

    FaceAligner(const FaceAligner&) = delete;
    FaceAligner& operator=(const FaceAligner&) = delete;
    /**
     * @brief 基于眼睛位置对齐人脸
     * @param image         输入原图，通常是 BGR 格式 cv::Mat
     * @param landmarks     68 点人脸关键点，坐标位于原图坐标系
     * @param target_size   输出对齐人脸尺寸，默认 96
     * @return cv::Mat      对齐后的人脸图像，尺寸为 target_size x target_size
     */
    cv::Mat align(const cv::Mat& image, const std::vector<cv::Point>& landmarks, int target_size = 96);

    /**
     * @brief 根据人脸框进行正方形裁剪和缩放
     * @param image 原始 BGR 图像   
     * @param face_rect FaceDetector 检测到的人脸框
     * @param landmarks 原图坐标下的 68 点关键点
     * @param target_size 模型输入尺寸，Wav2Lip 为 96
     * @param padding_ratio 人脸框外扩比例
     * @return FaceAlignmentResult
     */
    FaceAlignmentResult alignByRect(const cv::Mat& image,
                                    const cv::Rect& face_rect,
                                    const std::vector<cv::Point>& landmarks,
                                    int target_size = 96,
                                    double padding_ratio = 0.15);
    
    /**
     * @brief 将原图关键点通过仿射矩阵变换到新坐标系
     * @param landmarks     原图坐标系中的关键点
     * @param affine        2 x 3 仿射矩阵，通常是 M
     * @return std::vector<cv::Point2f> 
     */
    std::vector<cv::Point2f> transformLandmarks(const std::vector<cv::Point>& landmarks,
                                                const cv::Mat& affine) const;
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;

};

} // namespace Core
} // namespace DigitalHuman
