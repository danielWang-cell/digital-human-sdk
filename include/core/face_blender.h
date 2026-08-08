#ifndef DIGITAL_HUMAN_CORE_FACE_BLENDER_H_
#define DIGITAL_HUMAN_CORE_FACE_BLENDER_H_

#include <memory>               
#include <opencv2/core.hpp>    
/**
 * @brief 人脸融合模块
 * 负责将 Wav2Lip 生成的 96 x 96 嘴部结果贴回原图
 * 这个类不做人脸检测、不做人脸对齐、不做模型推理，只负责图像后处理：
 * 主要完成的工作有：
 *      1. 对生成的 96 x 96 图像轻微锐化；
 *      2. 使用反向仿射矩阵 M_inv 将生成图贴回原图；
 *      3. 将 96 x 96 mask 同样贴回原图；
 *      4. 使用 alpha mask 做平滑融合；
 *      5. 少量恢复原图细节，减轻生成区域的模糊感。
 */
namespace DigitalHuman {
namespace Core {

class FaceBlender {
public: 
    FaceBlender();
    ~FaceBlender();

    FaceBlender(FaceBlender&&) noexcept;
    FaceBlender& operator=(FaceBlender&&) noexcept;

    FaceBlender(const FaceBlender&) = delete;
    FaceBlender& operator=(const FaceBlender&) = delete;

    /**
     * @brief 对 Wav2Lip 输出的 96 x 96 图像做轻微锐化
     */
    cv::Mat sharpen96(const cv::Mat& generated_96) const;

    /**
     * @brief 将 96 x 96 生成图反变换回原图尺寸
     * @param generated_96 Wav2Lip 输出的 96 x 96 图像
     * @param inverse_affine FaceAligner 生成的 M_inv
     * @param output_size 原始图像尺寸
     */
    cv::Mat restoreToOriginal(const cv::Mat& generated_96, 
                              const cv::Mat& inverse_affine,
                              const cv::Size& output_size) const;
    
    /**
     * @brief 将 96 x 96 三通道 mask 反变换回原图尺寸
     */
    cv::Mat restoreMaskToOriginal(const cv::Mat& mask_96_3c,
                                  const cv::Mat& inverse_affine,
                                  const cv::Size& output_size) const;
    
    /**
     * @brief 使用 mask 将生成嘴部融合回原图
     */
    cv::Mat blendWithDetail(const cv::Mat& base_bgr,
                            const cv::Mat& restored_face_bgr,
                            const cv::Mat& restored_mask_3c) const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Core
} // namespace DigitalHuman
#endif // DIGITAL_HUMAN_CORE_FACE_BLENDER_H_
