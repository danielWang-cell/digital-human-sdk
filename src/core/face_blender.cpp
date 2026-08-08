#include "core/face_blender.h"

#include <opencv2/imgproc.hpp>

namespace DigitalHuman {
namespace Core {

struct FaceBlender::Impl {
    cv::Mat sharpen96(const cv::Mat& generated_96) const {
        if (generated_96.empty()) {
            return generated_96;
        }
        // 先做一张轻微模糊图
        cv::Mat blur;
        cv::GaussianBlur(generated_96, blur, cv::Size(0, 0), 1.0);

        // 原图 * 1.25 - 模糊图 * 0.25，可以增强边缘细节
        cv::Mat sharp;
        cv::addWeighted(generated_96, 1.25, blur, -0.25, 0.0, sharp);

        return sharp;
    }
    
    cv::Mat restoreToOriginal(const cv::Mat& generated_96,
                              const cv::Mat& inverse_affine,
                              const cv::Size& output_size) const {
        // 先创建一张和原图一样大的黑圈
        cv::Mat restored = cv::Mat::zeros(output_size, CV_8UC3);

        if (generated_96.empty() || inverse_affine.empty()) {
            return restored;
        }

        // 用 M_inv 把 96 x 96 生成图贴回原图位置
        cv::warpAffine(
            generated_96,
            restored,
            inverse_affine,
            output_size,
            cv::INTER_CUBIC,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );

        return restored;
    }

    cv::Mat restoreMaskToOriginal(const cv::Mat& mask_96_3c,
                                  const cv::Mat& inverse_affine,
                                  const cv::Size& output_size) const {
        cv::Mat restored = cv::Mat::zeros(output_size, CV_32FC3);

        if (mask_96_3c.empty() || inverse_affine.empty()) {
            return restored;
        }
        // mask 使用线性插值即可，避免边缘过硬
        cv::warpAffine(
            mask_96_3c,
            restored,
            inverse_affine,
            output_size,
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );

        // 再模糊一次，消除 warp 后可能出现的锯齿
        cv::GaussianBlur(restored, restored, cv::Size(7, 7), 0);

        return restored;
    }

    cv::Mat blendWithDetail(const cv::Mat& base_bgr,
                            const cv::Mat& restored_face_bgr,
                            const cv::Mat& restored_mask_3c) const {
        if (base_bgr.empty() || restored_face_bgr.empty() || restored_mask_3c.empty()) {
            return base_bgr.clone();
        }
        
        cv::Mat base_f;
        cv::Mat gen_f;
        cv::Mat mask_f;

        // 转 float 是为了做 0-1 的 alpha 混合
        base_bgr.convertTo(base_f, CV_32FC3);
        restored_face_bgr.convertTo(gen_f, CV_32FC3);
        restored_mask_3c.convertTo(mask_f, CV_32FC3);

        // 防止 mask 数值越界
        cv::min(mask_f, cv::Scalar(1.0, 1.0, 1.0), mask_f);
        cv::max(mask_f, cv::Scalar(0.0, 0.0, 0.0), mask_f);

        // 标准 alpha 融合
        cv::Mat blended_f = gen_f.mul(mask_f) + base_f.mul(cv::Scalar(1.0, 1.0, 1.0) - mask_f);

        // 提取原图高频细节
        // base_f - blur(base_f) 得到纹理，边缘等细节
        cv::Mat base_blur;
        cv::GaussianBlur(base_f, base_blur, cv::Size(0, 0), 1.2);

        cv::Mat detail = base_f - base_blur;

        // 只在 mask 区域少量加回细节
        // 强度不能太大，否则原图闭嘴纹理会压过生成嘴型
        const double detail_strength = 0.12;
        blended_f = blended_f + detail.mul(mask_f) * detail_strength;

        cv::Mat blended_u8;
        blended_f.convertTo(blended_u8, CV_8UC3);

        return blended_u8;

    }
};

FaceBlender::FaceBlender() : pImpl(std::make_unique<Impl>()) {}
FaceBlender::~FaceBlender() = default;

FaceBlender::FaceBlender(FaceBlender&&) noexcept = default;

FaceBlender& FaceBlender::operator=(FaceBlender&&) noexcept = default;

cv::Mat FaceBlender::sharpen96(const cv::Mat& generated_96) const {
    return pImpl->sharpen96(generated_96);
}

cv::Mat FaceBlender::restoreToOriginal(const cv::Mat& generated_96,
                                       const cv::Mat& inverse_affine,
                                       const cv::Size& output_size) const {
    return pImpl->restoreToOriginal(generated_96, inverse_affine, output_size);
}

cv::Mat FaceBlender::restoreMaskToOriginal(const cv::Mat& mask_96_3c,
                                           const cv::Mat& inverse_affine,
                                           const cv::Size& output_size) const {
    return pImpl->restoreMaskToOriginal(mask_96_3c, inverse_affine, output_size);                                        
}

cv::Mat FaceBlender::blendWithDetail(const cv::Mat& base_bgr,
                                     const cv::Mat& restored_face_bgr,
                                     const cv::Mat& restored_mask_3c) const {
    return pImpl->blendWithDetail(base_bgr, restored_face_bgr, restored_mask_3c);
}

} // namespace Core
} // namespace DigitalHuman