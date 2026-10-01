#include <iostream>
#include <thread>
#include <chrono>
#include <optional>
#include <cstring>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <ncnn/net.h>

#include "model/inference_process.h"
#include "core/face_mask_generator.h"
#include "core/face_detector.h"
#include "core/pipeline.h"
#include "utils/thread_safe_queue.h"
#include "core/frame_scheduler.h"
#include "video/video_frame.h"
#include "model/model_inference.h"
#include "model/output_processor.h"
#include "model/input_processor.h"
#include "model/wav2lip_backend.h"
#include "core/performance_metrics.h"

namespace fs = std::filesystem;

namespace DigitalHuman {
namespace Model {


// 只生成嘴唇及嘴周小范围 alpha，不再让 mask 覆盖整个下半脸
static cv::Mat buildMouthAlphaMask96(const std::vector<cv::Point2f>& landmarks_96) {
    CV_Assert(landmarks_96.size() >= 68);

    const cv::Size mask_size(96, 96);
    cv::Mat mask_u8 = cv::Mat::zeros(mask_size, CV_8UC1);

    std::vector<cv::Point> mouth_points;
    mouth_points.reserve(20);

    float mouth_x_min = landmarks_96[48].x;
    float mouth_x_max = landmarks_96[48].x;
    float mouth_y_min = landmarks_96[48].y;
    float mouth_y_max = landmarks_96[48].y;

    // 使用完整嘴部 landmarks：48-67
    for (int i = 48; i <= 67; ++i) {
        float x = landmarks_96[i].x;
        float y = landmarks_96[i].y;

        mouth_points.emplace_back(
            static_cast<int>(std::round(x)),
            static_cast<int>(std::round(y))
        );

        mouth_x_min = std::min(mouth_x_min, x);
        mouth_x_max = std::max(mouth_x_max, x);
        mouth_y_min = std::min(mouth_y_min, y);
        mouth_y_max = std::max(mouth_y_max, y);
    }

    std::vector<cv::Point> hull;
    cv::convexHull(mouth_points, hull);

    if (hull.size() < 3) {
        cv::Mat empty_mask;
        mask_u8.convertTo(empty_mask, CV_32FC1, 1.0 / 255.0);
        return empty_mask;
    }

    cv::fillConvexPoly(mask_u8, hull, cv::Scalar(255));

    /*
     * 适度膨胀：不能膨胀太多，否则 96×96 的生成结果会覆盖到下巴和脸颊，造成低分辨率区域扩大
     */
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(5, 5)
    );

    cv::dilate(mask_u8, mask_u8, kernel, cv::Point(-1, -1), 1);

    /*
     * 限制 mask 范围：只允许覆盖嘴部上下附近，不允许扩散到下巴。
     */
    const float mouth_w = std::max(1.0f, mouth_x_max - mouth_x_min);
    const float mouth_h = std::max(1.0f, mouth_y_max - mouth_y_min);

    int x_left = static_cast<int>(std::floor(mouth_x_min - 0.55f * mouth_w));
    int x_right = static_cast<int>(std::ceil(mouth_x_max + 0.55f * mouth_w));

    int y_top = static_cast<int>(std::floor(mouth_y_min - 0.90f * mouth_h));
    int y_bottom = static_cast<int>(std::ceil(mouth_y_max + 1.00f * mouth_h));

    x_left = std::max(0, x_left);
    x_right = std::min(95, x_right);
    y_top = std::max(0, y_top);
    y_bottom = std::min(95, y_bottom);

    if (x_left > 0) {
        mask_u8.colRange(0, x_left).setTo(0);
    }
    if (x_right + 1 < 96) {
        mask_u8.colRange(x_right + 1, 96).setTo(0);
    }
    if (y_top > 0) {
        mask_u8.rowRange(0, y_top).setTo(0);
    }
    if (y_bottom + 1 < 96) {
        mask_u8.rowRange(y_bottom + 1, 96).setTo(0);
    }

    /*
     * 防止 alpha 触碰 96×96 边界。
     * 如果 mask 接触边界，warp 回原图后容易形成框线。
     */
    const int border = 2;
    mask_u8.rowRange(0, border).setTo(0);
    mask_u8.rowRange(96 - border, 96).setTo(0);
    mask_u8.colRange(0, border).setTo(0);
    mask_u8.colRange(96 - border, 96).setTo(0);

    /*
     * 羽化边缘。这个值不能太小，否则边缘硬； 也不能太大，否则扩散到下巴。
     */
    cv::GaussianBlur(mask_u8, mask_u8, cv::Size(13, 13), 0);

    // blur 后再次清边界，避免边缘扩散
    mask_u8.rowRange(0, border).setTo(0);
    mask_u8.rowRange(96 - border, 96).setTo(0);
    mask_u8.colRange(0, border).setTo(0);
    mask_u8.colRange(96 - border, 96).setTo(0);

    cv::Mat mask_f;
    mask_u8.convertTo(mask_f, CV_32FC1, 1.0 / 255.0);

    return mask_f;
}

static cv::Mat blendRestoredFaceWithDetail(
    const cv::Mat& base_bgr,
    const cv::Mat& restored_face_bgr,
    const cv::Mat& restored_mask_3c
) {
    CV_Assert(!base_bgr.empty());
    CV_Assert(!restored_face_bgr.empty());
    CV_Assert(!restored_mask_3c.empty());

    cv::Mat base_f;
    cv::Mat gen_f;
    cv::Mat mask_f;

    base_bgr.convertTo(base_f, CV_32FC3);
    restored_face_bgr.convertTo(gen_f, CV_32FC3);
    restored_mask_3c.convertTo(mask_f, CV_32FC3);

    // 保证 mask 在 0~1 范围内
    cv::min(mask_f, cv::Scalar(1.0, 1.0, 1.0), mask_f);
    cv::max(mask_f, cv::Scalar(0.0, 0.0, 0.0), mask_f);

    cv::Mat blended_f =
        gen_f.mul(mask_f) +
        base_f.mul(cv::Scalar(1.0, 1.0, 1.0) - mask_f);

    /*
     * 高频细节回填：
     * 只在 mask 区域少量加回原图细节。
     * 参数不能太大，否则闭嘴原图纹理会压制生成嘴型。
     */
    cv::Mat base_blur;
    cv::GaussianBlur(base_f, base_blur, cv::Size(0, 0), 1.2);

    cv::Mat detail = base_f - base_blur;

    const double detail_strength = 0.12;
    blended_f = blended_f + detail.mul(mask_f) * detail_strength;

    cv::Mat blended_u8;
    blended_f.convertTo(blended_u8, CV_8UC3);

    return blended_u8;
}

static double matAtAsDouble(const cv::Mat& M, int r, int c) {
    if (M.depth() == CV_64F) {
        return M.at<double>(r, c);
    }
    return static_cast<double>(M.at<float>(r, c));
}

static std::vector<cv::Point2f> transformLandmarksTo96(
    const std::vector<cv::Point>& landmarks,
    const cv::Mat& M
) {
    std::vector<cv::Point2f> out;
    out.reserve(landmarks.size());

    for (const auto& p : landmarks) {
        const double x =
            matAtAsDouble(M, 0, 0) * static_cast<double>(p.x) +
            matAtAsDouble(M, 0, 1) * static_cast<double>(p.y) +
            matAtAsDouble(M, 0, 2);

        const double y =
            matAtAsDouble(M, 1, 0) * static_cast<double>(p.x) +
            matAtAsDouble(M, 1, 1) * static_cast<double>(p.y) +
            matAtAsDouble(M, 1, 2);

        out.emplace_back(static_cast<float>(x), static_cast<float>(y));
    }

    return out;
}

static std::string getMouthDebugDir() {
    // 程序通常从 /workspace/build 目录运行，
    // 因此 ./tmp 会对应 /workspace/build/tmp
    const fs::path debug_dir = fs::current_path() / "tmp";

    try {
        fs::create_directories(debug_dir);
    } catch (const std::exception& e) {
        std::cerr << "[MouthBlendDebug] failed to create debug dir: "
                  << debug_dir.string()
                  << ", error=" << e.what()
                  << std::endl;
    }

    return debug_dir.string();
}


struct InferenceProcessor::Impl {
    ::ThreadSafeQueue<Core::InferenceTask>& input_queue_;
    Core::FrameScheduler& frame_scheduler_;

    std::unique_ptr<ModelInference> wav2lip_engine_;
    std::unique_ptr<OutputProcessor> wav2lip_output_processor_;
    std::unique_ptr<InputProcessor> input_processor_;
    std::unique_ptr<Wav2LipBackend> wav2lip_backend_;
    Core::PerformanceMetrics metrics_;

    Core::FaceDetector face_detector_;
    bool dlib_ready_ = false;
    int inference_threads_ = 4;
    bool use_vulkan_ = false;
    bool use_fp16_ = false;
    bool light_mode_ = true;
    bool sharpening_enabled_ = false;
    double sharpening_strength_ = 0.25;

    // cv::Rect cached_roi_;
    // bool has_cached_roi_ = false;
    // ============================================================
    // 静态图片场景缓存：
    // 对于“单张图片 + 音频”的离线生成，人脸位置不变，
    // 所以 raw_rect / M / M_inv / aligned_face / face_tensor / mask
    // 只需要构建一次，后续每个音频 chunk 复用即可。
    //
    // 实时视频流场景不能直接复用，因为每帧人脸位置可能变化。
    // ============================================================
    struct StaticFaceCache {
        bool valid = false;

        // 判断当前输入帧是否和缓存兼容
        cv::Size frame_size;
        int frame_type = -1;

        // 原始静态图
        cv::Mat base_face;

        // dlib检测到的人脸框
        cv::Rect raw_rect;

        // 原图 -> 96x96的仿射矩阵
        cv::Mat M;

        // 96x96 -> 原图的逆仿射矩阵
        cv::Mat M_inv;

        // 人脸裁切图
        cv::Mat aligned_face;

        // Wav2lip 的 6 通道人脸输入， shape = 96 x 96 x 6
        // 这是一个静态图片，不会随着音频变化，可以缓存
        ncnn::Mat face_tensor;

        // 96 x 96 空间下的嘴部融合
        cv::Mat mouth_mask_96;

        // 96×96 对齐空间中的 68 点 landmarks
        std::vector<cv::Point2f> landmarks_96;

        // 单通道精细嘴部 alpha，CV_32FC1，范围 [0,1]
        cv::Mat mouth_alpha_96;
    };

    // 静态图片离线测试先默认开启
    bool static_face_cache_enabled_ = true;
    StaticFaceCache static_cache_;

    InferenceModelType current_model_type_ = InferenceModelType::Wav2Lip;
    std::thread worker_thread_;
    std::atomic<bool> is_running_{false};

    const int LATENCY_WARNING_MS = 100;

    Impl(::ThreadSafeQueue<Core::InferenceTask>& in_queue,
         Core::FrameScheduler& scheduler)
        : input_queue_(in_queue), frame_scheduler_(scheduler),
          metrics_(!std::getenv("DH_PERF_METRICS") ||
                   std::string(std::getenv("DH_PERF_METRICS")) != "0") {
        wav2lip_engine_ = std::make_unique<ModelInference>();
        wav2lip_output_processor_ = std::make_unique<OutputProcessor>();
        input_processor_ = std::make_unique<InputProcessor>();
    }

    ~Impl() {
        stop();
    }

    void clearStaticFaceCache() {
        static_cache_ = StaticFaceCache{};  // 直接使用空结构体替换
    }

    static std::vector<fs::path> buildLandmarkCandidates() {
        std::vector<fs::path> candidates;

        if (const char* env = std::getenv("DLIB_LANDMARK_MODEL")) {
            candidates.emplace_back(env);
        }

        if (const char* env2 = std::getenv("SHAPE_PREDICTOR_PATH")) {
            candidates.emplace_back(env2);
        }

        const fs::path file = "shape_predictor_68_face_landmarks.dat";

        candidates.emplace_back(file);
        candidates.emplace_back(fs::path("models") / file);
        candidates.emplace_back(fs::path("../models") / file);
        candidates.emplace_back(fs::path("../../models") / file);

        return candidates;
    }

    bool initWav2Lip(ncnn::Net* net, bool use_gpu) {
        if (!net) {
            return false;
        }

        wav2lip_engine_->bindModel(net);
        wav2lip_backend_ = std::make_unique<Wav2LipBackend>(net);

        dlib_ready_ = false;
        for (const auto& candidate : buildLandmarkCandidates()) {
            try {
                std::cout << "[InferenceProcessor] try landmark path: "
                        << candidate.string() << std::endl;

                if (fs::exists(candidate) &&
                    face_detector_.loadLandmarkModel(candidate.string())) {
                    dlib_ready_ = true;
                    std::cout << "[InferenceProcessor] Dlib 68-Landmarks 加载成功: "
                            << candidate.string() << std::endl;
                    break;
                }
            } catch (const std::exception& e) {
                std::cerr << "[InferenceProcessor] Landmark load exception: "
                        << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[InferenceProcessor] Landmark load unknown exception." << std::endl;
            }
        }

        if (!dlib_ready_) {
            std::cerr << "[InferenceProcessor] Warning: 无法加载 Dlib 模型，回退到人脸框裁剪。" << std::endl;
        }

        InferenceConfig config;
        config.use_vulkan = use_gpu || use_vulkan_;
        config.num_threads = inference_threads_;
        config.use_fp16 = use_fp16_;
        config.light_mode = light_mode_;
        wav2lip_engine_->setConfig(config);
        wav2lip_backend_->setConfig(config);

        return true;
    }

    bool initMuseTalk(const std::string& model_dir) {
        (void)model_dir;
        return false;
    }

    cv::Rect detectFaceRoi(const cv::Mat& frame) {
        if (frame.empty()) {
            return cv::Rect();
        }

        if (!dlib_ready_) {
            return (cv::Rect(frame.cols / 4, frame.rows / 4, frame.cols / 2, frame.rows / 2) &
                    cv::Rect(0, 0, frame.cols, frame.rows));
        }

        std::vector<cv::Rect> faces = face_detector_.detect(frame);
        if (faces.empty()) {
            std::cerr << "[InferenceProcessor] 未检测到人脸，使用全图作为 ROI。" << std::endl;
            return cv::Rect(0, 0, frame.cols, frame.rows);
        }

        const cv::Rect& face_rect = faces[0];
        std::vector<cv::Point> landmarks = face_detector_.getLandmarks(frame, face_rect);

        if (landmarks.size() != 68) {
            std::cerr << "[InferenceProcessor] 关键点检测失败，回退到人脸框。" << std::endl;
            return face_rect & cv::Rect(0, 0, frame.cols, frame.rows);
        }

        int x_min = frame.cols;
        int x_max = 0;
        int y_min = frame.rows;
        int y_max = 0;

        for (const auto& pt : landmarks) {
            x_min = std::min(x_min, pt.x);
            x_max = std::max(x_max, pt.x);
            y_min = std::min(y_min, pt.y);
            y_max = std::max(y_max, pt.y);
        }

        const int w = std::max(1, x_max - x_min);
        const int h = std::max(1, y_max - y_min);
        const int cx = x_min + w / 2;
        const int cy = y_min + h / 2;
        const int side = std::max(32, static_cast<int>(std::max(w, h) * 1.35));

        cv::Rect roi(cx - side / 2, cy - side / 2, side, side);
        return roi & cv::Rect(0, 0, frame.cols, frame.rows);
    }

    /**
     * @brief 构建缓存函数
     * @note 只做一次人脸检测，把后续的每帧不变的东西都缓存起来
     */
    bool buildFaceCacheFromFrame(const cv::Mat& frame, StaticFaceCache& cache) {
        const auto cache_build_start = std::chrono::steady_clock::now();
        if (frame.empty()) {
            std::cerr << "[InferenceProcessor] buildFaceCache failed: empty frame." << std::endl;
            return false;
        }

        std::vector<cv::Rect> faces = face_detector_.detect(frame);
        if (faces.empty()) {
            std::cerr << "[InferenceProcessor] buildFaceCache failed: no face detected." << std::endl;
            return false;
        }

        cv::Rect raw_rect = faces[0] & cv::Rect(0, 0, frame.cols, frame.rows);
        if (raw_rect.width <= 0 || raw_rect.height <= 0) {
            std::cerr << "[InferenceProcessor] buildFaceCache failed: invalid face rect." << std::endl;
            return false;
        }

        // 必须获取 68 点 landmarks，用于构造嘴部精细 alpha mask
        std::vector<cv::Point> landmarks = face_detector_.getLandmarks(frame, raw_rect);
        if (landmarks.size() != 68) {
            std::cerr << "[InferenceProcessor] buildFaceCache failed: invalid landmarks, size="
                    << landmarks.size() << std::endl;
            return false;
        }

        int padding = static_cast<int>(raw_rect.width * 0.15);
        int side = std::max(raw_rect.width, raw_rect.height) + 2 * padding;
        side = std::max(side, 32);

        double cx = raw_rect.x + raw_rect.width / 2.0;
        double cy = raw_rect.y + raw_rect.height / 2.0;

        double scale = 96.0 / static_cast<double>(side);

        cv::Mat M = cv::Mat::zeros(2, 3, CV_64F);
        M.at<double>(0, 0) = scale;
        M.at<double>(1, 1) = scale;
        M.at<double>(0, 2) = -(cx - side / 2.0) * scale;
        M.at<double>(1, 2) = -(cy - side / 2.0) * scale;

        cv::Mat M_inv;
        cv::invertAffineTransform(M, M_inv);

        cv::Mat aligned_face;
        cv::warpAffine(
            frame,
            aligned_face,
            M,
            cv::Size(96, 96),
            cv::INTER_CUBIC,
            cv::BORDER_CONSTANT,
            cv::Scalar(0, 0, 0)
        );

        ncnn::Mat face_tensor =
            input_processor_->processImage(aligned_face, cv::Rect(0, 48, 96, 48));

        if (face_tensor.empty()) {
            std::cerr << "[InferenceProcessor] buildFaceCache failed: empty face_tensor." << std::endl;
            return false;
        }

        // ============================================================
        // 基于 dlib 68 点构造精细嘴部 alpha mask
        // ============================================================
        std::vector<cv::Point2f> landmarks_96 = transformLandmarksTo96(landmarks, M);

        cv::Mat mouth_alpha_96 = buildMouthAlphaMask96(landmarks_96);

        cv::Mat mouth_mask_96;
        {
            std::vector<cv::Mat> mask_channels(3, mouth_alpha_96);
            cv::merge(mask_channels, mouth_mask_96);
        }

        cache.valid = true;
        cache.frame_size = frame.size();
        cache.frame_type = frame.type();
        cache.base_face = frame.clone();
        cache.raw_rect = raw_rect;
        cache.M = M.clone();
        cache.M_inv = M_inv.clone();
        cache.aligned_face = aligned_face.clone();
        cache.face_tensor = face_tensor;
        cache.landmarks_96 = landmarks_96;
        cache.mouth_alpha_96 = mouth_alpha_96.clone();
        cache.mouth_mask_96 = mouth_mask_96.clone();

        const size_t cache_bytes =
            cache.base_face.total() * cache.base_face.elemSize() +
            cache.M.total() * cache.M.elemSize() +
            cache.M_inv.total() * cache.M_inv.elemSize() +
            cache.aligned_face.total() * cache.aligned_face.elemSize() +
            cache.face_tensor.total() * cache.face_tensor.elemsize +
            cache.mouth_alpha_96.total() * cache.mouth_alpha_96.elemSize() +
            cache.mouth_mask_96.total() * cache.mouth_mask_96.elemSize() +
            cache.landmarks_96.capacity() * sizeof(cv::Point2f);
        const auto cache_build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cache_build_start).count();

        std::cout << "[InferenceProcessor] Static face cache built. raw_rect="
                << raw_rect
                << ", frame=" << frame.cols << "x" << frame.rows
                << ", mouth_alpha_96=" << mouth_alpha_96.cols << "x" << mouth_alpha_96.rows
                << ", build_ms=" << cache_build_ms
                << ", bytes=" << cache_bytes
                << std::endl;

        return true;
    }

    /**
     * @brief 获取缓存
     */
    bool getFaceCacheForTask(const cv::Mat& frame, StaticFaceCache& out_cache) {
        if (frame.empty()) {
            return false;
        }

        const auto cache_lookup_start = std::chrono::steady_clock::now();

        // 实时模式，每一帧都需要重新检测
        if (!static_face_cache_enabled_) {
            const bool ready = buildFaceCacheFromFrame(frame, out_cache);
            metrics_.add("face_cache_miss", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cache_lookup_start).count());
            return ready;
        }

        // 静态模式
        if (static_cache_.valid &&
            static_cache_.frame_size == frame.size() &&
            static_cache_.frame_type == frame.type()) {
            const auto cache_hit_start = std::chrono::steady_clock::now();
            out_cache = static_cache_;
            static int cache_hit_log_count = 0;
            if (cache_hit_log_count++ < 3) {
                const auto cache_hit_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - cache_hit_start).count();
                std::cout << "[InferenceProcessor] Static face cache hit: "
                          << cache_hit_us << " us" << std::endl;
            }
            metrics_.add("face_cache_hit", std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - cache_lookup_start).count());
            return true;
        }

        // 当第一次进入或者输入尺寸发生变化时，重新构建缓存
        clearStaticFaceCache();

        if (!buildFaceCacheFromFrame(frame, static_cache_)) {
            return false;
        }

        metrics_.add("face_cache_miss", std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cache_lookup_start).count());

        out_cache = static_cache_;
        return true;
    }

    void processLoop() {
        Core::FaceMaskGenerator face_mask_generator; // 实例化掩码生成器

        while (is_running_.load(std::memory_order_acquire)) {
            auto task_opt = input_queue_.pop(50);
            if (!task_opt.has_value()) {
                continue;
            }

            Core::InferenceTask task = std::move(task_opt.value());
            if (task.base_face.empty() || task.audio_feature.size() != 1280) {
                continue;
            }

            cv::Mat final_image = task.base_face.clone();

            if (current_model_type_ == InferenceModelType::Wav2Lip) {
                cv::Mat final_out = task.base_face.clone();

                // 初始化缓存
                StaticFaceCache face_cache;
                bool cache_ready = false;
                {
                    Core::PerformanceMetrics::Scope face_scope(metrics_, "face_detection_alignment");
                    cache_ready = getFaceCacheForTask(task.base_face, face_cache);
                }
                if (!cache_ready) {
                    std::cerr << "[InferenceProcessor] Failed to get face cache, use original frame."
                              << std::endl;

                    Video::VideoFrame out_frame;
                    out_frame.pts = task.pts_ms;
                    out_frame.image = final_out;
                    out_frame.background = cv::Mat();
                    out_frame.roi = cv::Rect();
                    frame_scheduler_.pushFrame(out_frame);
                    continue;
                }

                InferenceResult backend_result;
                FaceInput backend_face;
                backend_face.aligned_face = face_cache.aligned_face;
                backend_face.context.face_roi = face_cache.raw_rect;
                backend_face.context.mouth_roi = cv::Rect(0, 48, 96, 48);
                AudioFeature backend_audio;
                backend_audio.values = task.audio_feature;
                backend_audio.pts_ms = task.pts_ms;
                {
                    Core::PerformanceMetrics::Scope backend_scope(metrics_, "model_backend");
                    if (!wav2lip_backend_->process(backend_audio, backend_face, backend_result)) {
                        backend_result.success = false;
                    }
                }
                if (backend_result.success) {
                    Core::PerformanceMetrics::Scope output_scope(metrics_, "fusion_encoding");
                        cv::Mat generated_96 = backend_result.generated_face;

                    if (!generated_96.empty()) {
                        // 1. 对 96×96 输出轻微锐化，改善放大后的软糊感
                        cv::Mat generated_96_sharp = generated_96;
                        if (sharpening_enabled_) {
                            cv::Mat blur;
                            cv::GaussianBlur(generated_96, blur, cv::Size(0, 0), 1.0);
                            cv::addWeighted(generated_96, 1.0 + sharpening_strength_,
                                            blur, -sharpening_strength_, 0.0,
                                            generated_96_sharp);
                        }

                        // 2. 反变换 generated_96 到原图坐标
                        cv::Mat restored_face =
                            cv::Mat::zeros(task.base_face.size(), CV_8UC3);

                        cv::warpAffine(
                            generated_96_sharp,
                            restored_face,
                            face_cache.M_inv,
                            task.base_face.size(),
                            cv::INTER_CUBIC,
                            cv::BORDER_CONSTANT,
                            cv::Scalar(0, 0, 0)
                        );

                        // 3. 使用精细嘴部 alpha mask
                        cv::Mat mask_96_3c;
                        if (!face_cache.mouth_alpha_96.empty()) {
                            std::vector<cv::Mat> mask_channels(3, face_cache.mouth_alpha_96);
                            cv::merge(mask_channels, mask_96_3c);
                        } else {
                            mask_96_3c = face_cache.mouth_mask_96;
                        }

                        cv::Mat restored_mask =
                            cv::Mat::zeros(task.base_face.size(), CV_32FC3);

                        cv::warpAffine(
                            mask_96_3c,
                            restored_mask,
                            face_cache.M_inv,
                            task.base_face.size(),
                            cv::INTER_LINEAR,
                            cv::BORDER_CONSTANT,
                            cv::Scalar(0, 0, 0)
                        );

                        // 4. warp 后轻微羽化，消除锯齿和边界线
                        cv::GaussianBlur(restored_mask, restored_mask, cv::Size(7, 7), 0);

                        // 5. 使用带原图细节回填的融合
                        final_out = blendRestoredFaceWithDetail(
                            final_out,
                            restored_face,
                            restored_mask
                        );

                    #if 1
                        // 6. 调试图：用于确认 mask 是否过大、是否触碰下巴。
                        // 保存目录为当前运行目录下的 tmp，例如 /workspace/build/tmp。
                        static int mouth_debug_idx = 0;

                        if (mouth_debug_idx % 25 == 0) {
                            const std::string debug_dir = getMouthDebugDir();

                            const std::string idx_str = std::to_string(mouth_debug_idx);

                            cv::imwrite(debug_dir + "/mouth_debug_" + idx_str + "_00_base.jpg",
                                        task.base_face);

                            cv::imwrite(debug_dir + "/mouth_debug_" + idx_str + "_01_generated96.jpg",
                                        generated_96);

                            cv::imwrite(debug_dir + "/mouth_debug_" + idx_str + "_02_generated96_sharp.jpg",
                                        generated_96_sharp);

                            cv::imwrite(debug_dir + "/mouth_debug_" + idx_str + "_03_restored_face.jpg",
                                        restored_face);

                            cv::Mat mask_vis;
                            restored_mask.convertTo(mask_vis, CV_8UC3, 255.0);

                            cv::imwrite(debug_dir + "/mouth_debug_" + idx_str + "_04_restored_mask.jpg",
                                        mask_vis);

                            cv::imwrite(debug_dir + "/mouth_debug_" + idx_str + "_05_final_out.jpg",
                                        final_out);

                            std::cout << "[MouthBlendDebug] saved debug images to "
                                    << debug_dir
                                    << ", idx=" << mouth_debug_idx
                                    << ", generated_96=" << generated_96.cols << "x" << generated_96.rows
                                    << ", restored_mask=" << restored_mask.cols << "x" << restored_mask.rows
                                    << std::endl;
                        }

                        ++mouth_debug_idx;
                    #endif
                    }
                }
                Video::VideoFrame out_frame;
                out_frame.pts = task.pts_ms;
                out_frame.image = final_out;
                out_frame.background = cv::Mat();
                out_frame.roi = cv::Rect();

                frame_scheduler_.pushFrame(out_frame);
            }
        }
    }

    
    void setStaticFaceCacheEnabled(bool enabled) {
        static_face_cache_enabled_ = enabled;

        if (!enabled) {
            clearStaticFaceCache();
        }

        std::cout << "[InferenceProcessor] Static face cache: "
                << (enabled ? "ON" : "OFF") << std::endl;
    }

    bool start() {
        if (is_running_.load()) {
            return true;
        }
        is_running_.store(true);
        worker_thread_ = std::thread(&Impl::processLoop, this);
        return true;
    }

    void stop() {
        if (!is_running_.exchange(false)) {
            return;
        }
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }

        if (metrics_.enabled()) {
            std::cout << "[PerformanceMetrics] stages=" << metrics_.toJson() << std::endl;
        }
    }
}; 

InferenceProcessor::InferenceProcessor(::ThreadSafeQueue<Core::InferenceTask>& input_queue,
                                       Core::FrameScheduler& frame_scheduler)
    : pImpl(std::make_unique<Impl>(input_queue, frame_scheduler)) {}

InferenceProcessor::~InferenceProcessor() = default;

bool InferenceProcessor::initWav2Lip(ncnn::Net* net, bool use_gpu) {
    return pImpl->initWav2Lip(net, use_gpu);
}

bool InferenceProcessor::initMuseTalk(const std::string& model_dir) {
    return pImpl->initMuseTalk(model_dir);
}

void InferenceProcessor::setModelType(InferenceModelType type) {
    pImpl->current_model_type_ = type;
}

void InferenceProcessor::setStaticFaceCacheEnabled(bool enabled) {
    pImpl->setStaticFaceCacheEnabled(enabled);
}

void InferenceProcessor::setInferenceThreads(int threads) {
    pImpl->inference_threads_ = std::max(1, threads);
}

void InferenceProcessor::setInferenceOptions(bool use_vulkan, bool use_fp16, bool light_mode) {
    pImpl->use_vulkan_ = use_vulkan;
    pImpl->use_fp16_ = use_fp16;
    pImpl->light_mode_ = light_mode;
}

void InferenceProcessor::setSharpening(bool enabled, double strength) {
    pImpl->sharpening_enabled_ = enabled;
    pImpl->sharpening_strength_ = std::max(0.0, std::min(1.0, strength));
}

bool InferenceProcessor::start() {
    return pImpl->start();
}

void InferenceProcessor::stop() {
    pImpl->stop();
}

} // namespace Model
} // namespace DigitalHuman
