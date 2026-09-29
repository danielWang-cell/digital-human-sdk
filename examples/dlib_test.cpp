#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

// 辅助函数：绘制关键点
void draw_landmarks(cv::Mat& img, const dlib::full_object_detection& shape) {
    // 绘制 68 个关键点
    for (unsigned long i = 0; i < shape.num_parts(); ++i) {
        dlib::point p = shape.part(i);
        // 画一个小绿点
        cv::circle(img, cv::Point(p.x(), p.y()), 2, cv::Scalar(0, 255, 0), -1);
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: dlib Landmark Test ===" << std::endl;

    // 检查参数
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <image_path> <model_path>" << std::endl;
        return -1;
    }

    const char* image_path = argv[1];
    const char* model_path = argv[2];

    // 1. 初始化 dlib
    // frontal_face_detector 是基于 HOG+SVM 的检测器
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;

    // 2. 加载模型
    std::cout << "[Info] Loading model from: " << model_path << " ..." << std::endl;
    try {
        dlib::deserialize(model_path) >> predictor;
    } catch (const dlib::serialization_error& e) {
        std::cerr << "[Error] Loading model failed: " << e.what() << std::endl;
        std::cerr << "Did you download 'shape_predictor_68_face_landmarks.dat'?" << std::endl;
        return -1;
    }
    std::cout << "[Info] Model loaded successfully." << std::endl;

    // 3. 读取图像 (OpenCV)
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[Error] Could not load image: " << image_path << std::endl;
        return -1;
    }
    std::cout << "[Info] Image loaded: " << img.cols << "x" << img.rows << std::endl;

    // 4. 将 OpenCV 图像包装成 dlib 格式 (Zero-Copy)
    // dlib::cv_image 只是一个 wrapper，不拷贝数据，速度极快
    dlib::cv_image<dlib::bgr_pixel> dlib_img(img);

    // 5. 检测人脸
    std::cout << "[Info] Detecting faces..." << std::endl;
    std::vector<dlib::rectangle> dets = detector(dlib_img);
    std::cout << "[Info] Number of faces detected: " << dets.size() << std::endl;

    if (dets.empty()) {
        std::cout << "[Warning] No faces found. Try an image with a clearer frontal face." << std::endl;
        return 0;
    }

    // 6. 遍历每张脸，检测关键点并绘制
    for (size_t i = 0; i < dets.size(); ++i) {
        // 预测关键点 (核心步骤)
        dlib::full_object_detection shape = predictor(dlib_img, dets[i]);
        
        std::cout << "   -> Face " << i << ": " << shape.num_parts() << " landmarks detected." << std::endl;

        // 绘制人脸框 (蓝色)
        cv::Rect rect(dets[i].left(), dets[i].top(), dets[i].width(), dets[i].height());
        cv::rectangle(img, rect, cv::Scalar(255, 0, 0), 2);

        // 绘制关键点 (绿色)
        draw_landmarks(img, shape);
        
        // 打印鼻尖坐标 (索引 30) 用于验证
        if (shape.num_parts() > 30) {
            std::cout << "      Nose tip position: (" << shape.part(30).x() << ", " << shape.part(30).y() << ")" << std::endl;
        }
    }

    // 7. 保存结果
    std::string output_filename = "dlib_result.jpg";
    cv::imwrite(output_filename, img);
    std::cout << "[Success] Result saved to: " << output_filename << std::endl;

    return 0;
}