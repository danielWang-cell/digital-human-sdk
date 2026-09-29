#include <iostream>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>

// 打印前 N 个概率最大的分类
void print_topk(const std::vector<float>& cls_scores, int topk) {
    int size = cls_scores.size();
    std::vector<std::pair<float, int>> vec;
    vec.resize(size);
    for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(cls_scores[i], i);
    }

    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater<std::pair<float, int>>());

    for (int i = 0; i < topk; i++) {
        float score = vec[i].first;
        int index = vec[i].second;
        std::cout << "Top " << i + 1 << ": Class " << index << " = " << score << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "=== Digital Human SDK: ncnn Inference Test ===" << std::endl;

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " [imagepath]" << std::endl;
        return -1;
    }

    const char* imagepath = argv[1];

    // 1. 加载 OpenCV 图像
    cv::Mat img = cv::imread(imagepath, 1);
    if (img.empty()) {
        std::cerr << "Error: CV Load image failed: " << imagepath << std::endl;
        return -1;
    }
    std::cout << "Image loaded: " << img.cols << "x" << img.rows << std::endl;

    // 2. 加载 ncnn 模型
    // 这里的模型路径是相对运行时的路径，需要把 .param 和 .bin 放到 bin 目录下
    ncnn::Net squeezenet;
    squeezenet.opt.use_vulkan_compute = false; // 关闭 GPU，只测 CPU
    squeezenet.opt.num_threads = 4;

    // 尝试加载 (返回 0 表示成功)
    if (squeezenet.load_param("squeezenet_v1.1.param") != 0) {
        std::cerr << "Error: load_param failed! (Check file path)" << std::endl;
        return -1;
    }
    if (squeezenet.load_model("squeezenet_v1.1.bin") != 0) {
        std::cerr << "Error: load_model failed! (Check file path)" << std::endl;
        return -1;
    }
    std::cout << "Model loaded successfully." << std::endl;

    // 3. 预处理 (Pre-processing)
    // SqueezeNet 输入尺寸是 227x227
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, 227, 227);

    // 归一化参数 (SqueezeNet 的常见参数)
    const float mean_vals[3] = {104.f, 117.f, 123.f};
    in.substract_mean_normalize(mean_vals, 0);

    // 4. 推理 (Inference)
    ncnn::Extractor ex = squeezenet.create_extractor();

    // 输入节点名 "data" (可以从 .param 文件第一层看到)
    ex.input("data", in);

    // 获取输出节点 "prob"
    ncnn::Mat out;
    ex.extract("prob", out);

    // 5. 解析结果
    std::cout << "Inference done. Output shape: " << out.w << "x" << out.h << "x" << out.c << std::endl;

    // 将 ncnn::Mat 转换为 std::vector<float>
    std::vector<float> cls_scores;
    cls_scores.resize(out.w);
    for (int j = 0; j < out.w; j++) {
        cls_scores[j] = out[j];
    }

    print_topk(cls_scores, 3);

    return 0;
}
