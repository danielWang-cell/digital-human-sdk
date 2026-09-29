#pragma once

#include <string>
#include <memory>
#include <functional>
#include <ncnn/net.h>

namespace DigitalHuman {
namespace Model {

/**
 * @brief 模型加载回调函数
 * @param net 加载成功的 ncnn::Net指针，如果失败则是 nullptr
 * @param cost_ms 加载耗时
 * */   
using LoadCallback = std::function<void(ncnn::Net* net, float cost_ms)>;

/**
 * @brief 模型生命周期管理器，负责 Wav2Lip 的 ncnn 模型的加载、预热、异步加载和释放
 */
class ModelLoader {
public:
    ModelLoader();
    ~ModelLoader();

    ModelLoader(const ModelLoader&) = delete;
    ModelLoader& operator=(const ModelLoader&) = delete;
    ModelLoader(ModelLoader&&) noexcept;
    ModelLoader& operator=(ModelLoader&&) noexcept;

    /**
     * @brief 同步加载模型，阻塞当前线程
     * @param model_path .param文件路径，自动推导.bin路径
     * @param use_gpu 是否开启 Vulkan
     * @return bool 是否成功
     */
    bool load(const std::string& model_path, bool use_gpu = false);

    /**
     * @brief 异步加载模型 ，非阻塞，适合 UI 线程调用
     * @param model_path .param 文件路径
     * @param callback 加载完成后的回调
     */
    void loadAsync(const std::string& model_path, LoadCallback callback);

    /**
     * @brief 获取加载好的 ncnn 网络实例
     */
    ncnn::Net* getNet();

    /**
     * @brief 检查模型是否已加载
     */
    bool isLoaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};

} // namespace Model
} // namespace DigitalHuman