#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace DigitalHuman {
namespace Audio {

class AudioLoader {
public:
    // 构造函数，支持自定义目标采样率（默认16k）
    explicit AudioLoader(int target_sample_rate = 16000);
    ~AudioLoader();

    // 禁止拷贝
    AudioLoader(const AudioLoader&) = delete;
    AudioLoader& operator=(const AudioLoader&) = delete;

    /**
     * @brief 加载并重采样音频
     * @param filename 路径 (支持 wav/mp3/aac等)
     * @param out_pcm 输出缓冲区
     * @return 是否处理成功
     */
    bool load(const std::string& filename, std::vector<int16_t>& out_pcm);

    /**
     * @brief 获取当前设置的目标采样率
     */
    int getTargetSampleRate() const;

private:
    struct Impl;            
    std::unique_ptr<Impl> pimpl; 
};

} // namespace Core
} // namespace DigitalHuman