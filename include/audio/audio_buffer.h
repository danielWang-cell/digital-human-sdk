#pragma once

#include <vector>
#include <memory>
#include <functional>

namespace DigitalHuman {
namespace Audio {

/**
 * @brief 缓冲区溢出策略 
 */

enum class BufferOverflowStrategy {
    Block,      // 阻塞等待，直到有空间
    Overwrite,  // 覆盖旧数据
    Drop        // 丢弃新数据
};


/**
 * @brief 线程安全的音频环形缓冲区 
 */
class AudioBuffer {

public:

    // 告警回调函数类型：当前占用率，丢弃/覆盖的样本数
    using WarningCallback = std::function<void(float occupancy, size_t lost_samples)>;
    
    /**
     * @brief 构造函数
     * @param capacity_samples 缓冲区容量（采样点数） 
     */
    explicit AudioBuffer(size_t capacity_samples);
    ~AudioBuffer();

    // 移动语义与拷贝构造
    AudioBuffer(AudioBuffer&&) noexcept;
    AudioBuffer& operator=(AudioBuffer&&) noexcept;
    AudioBuffer(const AudioBuffer&) = delete;
    AudioBuffer& operator=(const AudioBuffer&) = delete;

    /**
     * @brief 生产者写入音频数据
     * @param data 输入数据
     * @param strategy 溢出处理策略，默认阻塞
     * @return 实际写入的样本数 
     */
    size_t push(const std::vector<float>& data, BufferOverflowStrategy strategy = BufferOverflowStrategy::Block);

    /**
     * @brief 消费者读取音频数据
     * @param out_data 输出容器
     * @param min_samples 期望读取的样本数
     * @param timeout_ms 超时时间，0 表示一直等待，-1 表示非阻塞立即返回 
     * @return 是否成功读取到了 min_samples 个数据
     */
    bool pull(std::vector<float>& out_data, size_t min_samples, int timeout_ms = 0);

    // 回调不再创建 std::vector，改为 pullRaw() 直接写入 PortAudio 的输出缓冲区
    bool pullRaw(float* out_data, size_t min_samples, int timeout_ms = 0);
    
    /**
     * @brief 设置溢出/异常回调 
     */
    void setWarningCallback(WarningCallback cb);

    /**
     * @brief 获取当前状态 
     */
    size_t size() const;        // 当前现有数据量
    size_t capacity() const;    // 总容量
    float occupancy() const;    // 占用率
    void clear();               // 清空


private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
};
} // namespace Audio
} // namespace DigitalHuman