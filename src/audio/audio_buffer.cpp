#include <mutex>
#include <condition_variable>
#include <iostream>
#include <algorithm>
#include <cstring> // for memcpy

#include "audio/audio_buffer.h"

namespace DigitalHuman {
namespace Audio {

struct AudioBuffer::Impl {
    std::vector<float> buffer; // 物理存储：一大块连续内存
    size_t capacity;           // 总容量
    
    // 逻辑状态：共享资源
    size_t read_pos = 0;       // 读指针
    size_t write_pos = 0;      // 写指针
    size_t current_size = 0;   // 当前有效数据量 

    // 线程同步工具
    mutable std::mutex mtx;               // 互斥锁：保护上述所有状态
    std::condition_variable not_empty_cv; // 信号量：通知有数据了
    std::condition_variable not_full_cv;  // 信号量：通知有空位了

    WarningCallback warning_cb; // 告警回调

    Impl(size_t cap) : capacity(cap) {
        buffer.resize(capacity, 0.0f);
    }

    // 写入逻辑 (必须在锁内调用)
    // 处理环形回绕的核心逻辑
    void write_internal(const float* data, size_t len) {
        // 分两步写入：
        // 1. 从 write_pos 到 数组末尾
        // 2. 如果没写完，从 数组开头 继续写
        size_t first_chunk = std::min(len, capacity - write_pos);
        std::copy(data, data + first_chunk, buffer.begin() + write_pos);
        
        size_t second_chunk = len - first_chunk;
        if (second_chunk > 0) {
            std::copy(data + first_chunk, data + len, buffer.begin());
        }

        // 更新写指针 (环形移动)
        write_pos = (write_pos + len) % capacity;
        current_size += len;
    }

    // 读取逻辑 (必须在锁内调用) 
    void read_internal(std::vector<float>& out, size_t len) {
        out.resize(len);
        
        // 分两步读取
        size_t first_chunk = std::min(len, capacity - read_pos);
        std::copy(buffer.begin() + read_pos, buffer.begin() + read_pos + first_chunk, out.begin());

        size_t second_chunk = len - first_chunk;
        if (second_chunk > 0) {
            std::copy(buffer.begin(), buffer.begin() + second_chunk, out.begin() + first_chunk);
        }

        // 更新读指针 (环形移动)
        read_pos = (read_pos + len) % capacity;
        current_size -= len;
    }

    size_t push(const std::vector<float>& data, BufferOverflowStrategy strategy) {
        // 保护共享资源：current_size, write_pos, read_pos
        std::unique_lock<std::mutex> lock(mtx);
        
        size_t input_len = data.size();
        
        // 当输入数据比整个缓冲区还大
        if (input_len > capacity) {
            if (warning_cb) {
                warning_cb(1.0f, input_len - capacity);
            }
            // 只取最后一段能装下的
            input_len = capacity; 
        }

        // 检查剩余空间
        size_t free_space = capacity - current_size;

        if (input_len > free_space) {
            // 发生溢出 (Overflow)
            if (strategy == BufferOverflowStrategy::Drop) {
                // 丢弃新数据
                if (warning_cb) {
                    warning_cb((float)current_size/capacity, input_len);
                }
                return 0; 
            }
            else if (strategy == BufferOverflowStrategy::Overwrite) {
                // 覆盖旧数据
                size_t needed = input_len - free_space;
                
                // 强制移动读指针 (相当于丢弃了最老的数据)
                read_pos = (read_pos + needed) % capacity;
                current_size -= needed; // 腾出空间
                
                if (warning_cb) {
                    warning_cb(1.0f, needed);
                }
            }
            else if (strategy == BufferOverflowStrategy::Block) {
                // 阻塞等待
                // wait 会释放锁，并休眠，直到被 notify 唤醒且条件满足
                not_full_cv.wait(lock, [&]{ 
                    return (capacity - current_size) >= input_len; 
                });
                // 醒来后，自动重新持有锁，继续执行
            }
        }

        // 写入 
        // 如果输入数据过大被截断，取后半部分
        size_t start_offset = data.size() - input_len;
        write_internal(data.data() + start_offset, input_len);

        // 唤醒正在等待数据的消费者
        not_empty_cv.notify_one();
        
        return input_len;
    }

    bool pull(std::vector<float>& out_data, size_t min_samples, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mtx);

        // 等待条件：当前有效数据量 >= 期望读取量
        bool success = false;
        
        if (timeout_ms < 0) {
            // 非阻塞模式：立即检查
            success = (current_size >= min_samples);
        } else if (timeout_ms == 0) {
            // 永久阻塞模式
            not_empty_cv.wait(lock, [&]{ return current_size >= min_samples; });
            success = true;
        } else {
            // 超时阻塞模式：等一段时间
            success = not_empty_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), 
                                          [&]{ return current_size >= min_samples; });
        }

        if (success) {
            // 执行读取
            read_internal(out_data, min_samples);
            
            // 唤醒正在等待空间的生产者 (如果是 Block 策略)
            not_full_cv.notify_one();
            return true;
        } else {
            return false; // 超时或数据不足
        }
    }
};

// Pimpl 外部接口实现
AudioBuffer::AudioBuffer(size_t capacity_samples) : pImpl(std::make_unique<Impl>(capacity_samples)) {}
AudioBuffer::~AudioBuffer() = default;
AudioBuffer::AudioBuffer(AudioBuffer&&) noexcept = default;
AudioBuffer& AudioBuffer::operator=(AudioBuffer&&) noexcept = default;

size_t AudioBuffer::push(const std::vector<float>& data, BufferOverflowStrategy strategy) {
    return pImpl->push(data, strategy);
}

bool AudioBuffer::pull(std::vector<float>& out_data, size_t min_samples, int timeout_ms) {
    return pImpl->pull(out_data, min_samples, timeout_ms);
}

void AudioBuffer::setWarningCallback(WarningCallback cb) {
    std::lock_guard<std::mutex> lock(pImpl->mtx);
    pImpl->warning_cb = cb;
}

size_t AudioBuffer::size() const {
    std::lock_guard<std::mutex> lock(pImpl->mtx);
    return pImpl->current_size;
}

size_t AudioBuffer::capacity() const {
    return pImpl->capacity;
}

float AudioBuffer::occupancy() const {
    std::lock_guard<std::mutex> lock(pImpl->mtx);
    if (pImpl->capacity == 0) return 0.0f;
    return (float)pImpl->current_size / pImpl->capacity;
}

void AudioBuffer::clear() {
    std::lock_guard<std::mutex> lock(pImpl->mtx);
    pImpl->read_pos = 0;
    pImpl->write_pos = 0;
    pImpl->current_size = 0;
}

} // namespace Audio
} // namespace DigitalHuman