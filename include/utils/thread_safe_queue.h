#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 50) : max_size_(max_size) {}

    /**
     * @brief 生产者线程调用这个函数将数据放到队列之中
     * @param item 数据
     * @return bool值放入成功
     */
    bool push(T item) {
        // 加锁
        std::unique_lock<std::mutex> lock(mutex_);  

        // 如果队列满了，且is_shutdown_ = false，则释放锁并阻塞
        // 当队列不满或者is_shutdown_ = true 则唤醒
        not_full_cv_.wait(lock, [this](){
            return queue_.size() < max_size_ || is_shutdown_;
        });

        if (is_shutdown_) {
            return false;
        }

        // 放入数据，使用 std::move 避免大型图片数据的拷贝
        queue_.push(std::move(item));

        // 通知消费者线程，队列中有任务可以消费了
        not_empty_cv_.notify_one();
        return true;
    }

    /**
     * @brief 消费者线程调用此函数从队列中获取数据（带超时机制）
     * @param timeout_ms 最大等待时长（毫秒），默认为 100ms
     * @return std::optional<T> 返回获取到的数据；若超时或队列已关闭则返回 std::nullopt
     */
    std::optional<T> pop(int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);

        bool success = false;

        if (timeout_ms < 0) {
            success = !queue_.empty();
        } else if (timeout_ms == 0) {
            not_empty_cv_.wait(lock, [this]() {
                return !queue_.empty() || is_shutdown_;
            });
            success = !queue_.empty();
        } else {
            success = not_empty_cv_.wait_for(
                lock,
                std::chrono::milliseconds(timeout_ms),
                [this]() {
                    return !queue_.empty() || is_shutdown_;
                }
            );
        }

        if (!success || (is_shutdown_ && queue_.empty())) {
            return std::nullopt;
        }

        T item = std::move(queue_.front());
        queue_.pop();

        not_full_cv_.notify_one();
        return item;
    }

    /**
     * @brief 关闭队列
     */
    void shutdown() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutdown_ = true;

        // 唤醒所有阻塞的线程
        not_empty_cv_.notify_all();
        not_full_cv_.notify_all();
    }

    // 获取当前队列中积压的任务数量
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        is_shutdown_ = false;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty_queue;
        queue_.swap(empty_queue);
        not_full_cv_.notify_all();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    bool isShutdown() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_shutdown_;
    }

private:
    std::queue<T> queue_;                   // 底层真实存放数据的队列
    mutable std::mutex mutex_;                      // 互斥锁
    std::condition_variable not_empty_cv_;  // 队列非空信号，传递给消费者
    std::condition_variable not_full_cv_;   // 队列非满信号，传递给生产者

    size_t max_size_;                       // 最大容量，防止内存爆炸
    bool is_shutdown_ = false;              // 关闭标志

};

