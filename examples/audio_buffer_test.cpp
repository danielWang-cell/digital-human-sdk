#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>
#include <cassert>

#include "audio/audio_buffer.h"

using namespace DigitalHuman::Audio;

int main() {
    std::cout << "=== Digital Human SDK: Thread-Safe Audio Buffer Test ===" << std::endl;

    // 1. 初始化缓冲区 (容量 2000 个采样点)
    // 假设是 16kHz，这大约是 125ms 的数据
    AudioBuffer buffer(2000);
    
    // 设置告警回调，监控溢出情况
    buffer.setWarningCallback([](float occupancy, size_t lost) {
        // 只有当丢失数据时才打印，避免刷屏
        if (lost > 0) {
            std::cerr << "[Alert] Buffer Overflow! Occupancy: " << occupancy * 100 
                      << "%, Lost/Overwritten: " << lost << std::endl;
        }
    });

    // ----------------------------------------------------
    // 测试 1: 多线程并发读写 (Blocking 模式)
    // 验证数据完整性：写入的总量必须等于读取的总量
    // ----------------------------------------------------
    std::cout << "\n[Test 1] Multi-thread Producer-Consumer (Blocking)..." << std::endl;
    
    std::atomic<bool> running{true};
    std::atomic<size_t> total_written{0};
    std::atomic<size_t> total_read{0};
    
    // 消费者线程 (Consumer)
    std::thread consumer([&]() {
        std::vector<float> data_chunk;
        while (running || buffer.size() > 0) { // 只要还在运行或缓冲区有剩
            // 尝试读取 100 个数据，超时 10ms
            if (buffer.pull(data_chunk, 100, 10)) {
                total_read += data_chunk.size();
                // 模拟稍微慢一点的处理 (Consumer 慢于 Producer)
                // std::this_thread::sleep_for(std::chrono::microseconds(100)); 
            } else {
                if (!running && buffer.size() < 100) break; // 结束条件
            }
        }
    });

    // 生产者线程 (Producer)
    std::thread producer([&]() {
        std::vector<float> data_chunk(100, 1.0f); // 每次写100个
        for (int i = 0; i < 100; ++i) { // 总共写 100 * 100 = 10000 个数据
            // 使用阻塞策略，保证数据不丢
            buffer.push(data_chunk, BufferOverflowStrategy::Block);
            total_written += data_chunk.size();
            // 极速写入
        }
        running = false; // 通知消费者生产结束
    });

    producer.join();
    consumer.join();

    std::cout << "   Total Written: " << total_written << std::endl;
    std::cout << "   Total Read:    " << total_read << std::endl;
    std::cout << "   Buffer Remaining: " << buffer.size() << std::endl;
    
    if (total_written == total_read + buffer.size()) {
        std::cout << "   -> PASS (Data Integrity Perfect)" << std::endl;
    } else {
        std::cout << "   -> FAIL (Data Mismatch - Race Condition Detected!)" << std::endl;
    }

    // ----------------------------------------------------
    // 测试 2: 溢出策略验证 (Overwrite - 环形特性)
    // ----------------------------------------------------
    std::cout << "\n[Test 2] Overflow Strategy (Overwrite)..." << std::endl;
    buffer.clear();
    
    // 1. 填满缓冲区 (2000个)
    std::vector<float> full_data(2000, 1.0f); // 全部是 1.0
    buffer.push(full_data); 
    std::cout << "   Buffer filled. Size: " << buffer.size() << std::endl;

    // 2. 再强行写入 500 个 2.0 (策略: 覆盖)
    // 预期：最旧的 500 个 1.0 被挤出去，缓冲区末尾变成 2.0
    // 缓冲区逻辑状态应该变成：[500个 2.0 (新)] ... [1500个 1.0 (旧)] ???
    // 不，环形缓冲区的逻辑是 FIFO (先进先出)。
    // 覆盖旧数据意味着：Read 指针被迫向前跳过 500 个。
    // 剩下的数据应该是：[1500个 1.0] (头部) + [500个 2.0] (尾部)
    
    std::vector<float> new_data(500, 2.0f); 
    buffer.push(new_data, BufferOverflowStrategy::Overwrite);

    std::cout << "   Pushed 500 more (Overwrite). Size: " << buffer.size() << " (Should be 2000)" << std::endl;
    
    // 3. 验证数据内容
    std::vector<float> read_back;
    buffer.pull(read_back, 2000, 0); // 读出所有
    
    bool head_ok = (read_back[0] == 1.0f);   // 头部应该是旧数据
    bool tail_ok = (read_back[1999] == 2.0f); // 尾部应该是新数据
    
    // 检查分界线 (第 1500 个应该是 1.0，第 1501 应该是 2.0 ?)
    // 由于覆盖了最老的数据，现在 Read 指针指向的是原来的第 500 个数据
    // 所以读取出来的序列前 1500 个是 1.0，后 500 个是 2.0
    
    int count_1 = 0;
    int count_2 = 0;
    for(float v : read_back) {
        if(v == 1.0f) count_1++;
        if(v == 2.0f) count_2++;
    }
    
    std::cout << "   Count of 1.0 (Old): " << count_1 << " (Expected 1500)" << std::endl;
    std::cout << "   Count of 2.0 (New): " << count_2 << " (Expected 500)" << std::endl;

    if (count_1 == 1500 && count_2 == 500) {
         std::cout << "   -> PASS (Ring Buffer Overwrite Logic Correct)" << std::endl;
    } else {
         std::cout << "   -> FAIL (Data Logic Error)" << std::endl;
    }

    return 0;
}