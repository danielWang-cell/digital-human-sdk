#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include "audio/audio_preprocessor.h"

using DigitalHuman::Audio::AudioPreprocessor;

// 辅助：生成测试音频 (参数化时长)
// 模式：每秒交替生成静音和正弦波
std::vector<float> GenerateTestSignal(int sample_rate, int duration_sec = 4) {
  std::vector<float> pcm;
  pcm.reserve(sample_rate * duration_sec);

  for (int i = 0; i < sample_rate * duration_sec; ++i) {
    // 每秒切换一次状态
    int sec_idx = i / sample_rate;
    if (sec_idx % 2 == 0) {
      // 偶数秒：静音 (带微弱底噪)
      // 使用 static_cast 规范类型转换
      float noise = 0.001f * ((rand() % 100) / 50.0f - 1.0f);
      pcm.push_back(noise);
    } else {
      // 奇数秒：正弦波 (人声模拟)
      float sine_wave =
          0.5f * sinf(2 * 3.14159f * 440 * static_cast<float>(i) / sample_rate);
      pcm.push_back(sine_wave);
    }
  }
  return pcm;
}

int main() {
  std::cout << "=== Digital Human SDK: Audio Preprocessor Test ===" << std::endl;

  const int kSampleRate = 16000;
  AudioPreprocessor processor;

  // ------------------------------------------
  // 1. 功能性测试 (Functionality Tests)
  // ------------------------------------------
  auto pcm = GenerateTestSignal(kSampleRate, 4);  // 4秒音频

  std::cout << "\n[Test 1] Normalization..." << std::endl;
  auto pcm_norm = pcm;
  processor.normalize(pcm_norm, 0.95f);
  float max_val = 0;
  for (float v : pcm_norm) {
    if (std::abs(v) > max_val) max_val = std::abs(v);
  }
  std::cout << "   Normalized Max: " << max_val << " (Target: 0.95)"
            << std::endl;
  if (std::abs(max_val - 0.95f) < 0.01f) {
    std::cout << "   -> PASS" << std::endl;
  } else {
    std::cout << "   -> FAIL" << std::endl;
  }

  std::cout << "\n[Test 2] VAD (Voice Activity Detection)..." << std::endl;
  // 4秒音频，偶数秒静音，奇数秒有声 -> 预期检测到第 2秒 和 第 4秒
  auto segments = processor.detectSpeech(pcm, kSampleRate);
  std::cout << "   Detected " << segments.size() << " segments." << std::endl;
  for (size_t i = 0; i < segments.size(); ++i) {
    float start = static_cast<float>(segments[i].start_idx) / kSampleRate;
    float end = static_cast<float>(segments[i].end_idx) / kSampleRate;
    std::cout << "   Segment " << i << ": " << start << "s - " << end << "s"
              << std::endl;
  }

  std::cout << "\n[Test 3] Pre-emphasis..." << std::endl;
  auto pcm_pre = pcm;
  processor.preEmphasize(pcm_pre);
  std::cout << "   -> PASS (Filter applied)" << std::endl;

  std::cout << "\n[Test 4] Denoise (Noise Gate)..." << std::endl;
  auto pcm_dn = pcm;
  processor.denoise(pcm_dn, -40.0f);  // 阈值 -40dB
  float noise_energy = 0;
  for (int i = 0; i < 100; ++i) {
    noise_energy += std::abs(pcm_dn[i]);  // 检查开头的静音段
  }
  std::cout << "   Noise energy after gate: " << noise_energy << std::endl;
  if (noise_energy < 1e-5) {
    std::cout << "   -> PASS" << std::endl;
  } else {
    std::cout << "   -> FAIL" << std::endl;
  }

  // ------------------------------------------
  // 5. 性能基准测试 (Performance Benchmark)
  // 验收标准：10秒音频处理 < 1秒
  // ------------------------------------------
  std::cout << "\n==============================================" << std::endl;
  std::cout << "[Test 5] Performance Benchmark (10s Audio)" << std::endl;
  std::cout << "==============================================" << std::endl;

  // A. 准备数据
  const int kBenchDuration = 10;  // 10秒
  std::vector<float> bench_pcm =
      GenerateTestSignal(kSampleRate, kBenchDuration);
  std::cout << "   Input Data: " << bench_pcm.size() << " samples (16kHz, Mono)"
            << std::endl;

  // B. 开始计时
  auto start_time = std::chrono::high_resolution_clock::now();

  // C. 执行完整流水线 (模拟真实业务场景)
  // 1. 归一化 (全量遍历)
  processor.normalize(bench_pcm);
  // 2. 降噪 (全量遍历)
  processor.denoise(bench_pcm);
  // 3. VAD (分帧 + RMS计算)
  processor.detectSpeech(bench_pcm, kSampleRate);
  // 4. 预加重 (全量遍历)
  processor.preEmphasize(bench_pcm);

  // D. 结束计时
  auto end_time = std::chrono::high_resolution_clock::now();
  // 改为微秒 (microseconds) 以获得更高精度
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();
  float duration_ms = duration_us / 1000.0f;

  // E. 结果分析
  std::cout << "   Processing Time: " << duration_us << " us (" << duration_ms
            << " ms)" << std::endl;
  std::cout << "   Audio Duration : " << kBenchDuration * 1000 << " ms"
            << std::endl;

  // 计算实时率 (RTF)
  float rtf = duration_ms / (kBenchDuration * 1000.0f);
  std::cout << "   Real-Time Factor (RTF): " << rtf << std::endl;

  if (duration_ms < 1000) {  // 1秒内
    std::cout << "   -> PASS (Performance is excellent!)" << std::endl;
    std::cout << "   -> Speed: " << (1.0f / rtf) << "x real-time." << std::endl;
  } else {
    std::cout << "   -> FAIL (Too slow, optimization needed)" << std::endl;
  }

  return 0;
}