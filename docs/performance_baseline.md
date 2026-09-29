# 性能基线

性能比较必须固定输入图片、音频、模型、分辨率和线程数。每次只打开一个开关，避免把模型、队列和图像处理的变化混在一起。

建议记录以下阶段：音频特征、单帧推理、人脸检测与对齐、融合、编码，以及 30 秒端到端耗时。另行记录首帧延迟、峰值 RSS、推理队列长度和渲染队列长度。

`PipelineConfig` 提供了第一阶段可切换项：`inference_threads`、`use_vulkan`、`use_fp16`、`enable_sharpen` 和 `sharpen_strength`。默认关闭锐化，便于得到没有额外图像处理开销的基线。

`ModelInference::getLatencyStats()` 可用于导出单帧推理的样本数、平均值、最小值和最大值。`PerformanceMetrics` 可用于在音频、检测、融合和编码边界增加同一口径的阶段计时。

推荐比较四组配置：

1. baseline；
2. baseline + logging；
3. baseline + zero-copy；
4. baseline + zero-copy + optimized pipeline。

只有在相同输入和统计口径下重复测量，才将结果用于模型或流水线默认值决策。

## 端到端基准命令

`scripts/run_e2e_benchmark.sh` 包装现有的 `digital_human_app` 离线路径，避免基准程序和实际输出路径出现差异。它会采集墙钟总耗时、首帧延迟、全链路和 FFmpeg 合并耗时、应用及进程树峰值 RSS，以及推理/渲染队列最大积压，并在构建目录下写出日志、RSS 采样和 JSON 结果。

在 SDK Docker 容器中运行 30 秒样本：

```bash
docker run --rm \
  -v "$PWD:/workspace" \
  -w /workspace \
  weijhang/digital-human-env:ubuntu22.04-cpu-v1.0 \
  bash -lc 'BUILD_DIR=/workspace/.docker-build \
    /workspace/scripts/run_e2e_benchmark.sh \
    /workspace/face.jpg \
    /workspace/test_audio_30s_sync.wav \
    /workspace/models'
```

可通过 `TIMEOUT_S` 调整超时，通过 `SAMPLE_MS` 调整 RSS 采样间隔。脚本退出码非零时仍会保留失败日志和 JSON，便于诊断模型加载、非法指令或编码失败。
