# 后续阶段实施状态

## 已完成

- Docker 容器内完整 CMake 构建。
- Wav2Lip 单帧 benchmark：模型加载、热身、平均/最小/最大延迟和 FPS 输出。
- OpenCV、Frame Scheduler、Output Processor 回归测试。
- `PipelineConfig`、`FrameContext`、`ModelBackend` 和推理延迟统计基础。
- React/Vite 深色实验室仪表盘原型，使用预制数据和模拟指标。

## 当前基线示例

在当前 4 vCPU CPU 容器中，3 次热身后的 Wav2Lip 单帧测试约为 31.9 ms，约 31.4 FPS。该数字只代表 dummy tensor 推理，不等于完整视频端到端吞吐。

运行：

```bash
./scripts/run_benchmark.sh ./models 50
```

## 下一步

1. 给音频、检测、对齐、融合和编码阶段接入同一套计时输出。
2. 用真实图片和音频完成 30 秒端到端基准。
3. 将动态视频检测与跟踪接入 `FrameContext`。
4. 在 MuseTalk 模型资源完整后接入统一 `ModelBackend`。
5. 确认展馆视觉方向后接 FastAPI 和真实结果文件。
