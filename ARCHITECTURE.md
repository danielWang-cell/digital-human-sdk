# Digital Human SDK 系统架构

## API integration branch (2026-10-01)

The `codex/cloud-api-integration` branch migrates the existing development
copy's Wav2Lip backend, stage metrics, FastAPI sample task service and Compose
configuration. FaceTracker changes are not part of this branch.

`POST /api/tasks` accepts only `sample_id=default`. A single executor runs the
local C++ renderer; task status and output are persisted. Renderer processes
start in a separate process group so timeout terminates FFmpeg descendants.
Queued/running task records become failed after API restart.

The default proxy binds to loopback. This service has no authentication or
remote Worker claim/heartbeat/result upload protocol. See `docs/deployment.md`
for build/resource requirements and `docs/cloud_api_validation.md` for evidence.
Historical baseline descriptions below predate this integration branch.

## 业务目标

系统将静态照片或视频中的人脸与音频特征结合，生成嘴型同步的数字人视频。当前实现优先支持 CPU、单人、静态图片加音频和 Wav2Lip；动态视频、MuseTalk 和在线任务服务按阶段接入。

## 当前运行边界

已支持：Linux/Docker、x86-64 通用编译、ncnn CPU Wav2Lip、16 kHz 音频处理和 Mel 特征、dlib 68 点人脸关键点、静态人脸缓存、实时预览、离线输出和 FFmpeg 音视频合并。

当前限制：

- 动态视频跟踪尚未生产化。
- MuseTalk 目前只有 UNet 探索代码，尚未接通 VAE 和真实音频编码。
- 完整应用在当前容器中曾触发 dlib `SIGILL`，端到端基准需先完成兼容性排障。
- API、上传任务和公网部署尚未实现。

## 总体数据流

```text
图片/视频 + 音频
       │
       ├── 音频加载/重采样 → 分帧 → Mel 特征 → AudioFeature
       │
       └── 视频解码/图像输入 → 检测 → 关键点 → FrameContext

AudioFeature + FrameContext
       │
       ├── FaceInput / 对齐 / mask
       ├── ModelBackend
       │       ├── Wav2Lip（实时、CPU 优先）
       │       └── MuseTalk（离线、高质量）
       ├── InferenceResult
       ├── 嘴部融合与细节回填
       ├── PTS 调度和队列反压
       └── RenderProcessor → FFmpeg → 输出视频
```

## C++ 模块依赖

```text
Pipeline
 ├── ThreadSafeQueue<InferenceTask>
 ├── AudioPlayer / AudioProcessor
 ├── InferenceProcessor
 │    ├── InputProcessor
 │    ├── ModelInference / ncnn::Net
 │    ├── FaceDetector / dlib
 │    ├── FaceAligner
 │    └── OutputProcessor
 ├── FrameScheduler
 └── RenderProcessor / FFmpeg
```

### Audio

- `AudioLoader`：FFmpeg 解码和重采样。
- `AudioFramer`：音频分帧和窗口处理。
- `AudioMelFeatureExtract`：生成 Wav2Lip 所需 Mel 特征。
- `AudioPreprocessor`：音量、预加重和 VAD 等预处理。
- `AudioPlayer`：实时模式下提供音频时钟。

### Face 与 FrameContext

`FrameContext` 是 Wav2Lip、MuseTalk 和动态跟踪共享的帧级数据对象，包含原始帧、帧号、PTS、人脸框、关键点、对齐矩阵、逆矩阵、嘴部 ROI、mask 和跟踪状态。

静态图片模式可缓存检测结果、对齐结果、人脸输入张量和嘴部 mask；动态视频模式必须按帧更新会变化的字段。

### ModelBackend

上层模型契约由 `AudioFeature`、`FaceInput`、`InferenceResult` 和 `ModelBackend` 组成。Wav2Lip 和 MuseTalk 应在该接口下被任务级选择，不能在每帧之间频繁切换。

## Pipeline 生命周期

```text
构造 → 配置 PipelineConfig → 加载模型和关键点资源
  → 启动音频/推理/渲染线程 → pushTask / pushAudioData
  → 队列调度、推理、融合、渲染 → stop / 释放资源
```

`PipelineConfig` 当前控制 ncnn 线程数、Vulkan、FP16、light mode、锐化开关和锐化强度；后续可扩展队列大小、检测间隔和跟踪阈值。

## 队列和时序

- 输入任务进入推理队列，推理结果进入渲染队列。
- 离线模式保留完整帧序列。
- 实时预览模式可以清理旧渲染帧，只保留较新的结果。
- `pts_ms` 是音频和视频同步的主要时间基准。
- 队列积压必须进入性能指标，不能只依赖日志判断。

## 性能指标

基线至少记录音频特征、检测与对齐、单帧推理、融合、编码、首帧延迟、端到端墙钟时间、峰值 RSS、推理队列和渲染队列最大长度。单帧 benchmark 和端到端 benchmark 位于 `examples/` 与 `scripts/`；单帧结果不能替代真实视频端到端结果。

## 模型策略

```text
realtime / CPU / low_latency -> Wav2Lip
offline / quality_first      -> MuseTalk
MuseTalk 资源不完整          -> 回退 Wav2Lip
```

MuseTalk 完整接入必须具备音频编码器、VAE encoder、UNet、VAE decoder 和明确的输入输出尺寸契约。现有 UNet 探索代码不能作为生产后端。

## 服务层演进

后续服务层建议分为：

```text
FastAPI
 ├── /api/samples
 ├── /api/tasks
 ├── /api/tasks/{id}
 ├── /api/metrics
 └── /assets/results
```

第一版 API 使用预制样本和已有输出；上传、任务队列、文件限制、清理策略和公网安全在核心链路稳定后加入。

## 版本与部署

- `main`：当前融合开发基线。
- `master`：同步前的历史版本，仅供对照。
- Docker 镜像：`weijhang/digital-human-env:ubuntu22.04-cpu-v1.0`。
- 运行时模型放在本地 `models/`，不提交 Git。
- 构建目录、日志、视频、音频、前端依赖不提交 Git。

## 变更原则

涉及公共数据结构、线程生命周期、PTS、模型输入输出或队列行为的改动，必须同时更新本文件和 `TODO.md`，并在 Docker 中完成至少一次构建和相关测试。
