# 任务看板

更新时间：2026-09-29  
当前基线：`main` / `v0.1.0-server-baseline`  
旧版本：`master` / `backup/before-current-sync`

## 状态说明

- `DONE`：已实现并完成验证
- `IN_PROGRESS`：当前有人正在修改
- `READY`：依赖已满足，可以领取
- `BLOCKED`：有明确外部阻塞
- `PARKED`：暂缓，不得擅自启动

## 全局锁

当前没有活跃代码锁。领取任务前必须把任务改为 `IN_PROGRESS`，填写 Agent、分支和文件范围。

锁格式：

```text
T-XXX | owner=<name> | branch=<branch> | status=IN_PROGRESS
files: <具体文件或目录>
started: <时间>
```

## 已完成

| ID | 模块 | 状态 | 结果 |
|---|---|---|---|
| T-001 | C++ 服务端融合 | DONE | 音频、模型、Pipeline、渲染和测试已合入 `main` |
| T-002 | Docker 构建 | DONE | Ubuntu 22.04 容器完整 CMake 构建通过 |
| T-003 | Pipeline 配置 | DONE | 线程数、FP16/Vulkan、锐化和 light mode 可配置 |
| T-004 | 统一基础抽象 | DONE | `FrameContext`、`ModelBackend`、`PipelineConfig` 已建立 |
| T-005 | 单帧性能基准 | DONE | 输出平均/最小/最大耗时和 FPS |
| T-006 | 端到端基准脚本 | DONE | 采集墙钟、首帧、阶段耗时、RSS、队列和 JSON |
| T-007 | 展馆原型 | PARKED | React/Vite 原型已提交，等待视觉方向确认 |
| T-008 | Git 同步 | DONE | 当前融合版本已推送到 GitHub `main` |

## 当前优先级

| ID | 模块 | 状态 | 目标 | 文件范围 |
|---|---|---|---|---|
| T-101 | dlib SIGILL 排障 | READY | 找到完整应用触发非法指令的依赖或编译选项 | `docker/`, `CMakeLists.txt`, `src/core/face_detector.cpp`, `scripts/` |
| T-102 | 真实 30 秒基准 | BLOCKED | 在 T-101 完成后获得端到端基线 | `scripts/run_e2e_benchmark.sh`, `docs/` |
| T-103 | 阶段级计时 | READY | 音频、检测、对齐、融合、编码统一输出指标 | `src/audio/`, `src/core/`, `src/video/`, `include/core/performance_metrics.h` |
| T-104 | 动态视频跟踪 | READY | 单人检测、跟踪、重检测和丢失恢复 | `include/core/frame_context.h`, `src/core/`, `src/model/inference_process.cpp` |
| T-105 | Wav2Lip 后端统一 | READY | 将现有 Wav2Lip 接入 `ModelBackend` | `include/model/`, `src/model/`, `include/core/pipeline.h` |
| T-106 | MuseTalk 完整后端 | BLOCKED | 等待 VAE、音频编码器和完整模型资源 | `src/core/musetalk_engine.cpp`, `src/model/` |
| T-107 | FastAPI 任务服务 | READY | 样本、任务状态、结果和指标 API | 新增 `server/` |
| T-108 | Docker Compose 部署 | READY | Nginx、FastAPI、SDK 服务统一启动 | `docker/`, 新增 `docker-compose.yml` |

## 领取规则

1. 优先领取 `READY` 且依赖已满足的任务。
2. `BLOCKED` 任务必须先写清阻塞证据，不得通过假数据标记完成。
3. 动态跟踪依赖 `FrameContext`，MuseTalk 依赖完整模型契约，API 服务依赖稳定的任务输出格式。
4. 同一时间只允许一个 Agent 修改同一任务的锁定文件。
5. 子任务完成后更新本表，并附测试命令和结果。

## 推荐执行顺序

```text
T-101 dlib SIGILL
  -> T-102 真实 30 秒基准
  -> T-103 阶段级计时
  -> T-104 动态视频跟踪
  -> T-105 Wav2Lip ModelBackend
  -> T-106 MuseTalk
  -> T-107 FastAPI
  -> T-108 Docker Compose
```

## 暂缓事项

- 前端视觉和页面细节：等待产品方向审批后再修改 `web/`。
- 多人脸、复杂姿态、遮挡恢复：单人动态视频稳定后再做。
- 公网部署和上传安全策略：API 服务跑通后再做。
