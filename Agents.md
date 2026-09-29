# AI 协作与开发规约

本文件是 AI Agent 进入项目后的第一份必读文档。任何 AI 在阅读本文件、`TODO.md` 和 `ARCHITECTURE.md` 之前，不得修改代码、删除文件、切换分支或提交 Git。

## 启动检查

每次开始工作必须依次执行：

```bash
pwd
git status --short
git branch -vv
sed -n '1,240p' Agents.md
sed -n '1,260p' TODO.md
sed -n '1,320p' ARCHITECTURE.md
```

确认当前分支、工作树、任务看板和已有锁之后，才能选择任务。若工作树存在不属于当前任务的修改，不得覆盖、重置或清理；先记录并避开相关文件。

## 分支与提交

- `main` 是当前融合基线，所有新工作从最新 `main` 创建功能分支。
- `master` 是同步前旧版本，只用于历史对照，不在其上开发。
- 分支命名使用 `feature/<module>`、`fix/<module>`、`perf/<module>` 或 `docs/<module>`。
- 一个提交只表达一个逻辑模块，提交信息使用 Conventional Commits：`feat:`、`fix:`、`perf:`、`test:`、`docs:`、`chore:`。
- 提交前必须运行与改动相关的 Docker 构建或测试，并在提交说明或任务看板中记录结果。
- 不允许在 `main` 上直接开发；完成后推送功能分支并通过审查再合并。

## 任务锁与冲突规避

所有任务必须先登记在 `TODO.md`。登记内容至少包括任务编号、负责人、分支、状态、文件范围和锁定时间。

- `TODO.md` 中标记为 `LOCKED` 的文件或模块，其他 Agent 不得修改。
- 同一文件不能被两个 Agent 同时锁定。
- 需要跨模块修改时，先在任务看板中扩大文件范围，再开始编辑。
- 完成、放弃或阻塞任务时必须释放锁并更新状态。

推荐锁记录格式：

```text
T-XXX | owner=<agent> | branch=<branch> | status=IN_PROGRESS
files: src/core/..., include/core/...
started: YYYY-MM-DD HH:MM Asia/Shanghai
```

## 技术约束

- 服务端依赖在 Docker 镜像 `weijhang/digital-human-env:ubuntu22.04-cpu-v1.0` 中验证。
- 保持通用 x86-64 编译参数，不引入宿主机专用 AVX/AVX2 指令。
- 默认 CPU、4 线程、锐化关闭；任何默认值改变必须有 benchmark 数据支持。
- 不把模型、构建产物、视频、音频、日志和 `web/node_modules` 提交到 Git。
- 不在热路径加入无界日志；调试输出必须可关闭或限频。
- 优先使用移动语义和已有缓存，但只有 profiling 证明复制或锁竞争是瓶颈时才做零拷贝重构。
- API 和数据结构保持向后兼容；涉及 `FrameContext`、`PipelineConfig`、`ModelBackend` 的修改必须同步更新 `ARCHITECTURE.md`。

## 验证命令

服务端：

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace \
  weijhang/digital-human-env:ubuntu22.04-cpu-v1.0 \
  bash -lc 'cmake -S /workspace -B /workspace/.docker-build -DCMAKE_BUILD_TYPE=Release && cmake --build /workspace/.docker-build -j2'
```

单帧基准：`./scripts/run_benchmark.sh ./models 50`

端到端基准：`BUILD_DIR=/workspace/.docker-build ./scripts/run_e2e_benchmark.sh`

前端：

```bash
cd web
npm install --registry=https://registry.npmmirror.com
npm run build
```

## 完成标准

任务只有在以下条件都满足时才能标记 `DONE`：

1. 实现和文档与任务目标一致。
2. 相关 Docker 构建或测试通过。
3. 没有越界修改其他 Agent 的锁定文件。
4. `TODO.md` 已更新结果、风险和后续任务。
5. 提交信息能清楚说明变更内容。
