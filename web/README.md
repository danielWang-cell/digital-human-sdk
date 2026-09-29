# Digital Human Lab preview

这是第一版静态展馆原型，使用预制状态和模拟性能数据，不依赖在线推理。

```bash
npm install
npm run dev
```

生产构建：

```bash
npm run build
npm run preview
```

页面包含视频演示占位、Pipeline 节点交互、性能曲线、队列状态和工程日志。确认视觉方向后，再把 mock 数据替换成 FastAPI 同源接口。
