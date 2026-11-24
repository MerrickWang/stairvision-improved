# 推理权重清单

检查点放入 `weights/` 后可按 README 启动。表中的训练产物名称用于核对来源。SHA-256 对应项目最终部署文件。

| 启动所用文件 | 训练产物 | SHA-256 |
|---|---|---|
| `yolo_fragments.pt` | YOLO 短片段，24 轮，踏面／立面辅助监督 | `a0d3b76f3bd3bd2f38b731076178c1e963521f5b1023cdd9eee33c64d695d1a0` |
| `depthout.pt` | 本项目训练的 StairNet-DepthOut 检查点 | `3ab42a0f2296cc725fa907025c526defa7fc8b39bbb4f8db6d46561656065369` |
| `dense_edges.pt` | 凸边响应头，4 轮 | `e2d4c293aea7cb8c721893d3a778d80da0ffcf7c8c0ca7d1d05f4a9caa3eb112` |
| `decoder_only.pt` | Depth Anything V2 解码器，2 轮 | `180ffe4674a7d875739aa8cdf38db8956898ba6b00cdee101409f86f3703dac5` |

Depth Anything V2 的官方编码器和配置使用 README 中固定的 Hugging Face revision 下载。这里的解码器检查点与该版本的官方 Small 权重配合加载。
