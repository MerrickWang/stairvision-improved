# StairLab · 楼梯感知

输入单张 RGB 图像，输出台阶凸边、可见阶数和逐像素相对深度。GPU 交互页面支持上传图片、切换原图与结果、点选读取相对深度，以及下载深度数组。

## 方法与结果

- **阶缘与计数**采用 YOLO 凸边短片段预测、方向和覆盖约束拼接，再用踏面与立面辅助去重、物理台阶归组和有条件的漏阶补检。短片段设计参考 StairNet-DepthOut，相邻阶缘拆分和位置校正用于处理密集样本。
- **相对深度**使用 Depth Anything V2 Small，冻结预训练编码器并用整图 RGB-D 数据微调解码器。踏面排序监督作为消融对照。深度输出为相对逆深度，同图内数值较大表示较近。

| 固定数据对照 | 基线 | 最终方案 |
|---|---:|---:|
| 556 张验证图，凸边匹配 F1 | 43.68% | **85.44%** |
| 30 张人工计数图，数对 | 4/30 | **24/30** |
| 30 张人工计数图，平均绝对误差 | 2.57 阶 | **0.30 阶** |
| 154 张官方 RGB-D 测试图，归一化深度平均绝对误差 | 0.40485 | **0.22117** |

检测与计数以早期实现为基线，深度以官方 Small 权重为基线。30 张计数图包含 24 张开发图，6 张有完整计数的外部图只数对 1 张。官方 154 张图提供深度参考，缺少逐图阶数标注。深度误差按每张图归一化，单位为相对深度。

[变更日志](CHANGELOG.md) · [识别与计数统计](results/stair_detection.json) · [深度逐图结果](results/relative_depth.json)

## 环境和数据

使用 Linux、Python 3.12 和兼容的 CUDA PyTorch/torchvision。运行依赖见 `requirements.txt`。数据目录 `DATASET` 指向包含 `train`、`val`、`test` 的 [RGB-D Stair Dataset](https://data.mendeley.com/datasets/6kffmjt7g2/1) 根目录。记录的模型和数据遵循各自来源许可，权重与数据通过独立路径提供。

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
huggingface-cli download depth-anything/Depth-Anything-V2-Small-hf \
  --revision 5426e4f0f36572d16453bbda7a8389317b1bef99 \
  --local-dir models/depth-anything-v2-small
git clone https://github.com/MrChenWang/StairNet-DepthOut.git third_party/StairNet-DepthOut
git -C third_party/StairNet-DepthOut checkout 29457b1
```

## 启动交互页面

从仓库根目录运行。将 `weights/` 中的文件指向实际检查点，文件来源与校验值见 [权重清单](docs/weights.md)。`--examples` 使用官方测试图目录，提供正面、斜视和暗光示例。

```bash
python web_app.py \
  --yolo weights/yolo_fragments.pt \
  --depthout third_party/StairNet-DepthOut \
  --depthout-weights weights/depthout.pt \
  --dense weights/dense_edges.pt \
  --depth-model models/depth-anything-v2-small \
  --decoder weights/decoder_only.pt \
  --examples "$DATASET/test/images"
```

服务监听 `127.0.0.1:8765`。在远程主机运行时，通过 SSH 转发该端口后在本机浏览器访问。页面支持 JPEG、PNG 和 WebP，单张最大 12 MB，推理前缩放到长边不超过 1280 像素。

## 训练与评估

用现有数据生成短线缓存和训练 DepthOut。`PYTHONPATH` 用于定位外部网络定义。

```bash
python training/depthout/depthout_data.py --raw "$DATASET" --cache data/prepared_depthout
PYTHONPATH="$PWD/third_party/StairNet-DepthOut" python training/depthout/train_remote.py \
  --cache data/prepared_depthout --output runs/depthout

python train_yolo_fragments.py --label-cache data/prepared_depthout \
  --backbone "$POSE_WEIGHTS" --output runs/yolo_fragments --epochs 24 --surface-aux
python train_dense_stair_edges.py --dataset "$DATASET" --output runs/dense_edges --epochs 4
```

`POSE_WEIGHTS` 为楼梯姿态检测初始化权重，可用原有的 `prepare_data.py` 和 `train_yolo.py` 训练。短片段评估入口为 `evaluate_yolo_fragments.py`，阶数验证入口为 `evaluate_physical_counts.py`，历史配对评估入口为 `evaluate_version_comparison.py`。

```bash
python train_relative_depth_v40.py --dataset "$DATASET" --near-is low \
  --size 392 --batch 8 --epochs 2 --output runs/depth/4.0
python evaluate_relative_depth_v40.py --test "$DATASET/test" --run runs/depth/4.0
python predict_relative_depth.py --source path/to/image.jpg \
  --checkpoint runs/depth/4.0/decoder_only.pt --output runs/relative_depth
```

参考实现与许可见 [StairNet-DepthOut](https://github.com/MrChenWang/StairNet-DepthOut) 和 [Depth Anything V2 Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)。
