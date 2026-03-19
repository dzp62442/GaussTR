# 数据准备指南 — GaussTR

本文面向已经拥有以下原始数据的情况：

- 原始 nuScenes 数据
- Occ3D occupancy ground truth 数据

目标是把 **GaussTR 真正需要的全部中间产物** 说明清楚，并给出可直接执行的准备流程。

## 1. 你最终需要准备出的东西

### 所有模型都需要

- `data/nuscenes/`
  - 原始 nuScenes 数据
  - `nuscenes_infos_train.pkl`
  - `nuscenes_infos_val.pkl`
  - `gts/`（Occ3D ground truth）
- `data/nuscenes_metric3d/`：每张图像对应一个 `.npy` 深度文件

### `GaussTR-FeatUp` 额外需要

- `data/nuscenes_featup/`：每张图像对应一个 `.npy` 特征文件
- `data/nuscenes_grounded_sam2/`：每张图像对应一个 `.npy` 语义伪标签文件
- `ckpts/text_proto_embeds_clip.pth`

### `GaussTR-Talk2DINO` 额外需要

- `ckpts/text_proto_embeds_talk2dino.pth`

## 2. 先看结论：按模型分支需要做什么

### 如果你只跑 `GaussTR-Talk2DINO`

你只需要完成：

1. 准备 `data/nuscenes`
2. 准备 `data/nuscenes/nuscenes_infos_{train,val}.pkl`
3. 运行 `tools/update_data.py`
4. 放置 `data/nuscenes/gts`
5. 运行 `tools/generate_depth.py`
6. 下载 `ckpts/text_proto_embeds_talk2dino.pth`

你**不需要**生成：

- `data/nuscenes_featup`
- `data/nuscenes_grounded_sam2`

### 如果你跑 `GaussTR-FeatUp`

你需要完成：

1. 准备 `data/nuscenes`
2. 准备 `data/nuscenes/nuscenes_infos_{train,val}.pkl`
3. 运行 `tools/update_data.py`
4. 放置 `data/nuscenes/gts`
5. 运行 `tools/generate_depth.py`
6. 运行 `tools/generate_featup.py`
7. 运行 `tools/generate_grounded_sam2.py`，或自行修改 `configs/gausstr_featup.py`
8. 下载 `ckpts/text_proto_embeds_clip.pth`

## 3. 推荐目录结构

建议最终整理成下面这样：

```text
GaussTR/
├── ckpts/
│   ├── text_proto_embeds_clip.pth
│   └── text_proto_embeds_talk2dino.pth
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-trainval/
│   │   ├── nuscenes_infos_train.pkl
│   │   ├── nuscenes_infos_val.pkl
│   │   └── gts/
│   ├── nuscenes_metric3d/
│   ├── nuscenes_featup/
│   └── nuscenes_grounded_sam2/
└── ...
```

## 4. 流程总览

完整流程分 4 个阶段：

1. 准备标准 nuScenes infos
2. 给 infos 补 `scene_idx`
3. 放置 Occ3D ground truth
4. 生成深度 / 特征 / 分割伪标签

## 5. 阶段 A：准备标准 nuScenes infos

GaussTR 不直接读取原始 nuScenes 元数据，而是通过 `mmdet3d` 风格的 `nuscenes_infos_train.pkl` 和 `nuscenes_infos_val.pkl` 读取样本信息。GaussTR README 已经明确依赖 OpenMMLab V2.0 格式，当前环境安装的也是 `mmdet3d==1.4.0`，所以直接使用 **`mmdetection3d v1.4.0`** 的官方脚本。

#### 5.1 前提目录

在运行 `mmdetection3d v1.4.0` 官方 `tools/create_data.py` 之前，原始 nuScenes 目录至少应满足：

```text
data/nuscenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
└── v1.0-test/        # 如果你有 test set
```

#### 5.2 运行指令

从 **GaussTR 根目录** 先准备 `mmdetection3d v1.4.0` 仓库，放到 `third_party/` 下面，再执行它的官方脚本生成 infos：

```bash
mkdir -p third_party
git clone https://github.com/open-mmlab/mmdetection3d.git third_party/mmdetection3d
cd third_party/mmdetection3d
git checkout v1.4.0

cd ../..  # 回到 GaussTR 根目录
PYTHONPATH=./third_party/mmdetection3d python third_party/mmdetection3d/tools/create_data.py nuscenes \
    --root-path ./data/nuscenes \
    --out-dir ./data/nuscenes \
    --extra-tag nuscenes
```

根据 `mmdetection3d v1.4.0` 官方 `tools/create_data.py`，运行后通常会在 `data/nuscenes/` 下生成：

- `nuscenes_gt_database/`
- `nuscenes_infos_train.pkl`
- `nuscenes_infos_val.pkl`（GaussTR 直接需要的核心文件）
- `nuscenes_infos_test.pkl`
- `nuscenes_dbinfos_train.pkl`（GaussTR 直接需要的核心文件）

## 6. 阶段 B：给 infos 补 `scene_idx`

这个项目要求 `nuscenes_infos_*.pkl` 中额外带上 `scene_token` 和 `scene_idx`，为了和 occupancy ground truth 对齐。

从 **GaussTR 仓库根目录** 运行：

```bash
python tools/update_data.py nuscenes \
  --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes \
  --extra-tag nuscenes
```

它会读取 `data/nuscenes/nuscenes_infos_train.pkl`、`data/nuscenes/nuscenes_infos_val.pkl`，然后把补充过 `scene_token` 和 `scene_idx` 的结果 **重新写回同名文件**。

## 7. 阶段 C：放置 Occ3D ground truth

把 Occ3D 的 ground truth 放到：

```text
data/nuscenes/gts
```

## 8. 阶段 D：生成中间监督数据

## 8.1 生成 Metric3D 深度

两条模型分支都需要。输出目录：`data/nuscenes_metric3d/`

从 **GaussTR 仓库根目录** 运行：

```bash
PYTHONPATH=. python tools/generate_depth.py
```

脚本行为：

- 使用 `Metric3D` 生成每张图像的深度
- 输出文件名格式是：
  - 输入图像：`xxx.jpg`
  - 输出深度：`data/nuscenes_metric3d/xxx.npy`
- 默认只处理：
  - `nuscenes_infos_train.pkl`
  - `nuscenes_infos_val.pkl`
- 在 `RTX 4090` 上大约需要 10 小时

结果规模：

- 按作者在 issue `#36` 中给出的说明，`v1.0-trainval` 下：
  - `nuscenes_infos_train.pkl` 有 `28,130` 条
  - `nuscenes_infos_val.pkl` 有 `6,019` 条
  - 每条样本对应 `6` 张图
- 因此 `nuscenes_metric3d` 的预期文件数约为：`(28130 + 6019) x 6 = 204894`

## 8.2 生成 FeatUp 特征

仅 `GaussTR-FeatUp` 需要。输出目录：`data/nuscenes_featup/`

先克隆 `FeatUp` 仓库到 `third_party/` 下，再执行**GaussTR 自带的** `tools/generate_featup.py`。从 **GaussTR 根目录** 运行：

```bash
mkdir -p third_party
git clone https://github.com/mhamilton723/FeatUp.git third_party/FeatUp
pip install -e ./third_party/FeatUp
PYTHONPATH=. python tools/generate_featup.py
```

脚本行为：

- 遍历 `data/nuscenes/samples/` 下所有相机目录
- 读取所有图像
- 生成高分辨率特征
- 经过 `avg_pool2d(..., 16)` 后保存为 `.npy`

结果规模：

- 作者在 issue `#36` 中说明，`generate_featup.py` 是按 `samples/` 全量目录扫图，因此通常会覆盖：
  - `trainval`
  - `mini`
  - `test`
- 预期文件数约为：`40157 x 6 = 240942`

这个数量比 `nuscenes_metric3d` 多是**正常现象**。

## 8.3 生成 Grounded-SAM-2 分割伪标签

仅 `GaussTR-FeatUp` 需要。输出目录 / 配置文件期望读取的是：`data/nuscenes_grounded_sam2/`

README 把它描述成“可选辅助监督”，但要注意：

- 默认提供的 [configs/gausstr_featup.py](/home/dzp62442/Projects/GaussTR/configs/gausstr_featup.py) 会直接读取 `data/nuscenes_grounded_sam2`
- 如果你不生成这部分数据，就需要同步修改配置，去掉对应的 `LoadFeatMaps(..., key='sem_seg')` 和分割监督

先克隆 `Grounded-SAM-2` 仓库到 `third_party/` 下：

```bash
mkdir -p third_party
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git third_party/Grounded-SAM-2
```

按 `Grounded-SAM-2` 官方 README，单独开一个环境安装：

```bash
conda create -n grounded-sam2 python=3.10 -y
conda activate grounded-sam2

pip install --upgrade pip wheel
pip install 'setuptools<81'
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

pip install -e ./third_party/Grounded-SAM-2
pip install --no-build-isolation -e ./third_party/Grounded-SAM-2/grounding_dino
```

然后从终端下载本项目实际需要的两个权重：

```bash
mkdir -p third_party/Grounded-SAM-2/checkpoints third_party/Grounded-SAM-2/gdino_checkpoints

wget -O third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt

wget -O third_party/Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
```

`tools/generate_grounded_sam2.py` 的路径已改成可直接从 **GaussTR 根目录** 运行。激活 `grounded-sam2` 环境后执行：

```bash
PYTHONPATH=./third_party/Grounded-SAM-2 python tools/generate_grounded_sam2.py
```

结果规模：

- 和 `FeatUp` 一样，这个脚本是扫 `samples/` 全目录，因此通常预期文件数也约为：`240942`

## 9. 文本原型 embedding

这部分不是数据集本身，但模型运行需要。来源：[GaussTR Releases](https://github.com/hustvl/GaussTR/releases/)

- GaussTR-FeatUp 下载：`ckpts/text_proto_embeds_clip.pth`
- GaussTR-Talk2DINO 下载：`ckpts/text_proto_embeds_talk2dino.pth`

## 10. 各分支与数据依赖对照表

| 产物 | Talk2DINO | FeatUp |
| --- | --- | --- |
| `data/nuscenes` 原始数据 | 必需 | 必需 |
| `nuscenes_infos_train.pkl` / `val.pkl` | 必需 | 必需 |
| `update_data.py` 更新后的 infos | 必需 | 必需 |
| `data/nuscenes/gts` | 必需 | 必需 |
| `data/nuscenes_metric3d` | 必需 | 必需 |
| `data/nuscenes_featup` | 不需要 | 必需 |
| `data/nuscenes_grounded_sam2` | 不需要 | 默认配置必需；跳过时需改配置 |
| `ckpts/text_proto_embeds_talk2dino.pth` | 必需 | 不需要 |
| `ckpts/text_proto_embeds_clip.pth` | 不需要 | 必需 |

## 11. 参考

- MMDetection3D 数据准备总文档  
  https://mmdetection3d.readthedocs.io/en/latest/user_guides/dataset_prepare.html
- MMDetection3D nuScenes 专项文档  
  https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/nuscenes.html
- `mmdetection3d v1.4.0` 官方 `create_data.py`  
  https://github.com/open-mmlab/mmdetection3d/blob/v1.4.0/tools/create_data.py
- V2.0 info 更新脚本  
  https://github.com/open-mmlab/mmdetection3d/blob/main/tools/dataset_converters/update_infos_to_v2.py
