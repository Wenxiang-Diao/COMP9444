| 内容 | 位置 |
|------|------|
| 训练/验证/测试划分（70% / 15% / 15%） | `data/train_split.csv`、`data/val_split.csv`、`data/test_split.csv` |
| 类别 ID ↔ 名称映射 | `data/class_maps.json` |
| PyTorch `Dataset` | `prepare_data.py` 中的 `LeafDataset` |
| 数据增强 | `train_transforms`（含增强）、`val_transforms`（无增强） |
| DataLoader | `train_loader`、`val_loader`、`test_loader` |

**随机种子：** `42`（保证划分可复现）  
**输入尺寸：** `299 × 299`  
**Batch size：** `32`

| 集合 | 样本数 |
|------|--------:|
| Train | 38813 |
| Val | 8317 |
| Test | 8318 |
| **合计** | **55448** |

---

## 环境与运行

1. 将原始数据集解压到**项目根目录**下，路径形如：

```text
Data for Identification of Plant Leaf Diseases Using a 9-layer Deep Convolutional Neural Network/
  └── Plant_leave_diseases_dataset_without_augmentation/
      ├── Apple___Apple_scab/
      ├── Tomato___healthy/
      └── ...
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 运行：

```bash
python devid_dataset/prepare_data.py
```

脚本会生成 `devide_dataset/data/` 下的 CSV 和 `class_maps.json`。

---

## CSV 字段说明

`train_split.csv` / `val_split.csv` / `test_split.csv` 字段如下：

| 列名 | 类型 | 含义 |
|------|------|------|
| `path` | str | 图片绝对路径 |
| `raw_class` | str | 原始文件夹名，如 `Tomato___Bacterial_spot` |
| `label_raw` | int | 组合类别 ID，**共 39 类**（默认多分类任务用这个） |
| `label_plant` | int | 植物种类 ID，**共 15 类** |
| `label_disease` | int | 病害类型 ID，**共 22 类** |


### 该用哪个标签？

| 角色 | 建议目标 |
|------|----------|
| Baseline / 单任务分类 | `label_raw`（39 类） |
| 多任务（植物头 + 病害头） | `label_plant` 和/或 `label_disease` |
| 评估 / 混淆矩阵 | 用 `class_maps.json` 把 ID 转成类名 |

类别数量：

- `label_raw`：0 … 38（39 类）
- `label_plant`：15 种植物
- `label_disease`：22 种病害

### `class_maps.json` 是什么？

模型输出的是数字 ID（例如 `29`），评估画图需要可读类名。该文件保存双向映射：

| 字段 | 用途 |
|------|------|
| `raw_to_idx` / `plant_to_idx` / `disease_to_idx` | 名称 → ID |
| `idx_to_raw` / `idx_to_plant` / `idx_to_disease` | ID → 名称（key 是字符串，如 `"29"`） |
| `num_classes` | `{"raw": 39, "plant": 15, "disease": 22}` |

评估示例：

```python
import json
from pathlib import Path

maps = json.loads(Path("devide_dataset/data/class_maps.json").read_text())
pred_id = 29
print(maps["idx_to_raw"][str(pred_id)])  # 例如 "Tomato___Bacterial_spot"
```

---

## 训练代码如何加载数据

### 方式 A：直接跑脚本后使用内存中的 loader

`prepare_data.py` 末尾会构建 `train_loader` 等。

### 方式 B：自己读 CSV

```python
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader

# 按实际项目结构调整 import
from devid_dataset.prepare_data import (
    LeafDataset,
    train_transforms,
    val_transforms,
    BATCH_SIZE,
)

DATA_DIR = Path("devide_dataset/data")  # 相对项目根目录

train_df = pd.read_csv(DATA_DIR / "train_split.csv")
val_df = pd.read_csv(DATA_DIR / "val_split.csv")
test_df = pd.read_csv(DATA_DIR / "test_split.csv")

train_loader = DataLoader(
    LeafDataset(train_df, transform=train_transforms),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)
val_loader = DataLoader(
    LeafDataset(val_df, transform=val_transforms),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)
test_loader = DataLoader(
    LeafDataset(test_df, transform=val_transforms),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
)

batch = next(iter(train_loader))
# batch["image"]:          (N, 3, 299, 299)，已按 ImageNet 均值方差归一化
# batch["label_raw"]:      (N,) 主任务 39 类标签
# batch["label_plant"]:    (N,)
# batch["label_disease"]:  (N,)
```

### 单任务训练循环示例

```python
for batch in train_loader:
    images = batch["image"].to(device)
    labels = batch["label_raw"].to(device)
    logits = model(images)          # shape: (N, 39)
    loss = criterion(logits, labels)
```