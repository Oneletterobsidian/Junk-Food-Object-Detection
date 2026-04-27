# ============================================================
# Notebook 2: Faster R-CNN 训练
# 在 Kaggle Notebook 中直接运行（新建一个 Notebook）
# GPU: P100, 预计总时间: 6-8小时
# ============================================================

# ── Step 1: 安装依赖 ──────────────────────────────────────────
!pip install pycocotools -q

# ── Step 2: 导入库 ────────────────────────────────────────────
import os, json, time, glob
import numpy as np
from PIL import Image
from collections import Counter

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

# ── Step 3: 数据集定义 ────────────────────────────────────────
DATASET_PATH = "/kaggle/input/junk-food-object-detection-dataset-yolo-format"

# 类别名（与 Notebook 1 保持一致）
NAMES = [
    "burger", "candy", "chips", "chocolate", "donut",
    "energy_drink", "french_fries", "fried_chicken", "hot_dog", "ice_cream",
    "nachos", "onion_rings", "pizza", "popcorn", "soda", "taco"
]
NC = len(NAMES)

class JunkFoodDataset(Dataset):
    def __init__(self, split="train", transforms=None):
        self.img_dir   = os.path.join(DATASET_PATH, split, "images")
        self.label_dir = os.path.join(DATASET_PATH, split, "labels")
        self.transforms = transforms
        self.imgs = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(img_name)[0] + ".txt"
        )

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts[:5])
                    # YOLO → xyxy
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    # 过滤无效框
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(int(cls) + 1)  # 0 保留给背景

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,),   dtype=torch.int64)
        else:
            boxes  = torch.tensor(boxes,  dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes":  boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        img = T.ToTensor()(img)

        if self.transforms:
            img = self.transforms(img)

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# 数据增强（仅训练集）
train_transforms = T.Compose([
    T.RandomHorizontalFlip(0.5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3)
])

train_dataset = JunkFoodDataset("train")
val_dataset   = JunkFoodDataset("valid")
test_dataset  = JunkFoodDataset("test")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True,
                          num_workers=2, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=4, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)
test_loader  = DataLoader(test_dataset,  batch_size=4, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

# ── Step 4: 构建模型 ──────────────────────────────────────────
def build_model(num_classes):
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    return model

model = build_model(NC)
model.to(device)
print("Faster R-CNN 模型已加载（ResNet50-FPN backbone，COCO预训练权重）")

# ── Step 5: 训练 ──────────────────────────────────────────────
EPOCHS = 10
optimizer = torch.optim.SGD(
    model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

train_losses = []
start_total = time.time()

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    start_epoch = time.time()

    for i, (images, targets) in enumerate(train_loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        if (i + 1) % 100 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | Step {i+1}/{len(train_loader)} | Loss: {losses.item():.4f}")

    lr_scheduler.step()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    elapsed = time.time() - start_epoch
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f} | Time: {elapsed/60:.1f} min")

total_time = time.time() - start_total
print(f"\n总训练时间: {total_time/60:.1f} 分钟")

# 保存模型
torch.save(model.state_dict(), "/kaggle/working/faster_rcnn_best.pth")
print("模型已保存: faster_rcnn_best.pth")

# ── Step 6: 评估（计算 mAP）─────────────────────────────────
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile

def evaluate_coco(model, data_loader, dataset, device):
    model.eval()
    coco_gt_data = {"images": [], "annotations": [], "categories": []}
    coco_dt_data = []
    ann_id = 1

    for i, name in enumerate(NAMES):
        coco_gt_data["categories"].append({"id": i+1, "name": name})

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for img, target, output in zip(images, targets, outputs):
                img_id = target["image_id"].item()
                h, w = img.shape[-2], img.shape[-1]

                coco_gt_data["images"].append({"id": img_id, "width": w, "height": h})

                for box, label in zip(target["boxes"].cpu(), target["labels"].cpu()):
                    x1, y1, x2, y2 = box.tolist()
                    coco_gt_data["annotations"].append({
                        "id": ann_id, "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "area": (x2-x1)*(y2-y1), "iscrowd": 0
                    })
                    ann_id += 1

                for box, label, score in zip(
                    output["boxes"].cpu(), output["labels"].cpu(), output["scores"].cpu()
                ):
                    if score < 0.05:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    coco_dt_data.append({
                        "image_id": img_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2-x1, y2-y1],
                        "score": float(score)
                    })

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_gt_data, f)
        gt_file = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(coco_dt_data, f)
        dt_file = f.name

    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(dt_file)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "map50":   round(float(coco_eval.stats[1]), 4),
        "map5095": round(float(coco_eval.stats[0]), 4)
    }

print("\n评估验证集...")
val_metrics = evaluate_coco(model, val_loader, val_dataset, device)
print(f"Val mAP@0.5:     {val_metrics['map50']}")
print(f"Val mAP@0.5:0.95: {val_metrics['map5095']}")

print("\n评估测试集...")
test_metrics = evaluate_coco(model, test_loader, test_dataset, device)
print(f"Test mAP@0.5:     {test_metrics['map50']}")
print(f"Test mAP@0.5:0.95: {test_metrics['map5095']}")

# ── Step 7: 速度测试 ──────────────────────────────────────────
model.eval()
dummy = [torch.zeros(3, 640, 640).to(device)]

# warmup
with torch.no_grad():
    for _ in range(10):
        model(dummy)

t0 = time.time()
with torch.no_grad():
    for _ in range(100):
        model(dummy)
fps = 100 / (time.time() - t0)
latency = 1000 / fps

print(f"\nFPS: {fps:.1f}")
print(f"Latency: {latency:.1f} ms")

# ── Step 8: 保存结果 ──────────────────────────────────────────
frcnn_results = {
    "model": "Faster R-CNN (ResNet50-FPN)",
    "val_map50":    val_metrics["map50"],
    "val_map5095":  val_metrics["map5095"],
    "test_map50":   test_metrics["map50"],
    "test_map5095": test_metrics["map5095"],
    "fps":          round(fps, 1),
    "latency_ms":   round(latency, 1),
    "train_time_min": round(total_time / 60, 1)
}

with open("/kaggle/working/frcnn_results.json", "w") as f:
    json.dump(frcnn_results, f, indent=2)

print("\n=== Faster R-CNN 结果 ===")
for k, v in frcnn_results.items():
    print(f"  {k}: {v}")
print("结果已保存: frcnn_results.json")

# ── Step 9: Loss 曲线 ─────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', color='#4472C4')
plt.title("Faster R-CNN Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("/kaggle/working/frcnn_loss_curve.png", dpi=150)
plt.show()
