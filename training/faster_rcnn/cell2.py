import glob, time, json, os
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from PIL import Image
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"

DATASET_PATH = "/kaggle/input/datasets/youssefahmed003/junk-food-object-detection-dataset-yolo-format/yolo_dataset"
SAVE_DIR = "/kaggle/working/faster_rcnn"
os.makedirs(SAVE_DIR, exist_ok=True)

# 自动读取类别数
label_files = glob.glob(f"{DATASET_PATH}/labels/train/*.txt")
class_ids = set()
for f in label_files[:500]:
    with open(f) as fp:
        for line in fp:
            parts = line.strip().split()
            if parts:
                class_ids.add(int(parts[0]))
nc = max(class_ids) + 1
NUM_CLASSES = nc + 1  # +1 for background
print(f"✅ 类别数: {nc}, NUM_CLASSES(含背景): {NUM_CLASSES}")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_EPOCHS = 20        # 测试先跑2个epoch，确认没问题后改为20
BATCH_SIZE = 4
LEARNING_RATE = 0.005

# ── Dataset ───────────────────────────────────────────────────
class JunkFoodDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.imgs = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        boxes, labels = [], []

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts)
                    cls = int(cls)
                    if cls >= nc:
                        continue
                    x1 = max(0, (cx - bw/2) * w)
                    y1 = max(0, (cy - bh/2) * h)
                    x2 = min(w, (cx + bw/2) * w)
                    y2 = min(h, (cy + bh/2) * h)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        img = torchvision.transforms.functional.to_tensor(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


train_dataset = JunkFoodDataset(f"{DATASET_PATH}/images/train", f"{DATASET_PATH}/labels/train")
val_dataset   = JunkFoodDataset(f"{DATASET_PATH}/images/val",   f"{DATASET_PATH}/labels/val")
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, collate_fn=collate_fn)
val_loader    = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=2, collate_fn=collate_fn)
print(f"训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张")

# ── 模型 ──────────────────────────────────────────────────────
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
model.to(DEVICE)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── mAP@0.5 评估 ──────────────────────────────────────────────
def evaluate_map(model, loader, device, iou_threshold=0.5):
    model.eval()
    aps = []
    all_pred = [[] for _ in range(NUM_CLASSES)]
    all_gt   = [[] for _ in range(NUM_CLASSES)]

    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                pb = output['boxes'].cpu()
                ps = output['scores'].cpu()
                pl = output['labels'].cpu()
                gb = target['boxes']
                gl = target['labels']
                for cls in range(1, NUM_CLASSES):
                    m_p = pl == cls
                    m_g = gl == cls
                    all_pred[cls].append((pb[m_p], ps[m_p]))
                    all_gt[cls].append(gb[m_g])

    for cls in range(1, NUM_CLASSES):
        n_gt = sum(len(g) for g in all_gt[cls])
        if n_gt == 0:
            continue
        boxes_list = [p[0] for p in all_pred[cls]]
        scores_list = [p[1] for p in all_pred[cls]]
        tp_list, fp_list, sc_list = [], [], []
        for pb, ps, gb in zip(boxes_list, scores_list, all_gt[cls]):
            if len(pb) == 0:
                continue
            sc_list.extend(ps.tolist())
            if len(gb) == 0:
                tp_list.extend([0]*len(pb)); fp_list.extend([1]*len(pb)); continue
            iou = box_iou(pb, gb)
            matched = torch.zeros(len(gb), dtype=torch.bool)
            for i in range(len(pb)):
                best_iou, best_j = iou[i].max(0)
                if best_iou >= iou_threshold and not matched[best_j]:
                    tp_list.append(1); fp_list.append(0); matched[best_j] = True
                else:
                    tp_list.append(0); fp_list.append(1)
        if not sc_list:
            continue
        idx = np.argsort(-np.array(sc_list))
        tp = np.cumsum(np.array(tp_list)[idx])
        fp = np.cumsum(np.array(fp_list)[idx])
        rec = tp / n_gt
        pre = tp / (tp + fp + 1e-6)
        aps.append(np.trapz(pre, rec))

    return float(np.mean(aps)) if aps else 0.0

# ── 训练循环 ──────────────────────────────────────────────────
print(f"\n开始训练 Faster R-CNN，共 {NUM_EPOCHS} epochs...")
train_losses, val_map50s = [], []
best_map50 = 0.0
start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    # 训练
    model.train()
    total_loss = 0
    for i, (images, targets) in enumerate(train_loader):
        images  = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(train_loader)}] loss: {losses.item():.4f}")
    train_loss = total_loss / len(train_loader)

    # 验证
    map50 = evaluate_map(model, val_loader, DEVICE)
    lr_scheduler.step()

    train_losses.append(train_loss)
    val_map50s.append(map50)

    epoch_time = time.time() - epoch_start
    elapsed    = time.time() - start
    eta        = elapsed / epoch * (NUM_EPOCHS - epoch)

    print(f"Epoch {epoch}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"mAP@0.5: {map50:.4f} | "
          f"Time: {epoch_time/60:.1f}min | "
          f"ETA: {eta/60:.1f}min")

    if map50 > best_map50:
        best_map50 = map50
        torch.save(model.state_dict(), f"{SAVE_DIR}/best.pt")
        print(f"  ✅ 保存最佳模型 (mAP@0.5: {map50:.4f})")

elapsed_total = time.time() - start
print(f"\n✅ 训练完成，用时 {elapsed_total/60:.1f} 分钟")
print(f"最佳 mAP@0.5: {best_map50:.4f}")

# ── FPS 测试 ──────────────────────────────────────────────────
model.eval()
dummy = torch.zeros(3, 640, 640).to(DEVICE)
for _ in range(10):
    with torch.no_grad(): model([dummy])
t0 = time.time()
for _ in range(100):
    with torch.no_grad(): model([dummy])
fps = round(100 / (time.time() - t0), 1)
print(f"FPS: {fps}")

# ── 可视化 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(range(1, NUM_EPOCHS+1), train_losses, color='#4472C4')
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
axes[1].plot(range(1, NUM_EPOCHS+1), val_map50s, color='#ED7D31')
axes[1].set_title("Validation mAP@0.5"); axes[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/training_curves.png", dpi=150)
plt.show()
print("训练曲线已保存")

# ── 结果汇总 ──────────────────────────────────────────────────
result = {
    "model": "Faster R-CNN ResNet50-FPN",
    "epochs": NUM_EPOCHS,
    "best_map50": round(best_map50, 4),
    "fps": fps,
    "train_time_min": round(elapsed_total / 60, 1),
    "final_train_loss": round(train_losses[-1], 4),
}
with open(f"{SAVE_DIR}/result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n=== Faster R-CNN 训练结果 ===")
for k, v in result.items():
    print(f"  {k}: {v}")

# ── 确认文件 ──────────────────────────────────────────────────
weight_path = f"{SAVE_DIR}/best.pt"
if os.path.exists(weight_path):
    print(f"\n✅ 找到模型: {weight_path} ({os.path.getsize(weight_path)/1024/1024:.1f} MB)")
    print("请在右侧 Output 面板中下载 best.pt")
else:
    print("❌ 找不到权重文件")
    for root, dirs, files in os.walk('/kaggle/working/'):
        for f in files:
            if f.endswith('.pt'):
                print(f"找到: {os.path.join(root, f)}")