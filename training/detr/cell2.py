import glob, time, json, os
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import box_iou
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"

DATASET_PATH = "/kaggle/input/datasets/youssefahmed003/junk-food-object-detection-dataset-yolo-format/yolo_dataset"
SAVE_DIR = "/kaggle/working/detr"
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
print(f"✅ 类别数: {nc}")

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_EPOCHS = 20        # 测试先跑2个epoch，确认没问题后改为20
BATCH_SIZE = 4
LEARNING_RATE = 1e-4

# ── Dataset ───────────────────────────────────────────────────
class JunkFoodDataset(Dataset):
    def __init__(self, img_dir, label_dir, processor):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.processor = processor
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

        annotations = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for ann_id, line in enumerate(f):
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, bw, bh = map(float, parts)
                    cls = int(cls)
                    if cls >= nc:
                        continue
                    x_min = (cx - bw/2) * w
                    y_min = (cy - bh/2) * h
                    box_w = bw * w
                    box_h = bh * h
                    annotations.append({
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": cls,
                        "bbox": [x_min, y_min, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    })

        target = {"image_id": idx, "annotations": annotations}
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"][0]
        return {"pixel_values": pixel_values, "labels": labels}


def collate_fn(batch):
    max_h = max(b["pixel_values"].shape[1] for b in batch)
    max_w = max(b["pixel_values"].shape[2] for b in batch)

    padded_images, masks = [], []
    for b in batch:
        c, h, w = b["pixel_values"].shape
        pad = torch.zeros(c, max_h, max_w)
        pad[:, :h, :w] = b["pixel_values"]
        padded_images.append(pad)
        mask = torch.zeros(max_h, max_w, dtype=torch.long)
        mask[:h, :w] = 1
        masks.append(mask)

    pixel_values = torch.stack(padded_images)
    pixel_mask   = torch.stack(masks)
    labels = [b["labels"] for b in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "labels": labels}


# ── 加载模型和 processor ──────────────────────────────────────
print("加载 DETR 预训练模型...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-50",
    num_labels=nc,
    ignore_mismatched_sizes=True
)
model.to(DEVICE)
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

train_dataset = JunkFoodDataset(f"{DATASET_PATH}/images/train", f"{DATASET_PATH}/labels/train", processor)
val_dataset   = JunkFoodDataset(f"{DATASET_PATH}/images/val",   f"{DATASET_PATH}/labels/val",   processor)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, collate_fn=collate_fn)
val_loader    = DataLoader(val_dataset,   batch_size=1,          shuffle=False, num_workers=2, collate_fn=collate_fn)
print(f"训练集: {len(train_dataset)} 张 | 验证集: {len(val_dataset)} 张")

# ── 优化器 ────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── mAP 评估函数 ──────────────────────────────────────────────
def compute_map_at_iou(all_pred, all_gt, iou_threshold):
    aps = []
    for cls in range(nc):
        n_gt = sum(len(g) for g in all_gt[cls])
        if n_gt == 0:
            continue
        tp_list, fp_list, sc_list = [], [], []
        for (pb, ps), gb in zip(all_pred[cls], all_gt[cls]):
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


def evaluate_map(model, loader, device):
    model.eval()
    all_pred = [[] for _ in range(nc)]
    all_gt   = [[] for _ in range(nc)]

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask   = batch["pixel_mask"].to(device)
            labels = batch["labels"]
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)

            target_sizes = torch.tensor(
                [[pixel_values.shape[-2], pixel_values.shape[-1]]] * len(labels)
            ).to(device)
            results = processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=target_sizes
            )

            for result, target in zip(results, labels):
                pb = result["boxes"].cpu()
                ps = result["scores"].cpu()
                pl = result["labels"].cpu()
                gb_norm = target["boxes"]
                gl = target["class_labels"]

                if len(gb_norm) > 0:
                    h, w = pixel_values.shape[-2], pixel_values.shape[-1]
                    gb = torch.stack([
                        (gb_norm[:, 0] - gb_norm[:, 2]/2) * w,
                        (gb_norm[:, 1] - gb_norm[:, 3]/2) * h,
                        (gb_norm[:, 0] + gb_norm[:, 2]/2) * w,
                        (gb_norm[:, 1] + gb_norm[:, 3]/2) * h,
                    ], dim=1)
                else:
                    gb = torch.zeros((0, 4))

                for cls in range(nc):
                    m_p = pl == cls
                    m_g = gl == cls
                    all_pred[cls].append((pb[m_p], ps[m_p]))
                    all_gt[cls].append(gb[m_g])

    map50 = compute_map_at_iou(all_pred, all_gt, 0.5)
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    map5095 = float(np.mean([compute_map_at_iou(all_pred, all_gt, t) for t in iou_thresholds]))
    return map50, map5095


# ── 训练循环 ──────────────────────────────────────────────────
print(f"\n开始训练 DETR，共 {NUM_EPOCHS} epochs...")
train_losses, val_map50s, val_map5095s = [], [], []
best_map50 = 0.0
start = time.time()

for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        pixel_values = batch["pixel_values"].to(DEVICE)
        pixel_mask   = batch["pixel_mask"].to(DEVICE)
        labels = [{k: v.to(DEVICE) for k, v in t.items()} for t in batch["labels"]]

        outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(train_loader)}] loss: {loss.item():.4f}")
    train_loss = total_loss / len(train_loader)

    map50, map5095 = evaluate_map(model, val_loader, DEVICE)
    lr_scheduler.step()

    train_losses.append(train_loss)
    val_map50s.append(map50)
    val_map5095s.append(map5095)

    epoch_time = time.time() - epoch_start
    elapsed    = time.time() - start
    eta        = elapsed / epoch * (NUM_EPOCHS - epoch)

    print(f"Epoch {epoch}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} | "
          f"mAP@0.5: {map50:.4f} | "
          f"mAP@0.5:0.95: {map5095:.4f} | "
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
dummy = torch.zeros(1, 3, 640, 640).to(DEVICE)
for _ in range(10):
    with torch.no_grad(): model(pixel_values=dummy)
t0 = time.time()
for _ in range(100):
    with torch.no_grad(): model(pixel_values=dummy)
fps = round(100 / (time.time() - t0), 1)
print(f"FPS: {fps}")

# ── 可视化 ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(range(1, NUM_EPOCHS+1), train_losses, color='#4472C4')
axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
axes[1].plot(range(1, NUM_EPOCHS+1), val_map50s, color='#ED7D31')
axes[1].set_title("Validation mAP@0.5"); axes[1].set_xlabel("Epoch")
axes[2].plot(range(1, NUM_EPOCHS+1), val_map5095s, color='#A9D18E')
axes[2].set_title("Validation mAP@0.5:0.95"); axes[2].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/training_curves.png", dpi=150)
plt.show()
print("训练曲线已保存")

# ── 结果汇总 ──────────────────────────────────────────────────
result = {
    "model": "DETR ResNet-50",
    "epochs": NUM_EPOCHS,
    "best_map50": round(best_map50, 4),
    "best_map5095": round(val_map5095s[val_map50s.index(max(val_map50s))], 4),
    "fps": fps,
    "train_time_min": round(elapsed_total / 60, 1),
    "final_train_loss": round(train_losses[-1], 4),
}
with open(f"{SAVE_DIR}/result.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n=== DETR 训练结果 ===")
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