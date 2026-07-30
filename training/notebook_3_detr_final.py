# ============================================================
# Notebook 3: DETR 预训练推理 + 最终结果汇总与可视化
# 在 Kaggle Notebook 中直接运行（新建一个 Notebook）
# GPU: P100, 预计总时间: 1-2小时
# ============================================================

# ── Step 1: 安装依赖 ──────────────────────────────────────────
!pip install transformers pycocotools -q

# ── Step 2: 导入库 ────────────────────────────────────────────
import os, json, time
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import DetrImageProcessor, DetrForObjectDetection
from torch.utils.data import Dataset, DataLoader
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

DATASET_PATH = "/kaggle/input/junk-food-object-detection-dataset-yolo-format"

NAMES = [
    "burger", "candy", "chips", "chocolate", "donut",
    "energy_drink", "french_fries", "fried_chicken", "hot_dog", "ice_cream",
    "nachos", "onion_rings", "pizza", "popcorn", "soda", "taco"
]

# ── Step 3: 加载预训练 DETR ───────────────────────────────────
print("加载 DETR 预训练模型 (COCO 权重)...")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.to(device)
model.eval()
print("DETR 加载完成")

# COCO 类别中与垃圾食品相关的 ID
# COCO 有 pizza(53), hot dog(52), donut(54), cake(55), sandwich(48)
COCO_FOOD_IDS = {48: "sandwich", 52: "hot_dog", 53: "pizza", 54: "donut", 55: "cake"}

# ── Step 4: 速度测试 ──────────────────────────────────────────
test_img_path = glob.glob(f"{DATASET_PATH}/test/images/*.jpg")[0]
test_img = Image.open(test_img_path).convert("RGB")

# warmup
inputs = processor(images=test_img, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    for _ in range(5):
        model(**inputs)

# FPS 测试
t0 = time.time()
with torch.no_grad():
    for _ in range(50):
        model(**inputs)
fps = 50 / (time.time() - t0)
latency = 1000 / fps
print(f"DETR FPS: {fps:.1f} | Latency: {latency:.1f} ms")

# ── Step 5: 在测试集上推理（定性评估）────────────────────────
test_imgs = glob.glob(f"{DATASET_PATH}/test/images/*.jpg")[:200]

all_detections = []
print(f"在 {len(test_imgs)} 张测试图像上推理...")

with torch.no_grad():
    for img_path in test_imgs:
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            label_id = label.item()
            coco_name = model.config.id2label.get(label_id, "unknown")
            detections.append({
                "label_id": label_id,
                "coco_name": coco_name,
                "score": round(score.item(), 3),
                "box": box.tolist()
            })
        all_detections.append({
            "img": os.path.basename(img_path),
            "detections": detections
        })

# 统计检测到的类别分布
from collections import Counter
coco_counter = Counter()
for d in all_detections:
    for det in d["detections"]:
        coco_counter[det["coco_name"]] += 1

print("\n检测到的 COCO 类别 Top 15:")
for name, cnt in coco_counter.most_common(15):
    print(f"  {name}: {cnt}")

# ── Step 6: 可视化推理结果（8张样本）────────────────────────
sample_imgs = [p for p in test_imgs if any(
    det["coco_name"] in ["pizza","donut","hot dog","sandwich"]
    for det in all_detections[test_imgs.index(p)]["detections"]
)][:8]

if len(sample_imgs) < 8:
    sample_imgs = test_imgs[:8]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

with torch.no_grad():
    for ax, img_path in zip(axes, sample_imgs):
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]]).to(device)
        results = processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=target_sizes
        )[0]

        ax.imshow(img)
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            x1, y1, x2, y2 = box.tolist()
            coco_name = model.config.id2label.get(label.item(), "?")
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{coco_name} {score:.2f}",
                    color='lime', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
        ax.axis('off')
        ax.set_title(os.path.basename(img_path)[:20], fontsize=8)

plt.suptitle("DETR Inference (Pretrained COCO Weights, No Fine-tuning)", fontsize=14)
plt.tight_layout()
plt.savefig("/kaggle/working/detr_qualitative.png", dpi=150)
plt.show()
print("DETR 定性结果已保存: detr_qualitative.png")

# ── Step 7: 保存 DETR 结果 ────────────────────────────────────
detr_results = {
    "model": "DETR (ResNet-50, COCO pretrained, no fine-tuning)",
    "note": "Evaluated with COCO pretrained weights only. "
            "mAP not computed due to category mismatch with training set.",
    "fps": round(fps, 1),
    "latency_ms": round(latency, 1),
    "top_detected_coco_classes": dict(coco_counter.most_common(10))
}

with open("/kaggle/working/detr_results.json", "w") as f:
    json.dump(detr_results, f, indent=2)

# ── Step 8: 三模型最终对比汇总 ───────────────────────────────
# 手动填入 Notebook 1 和 2 的最优结果
# （从 yolo_results.json 和 frcnn_results.json 读取）

try:
    with open("/kaggle/working/yolo_results.json") as f:
        yolo_all = json.load(f)
    # 取 mAP@0.5 最高的增强组
    best_yolo = max(yolo_all, key=lambda x: x["map50"])
except:
    best_yolo = {"exp": "aug_strong", "map50": None, "map5095": None, "fps": None}

try:
    with open("/kaggle/working/frcnn_results.json") as f:
        frcnn = json.load(f)
except:
    frcnn = {"test_map50": None, "test_map5095": None, "fps": None}

summary = [
    {
        "Model":        "YOLOv8s (best aug)",
        "Type":         "One-stage",
        "mAP@0.5":      best_yolo.get("map50"),
        "mAP@0.5:0.95": best_yolo.get("map5095"),
        "FPS":          best_yolo.get("fps"),
        "Fine-tuned":   "Yes"
    },
    {
        "Model":        "Faster R-CNN (ResNet50)",
        "Type":         "Two-stage",
        "mAP@0.5":      frcnn.get("test_map50"),
        "mAP@0.5:0.95": frcnn.get("test_map5095"),
        "FPS":          frcnn.get("fps"),
        "Fine-tuned":   "Yes"
    },
    {
        "Model":        "DETR (ResNet50)",
        "Type":         "Transformer",
        "mAP@0.5":      "N/A*",
        "mAP@0.5:0.95": "N/A*",
        "FPS":          round(fps, 1),
        "Fine-tuned":   "No (COCO weights)"
    },
]

import pandas as pd
df_summary = pd.DataFrame(summary)
print("\n=== 三模型最终对比 ===")
print(df_summary.to_string(index=False))
df_summary.to_csv("/kaggle/working/final_model_comparison.csv", index=False)

# ── Step 9: 精度-速度散点图 ──────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#4472C4", "#ED7D31", "#A9D18E"]
models = ["YOLOv8s", "Faster R-CNN", "DETR*"]

fps_vals  = [best_yolo.get("fps", 0), frcnn.get("fps", 0), round(fps, 1)]
map_vals  = [best_yolo.get("map50", 0), frcnn.get("test_map50", 0), None]

for i, (model_name, color) in enumerate(zip(models, colors)):
    if map_vals[i] is not None:
        ax.scatter(fps_vals[i], map_vals[i], s=200, color=color, zorder=5, label=model_name)
        ax.annotate(model_name, (fps_vals[i], map_vals[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)
    else:
        ax.axvline(x=fps_vals[i], color=color, linestyle='--', alpha=0.5,
                   label=f"{model_name} (no mAP)")

ax.set_xlabel("FPS (Frames Per Second) →  faster", fontsize=11)
ax.set_ylabel("mAP@0.5 →  more accurate", fontsize=11)
ax.set_title("Accuracy vs Speed Trade-off", fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.02, "* DETR: COCO pretrained, no fine-tuning, mAP not computed",
        transform=ax.transAxes, fontsize=8, color='gray')
plt.tight_layout()
plt.savefig("/kaggle/working/accuracy_speed_tradeoff.png", dpi=150)
plt.show()
print("精度-速度图已保存: accuracy_speed_tradeoff.png")
print("\n所有结果已保存完毕！")
