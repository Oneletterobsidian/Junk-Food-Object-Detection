# ============================================================
# Notebook 1: YOLOv8 Baseline + Augmentation Ablation
# 在 Kaggle Notebook 中直接运行
# GPU: P100, 预计总时间: 3-4小时
# ============================================================
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 
    "torch==2.2.0+cu118", "torchvision==0.17.0+cu118",
    "--index-url", "https://download.pytorch.org/whl/cu118", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics==8.2.0", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "ray", "-y"])

# ── Step 2: 查看数据集结构 ────────────────────────────────────
import os

DATASET_PATH = "/kaggle/input/datasets/youssefahmed003/junk-food-object-detection-dataset-yolo-format/yolo_dataset"

for root, dirs, files in os.walk(DATASET_PATH):
    level = root.replace(DATASET_PATH, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f'{indent}{os.path.basename(root)}/')
    if level < 2:
        subindent = ' ' * 2 * (level + 1)
        for f in files[:3]:
            print(f'{subindent}{f}')

# ── Step 3: 生成 data.yaml ────────────────────────────────────
import yaml, glob

# 自动读取类别名
label_files = glob.glob(f"{DATASET_PATH}/labels/train/*.txt")
class_ids = set()
for f in label_files[:500]:
    with open(f) as fp:
        for line in fp:
            class_ids.add(int(line.split()[0]))
nc = max(class_ids) + 1

# 16类垃圾食品名称（按数据集顺序，如有不符请手动修改）
names = [
    "burger", "candy", "chips", "chocolate", "donut",
    "energy_drink", "french_fries", "fried_chicken", "hot_dog", "ice_cream",
    "nachos", "onion_rings", "pizza", "popcorn", "soda", "taco"
]
if len(names) != nc:
    names = [str(i) for i in range(nc)]  # fallback: 用数字代替

data_yaml = {
    "path": DATASET_PATH,
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc":    nc,
    "names": names
}

yaml_path = "/kaggle/working/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"Classes: {nc}")
print(f"Names: {names}")

# ── Step 4: 类别分布分析 ──────────────────────────────────────
from collections import Counter
import matplotlib.pyplot as plt

counter = Counter()
for f in glob.glob(f"{DATASET_PATH}/labels/train/*.txt"):
    with open(f) as fp:
        for line in fp:
            counter[int(line.split()[0])] += 1

plt.figure(figsize=(12, 4))
plt.bar([names[i] if i < len(names) else str(i) for i in sorted(counter)],
        [counter[i] for i in sorted(counter)])
plt.xticks(rotation=45, ha='right')
plt.title("Class Distribution (Train Set)")
plt.tight_layout()
plt.savefig("/kaggle/working/class_distribution.png", dpi=150)
plt.show()
print("类别分布图已保存")

# ── Step 5: 训练函数定义 ──────────────────────────────────────
from ultralytics import YOLO
import time, json

def train_yolo(exp_name, augment_args, epochs=50, imgsz=640):
    """
    训练一个 YOLOv8s 模型并返回结果
    augment_args: dict，传入 model.train() 的增强参数
    """
    print(f"\n{'='*50}")
    print(f"Training: {exp_name}")
    print(f"Augmentation: {augment_args}")
    print('='*50)
    import os
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    model = YOLO("yolov8s.pt")  # 自动下载预训练权重

    start = time.time()
    results = model.train(
        project="runs",
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        device=0,           # GPU
        name=exp_name,
        exist_ok=True,
        verbose=False,
        # 增强参数
        **augment_args
    )
    elapsed = time.time() - start

    # 读取最终 mAP
    metrics = model.val(data=yaml_path, device=0, verbose=False)
    map50    = metrics.box.map50
    map5095  = metrics.box.map

    # 速度测试 (FPS)
    import torch
    dummy = torch.zeros(1, 3, 640, 640).to("cuda")
    # warmup
    for _ in range(10):
        model.predict(source=dummy, verbose=False)
    t0 = time.time()
    for _ in range(100):
        model.predict(source=dummy, verbose=False)
    fps = 100 / (time.time() - t0)

    result = {
        "exp":      exp_name,
        "map50":    round(map50, 4),
        "map5095":  round(map5095, 4),
        "fps":      round(fps, 1),
        "train_time_min": round(elapsed / 60, 1)
    }
    print(f"Result: {result}")
    return result

# ── Step 6: 消融实验（4组）────────────────────────────────────
# 关闭所有增强的基础参数
NO_AUG = dict(
    fliplr=0.0, flipud=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0,
    mosaic=0.0, mixup=0.0, copy_paste=0.0
)

experiments = [
    # (实验名, 增强参数)
    ("baseline_no_aug", NO_AUG),

    ("aug_basic", dict(
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=0.0, mixup=0.0
    )),

    ("aug_strong", dict(
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        mosaic=1.0, mixup=0.1
    )),

    ("aug_rotation", dict(
        fliplr=0.5,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=15.0, translate=0.1, scale=0.3,
        mosaic=0.0, mixup=0.0
    )),
]
experiments = experiments[:1]
all_results = []
for exp_name, aug_args in experiments:
    res = train_yolo(exp_name, aug_args, epochs=2)
    all_results.append(res)
    # 保存中间结果，防止断线丢失
    with open("/kaggle/working/yolo_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

# ── Step 7: 结果汇总表 ────────────────────────────────────────
import pandas as pd

df = pd.DataFrame(all_results)
df.columns = ["实验组", "mAP@0.5", "mAP@0.5:0.95", "FPS", "训练时间(min)"]
print("\n=== YOLOv8 消融实验结果 ===")
print(df.to_string(index=False))
df.to_csv("/kaggle/working/yolo_ablation_results.csv", index=False)

# ── Step 8: 可视化 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# mAP@0.5 对比
axes[0].bar(df["实验组"], df["mAP@0.5"], color=['#4472C4','#ED7D31','#A9D18E','#FF0000'])
axes[0].set_title("mAP@0.5 by Augmentation")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis='x', rotation=30)

# mAP@0.5:0.95 对比
axes[1].bar(df["实验组"], df["mAP@0.5:0.95"], color=['#4472C4','#ED7D31','#A9D18E','#FF0000'])
axes[1].set_title("mAP@0.5:0.95 by Augmentation")
axes[1].set_ylim(0, 1)
axes[1].tick_params(axis='x', rotation=30)

# FPS 对比
axes[2].bar(df["实验组"], df["FPS"], color=['#4472C4','#ED7D31','#A9D18E','#FF0000'])
axes[2].set_title("FPS by Augmentation")
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig("/kaggle/working/yolo_ablation_chart.png", dpi=150)
plt.show()
print("图表已保存: yolo_ablation_chart.png")

# ── Step 9: 保存最优模型路径（供后续使用）──────────────────────
best_exp = df.loc[df["mAP@0.5"].idxmax(), "实验组"]
best_model_path = f"/kaggle/working/runs/{best_exp}/weights/best.pt"
print(f"\n最优模型: {best_exp}")
print(f"模型路径: {best_model_path}")

with open("/kaggle/working/best_yolo_info.json", "w") as f:
    json.dump({"best_exp": best_exp, "best_model_path": best_model_path}, f)
