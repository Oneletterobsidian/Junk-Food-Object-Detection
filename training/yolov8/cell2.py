import yaml, glob, time, json, os
import torch
from ultralytics import YOLO

os.environ["WANDB_DISABLED"] = "true"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

DATASET_PATH = "/kaggle/input/datasets/youssefahmed003/junk-food-object-detection-dataset-yolo-format/yolo_dataset"

label_files = glob.glob(f"{DATASET_PATH}/labels/train/*.txt")
class_ids = set()
for f in label_files[:500]:
    with open(f) as fp:
        for line in fp:
            class_ids.add(int(line.split()[0]))
nc = max(class_ids) + 1

names = ["burger","candy","chips","chocolate","donut","energy_drink","french_fries","fried_chicken","hot_dog","ice_cream","nachos","onion_rings","pizza","popcorn","soda","taco"]
if len(names) != nc:
    names = [str(i) for i in range(nc)]

yaml_path = "/kaggle/working/data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump({"path": DATASET_PATH, "train": "images/train", "val": "images/val", "test": "images/test", "nc": nc, "names": names}, f, default_flow_style=False)
print(f"✅ data.yaml 生成完成，Classes: {nc}")

model = YOLO("yolov8s.pt")
start = time.time()
model.train(
    project="runs", data=yaml_path, epochs=50,
    imgsz=640, batch=16, device=0, name="aug_basic", exist_ok=True, verbose=False,
    fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=0.0, mixup=0.0,
)
elapsed = time.time() - start
print(f"✅ 训练完成，用时 {elapsed/60:.1f} 分钟")

weight_path = "/kaggle/working/runs/aug_basic/weights/best.pt"
if os.path.exists(weight_path):
    print(f"✅ 找到模型: {weight_path} ({os.path.getsize(weight_path)/1024/1024:.1f} MB)")
else:
    print("❌ 找不到权重文件")
    for root, dirs, files in os.walk('/kaggle/working/'):
        for f in files:
            if f.endswith('.pt'):
                print(f"找到: {os.path.join(root, f)}")