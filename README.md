"# Junk-Food-Object-Detection" 
# Junk Food Object Detection

A comparative study of deep learning-based object detection systems for junk food identification. This project goes beyond simple single-model transfer learning by incorporating multi-model comparison, augmentation analysis, and generalization testing.

**Course:** EECE7370 Advanced Computer Vision — Northeastern University  
**Author:** Xinyao Wan  

---

## Overview

This project evaluates three state-of-the-art object detection architectures — **YOLOv8**, **Faster R-CNN**, and **DETR** — on a 35-class junk food detection dataset (~10,000 images). An augmentation ablation study was first conducted on YOLOv8 to identify the optimal training configuration, followed by a standardized multi-model comparison.

---

## Dataset

- **Source:** [Junk Food Object Detection Dataset (YOLO Format)](https://www.kaggle.com/datasets/youssefahmed003/junk-food-object-detection-dataset-yolo-format) by youssefahmed003 on Kaggle
- **Size:** ~10,400 images across 35 food categories
- **Format:** YOLO bounding box annotations
- **Split:** 80% train / 10% val / 10% test

| Split | Images |
|-------|--------|
| Train | 8,320 |
| Validation | 1,040 |
| Test | 1,040 |

**Food categories include:** burger, hotdog, chips, pizza, ice cream, waffles, nachos, cookies, French fry, cheesecake, crispy chicken, sandwich, noodles, muffin, biryani, kabab, and more.

---

## Models

| Model | Type | Backbone | Epochs | Batch Size |
|-------|------|----------|--------|------------|
| YOLOv8s | One-stage | CSPDarknet | 50 | 16 |
| Faster R-CNN | Two-stage | ResNet-50 FPN | 20 | 4 |
| DETR | Transformer | ResNet-50 | 25 | 4 |

All models were initialized with ImageNet-pretrained weights and trained on Kaggle with an **NVIDIA Tesla P100-PCIE-16GB GPU**.

---

## Augmentation Ablation (YOLOv8)

Four controlled experiments were conducted with different augmentation configurations:

| Experiment | Configuration |
|------------|---------------|
| baseline_no_aug | No augmentation |
| aug_basic | Horizontal flip (p=0.5) + HSV color jitter |
| aug_strong | aug_basic + Mosaic (p=1.0) + MixUp (p=0.1) |
| aug_rotation | aug_basic + Rotation (±15°) + Translation + Scale |

---

## Results

### Augmentation Ablation

| Experiment | mAP@0.5 | mAP@0.5:0.95 | FPS | Training Time (min) |
|------------|---------|---------------|-----|----------------------|
| baseline_no_aug | 0.8797 | 0.6611 | 92.3 | 137.8 |
| **aug_basic** | **0.9259** | **0.7060** | 84.3 | 137.2 |
| aug_strong | 0.9237 | 0.7047 | 86.7 | 138.4 |
| aug_rotation | 0.9256 | 0.6878 | 89.3 | 138.4 |

### Multi-Model Comparison

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS | Training Time (min) |
|-------|---------|---------------|-----|----------------------|
| **YOLOv8 (aug_basic)** | **0.9259** | **0.7060** | **84.3** | 138 |
| Faster R-CNN | 0.8868 | N/A | 8.7 | 387.5 |
| DETR | 0.0116 | 0.0052 | 8.7 | 325 |

---

## Key Findings

- **YOLOv8 with basic augmentation** achieved the best overall performance: mAP@0.5 of 0.9259 at 84.3 FPS.
- **Basic augmentation** (flip + HSV jitter) outperforms more aggressive strategies. Mosaic and rotation may introduce spatial distortions inconsistent with natural food appearance.
- **Faster R-CNN** achieved competitive accuracy (mAP@0.5: 0.8868) but at ~9.7× slower inference speed (8.7 FPS).
- **DETR's low performance** (mAP@0.5: 0.0116) is due to insufficient training epochs — the original paper requires 300 epochs for convergence, while this study was limited to 25 epochs due to GPU quota constraints.

---

## Project Structure

```
├── dataset_generator.py       # Dataset preparation and splitting
├── feature_encoding.py        # Feature encoding utilities
├── linear_regression.py       # Baseline regression model
├── mc_simulation.py           # Monte Carlo simulation
├── results_logger.py          # Logging experiment results
├── yolov8/                    # YOLOv8 training and evaluation
│   ├── junkfood.ipynb
│   ├── demo.py
│   ├── best.pt                # Best YOLOv8 model weights
│   └── ...
```

---

## Demo

A real-time interactive inference interface was built with **Gradio**. Upload any food image to receive bounding box predictions with class labels and confidence scores.

```bash
python yolov8/demo.py
```

---

## References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- youssefahmed003. (2024). [Junk Food Object Detection Dataset](https://www.kaggle.com/datasets/youssefahmed003/junk-food-object-detection-dataset-yolo-format). Kaggle.
