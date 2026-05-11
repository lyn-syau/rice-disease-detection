# DCA-YOLO: Lightweight Visual Detection Framework for Rice Leaf Disease Identification

This repository contains the official implementation of **DCA-YOLO**, associated with the manuscript:

> **Lightweight Visual Detection Framework for Real-Time Rice Leaf Disease Identification on Edge Mobile Robots**
> Xu Yan, Liu Yinan, Meng Xiangchen, Yuan Qing, Wang Dazhong, Wu Liyan, Yue Xiang, Feng Longlong, Liu Cuihong
> *Submitted to The Visual Computer, Springer*

If you use this code or model, please cite our paper (citation info will be updated upon acceptance).

---

## Overview

DCA-YOLO is a lightweight object detection model designed for real-time rice leaf disease detection on resource-constrained edge mobile robots. Built upon YOLO11n, it introduces three core improvements:

- **C3k2-DICN**: Dynamic Inception Mixer for adaptive multi-scale feature extraction
- **CSDH**: Cross-Scale Shared Detection Head to reduce detection head parameter redundancy
- **ADown**: Adaptive Dual-path Downsampling to preserve disease-discriminative features

| Model | GFLOPs | Params | mAP@0.5 | mAP@0.5:0.95 | FPS (Jetson TX2) |
|---|---|---|---|---|---|
| YOLO11n (baseline) | 6.3 | 2.6M | 87.18% | 45.16% | 45.3 |
| **DCA-YOLO (ours)** | **4.0** | **1.7M** | **88.42%** | **45.82%** | **48.7** |

---

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:
- Python 3.10
- PyTorch >= 2.0.0
- Ultralytics >= 8.0.0
- OpenCV >= 4.8.0

---

## Quick Start

### Single image
```bash
python inference.py --source path/to/image.jpg
```

### Folder of images
```bash
python inference.py --source path/to/images/
```

### Adjust confidence threshold
```bash
python inference.py --source image.jpg --conf 0.5
```

### Run on CPU explicitly
```bash
python inference.py --source image.jpg --device cpu
```

All results are saved to the `results/` folder by default.

---

## Model Weights

The pretrained model `best.pt` is included in this repository, trained on our self-constructed rice disease dataset covering three categories:

| Class | Disease |
|---|---|
| 0 | Rice Blast (*Magnaporthe oryzae*) |
| 1 | Brown Spot (*Bipolaris oryzae*) |
| 2 | Bacterial Blight (*Xanthomonas oryzae* pv. *oryzae*) |

---

## Dataset

- **Self-constructed dataset**: 4,622 annotated images (available from the corresponding author upon reasonable request: cuihongliu77@syau.edu.cn)
- **Public benchmark**: Roboflow rice disease dataset — https://universe.roboflow.com/pest-i83ul/rice-leaf-disease-weadf

---

## Edge Deployment

DCA-YOLO has been validated on **NVIDIA Jetson TX2** using TensorRT FP16 acceleration:

- Inference latency: **20.54 ms**
- Throughput: **48.7 FPS**
- Deployment pipeline: PyTorch → ONNX → TensorRT (JetPack 4.6.2, TensorRT 8.2.1)

---

## License

This project is released for academic research use only.

---

## Contact

Corresponding author: **Liu Cuihong**
Email: cuihongliu77@syau.edu.cn
College of Engineering, Shenyang Agricultural University, Shenyang 110866, China
