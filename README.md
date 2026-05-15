# DCA-YOLO: Lightweight Visual Detection Framework for Rice Leaf Disease

<p align="center">
  <img src="assets/example_result.jpg" width="700" alt="DCA-YOLO detection example"/>
</p>

<p align="center">
  <a href="https://doi.org/10.1007/s00371-025-XXXXX"><img src="https://img.shields.io/badge/Paper-The Visual Computer-blue" alt="Paper"/></a>
  <a href="https://doi.org/10.5281/zenodo.XXXXXXX"><img src="https://img.shields.io/badge/DOI-Zenodo-lightblue" alt="Zenodo DOI"/></a>
  <img src="https://img.shields.io/badge/Platform-Jetson TX2-green" alt="Platform"/>
  <img src="https://img.shields.io/badge/Framework-Ultralytics-orange" alt="Framework"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

> **Note for reviewers and readers**: This repository is the official code release directly associated with the manuscript currently under review at **The Visual Computer** (Springer). If you use this code or the pretrained weights, please cite the paper below.

This repository provides the **inference script and pretrained model weights** for **DCA-YOLO**, as described in:

> **Lightweight Visual Detection Framework for Real-Time Rice Leaf Disease Identification on Edge Mobile Robots**
> Xu Yan, Liu Yinan, Meng Xiangchen, Yuan Qing, Wang Dazhong, Wu Liyan, Yue Xiang, Feng Longlong, Liu Cuihong
> *The Visual Computer*, 2026. [DOI: 10.1007/s00371-025-XXXXX]

---

## Highlights

| Metric | YOLO11n (baseline) | **DCA-YOLO (Ours)** |
|---|---|---|
| mAP@0.5 | 87.18% | **88.42%** |
| mAP@0.5:0.95 | 45.16% | **45.82%** |
| Params | 2.6 M | **1.7 M (↓34.6%)** |
| GFLOPs | 6.3 | **4.0 (↓36.5%)** |
| FPS on Jetson TX2 (TRT FP16) | 45.4 | **28.3** ✅ |

---

## Key Algorithmic Contributions

### C3k2-DICN — Dynamic Hybrid Convolution Module
Replaces the standard bottleneck in C3k2 with a **Dynamic Inception Mixer (DIM)** that deploys three depthwise separable convolution branches in parallel (3×3 square, 1×k horizontal strip, k×1 vertical strip). A data-driven **Dynamic Kernel Weights (DKW)** generator produces content-adaptive attention weights via global average pooling and Softmax normalization, enabling each input image to dynamically select the most discriminative receptive field configuration for multi-scale lesion feature extraction.

### CSDH — Cross-Scale Shared Detection Head
Replaces the independently parameterized per-scale prediction branches of the standard YOLO detection head with a **cross-scale parameter-sharing** design. A per-scale 3×3 projection layer first maps features to a unified hidden dimension, after which a single shared refinement module (depthwise + pointwise convolution) and shared regression/classification heads are applied across all detection scales. This structural constraint encourages scale-agnostic feature learning while substantially reducing detection head parameter count.

### ADown — Adaptive Dual-path Downsampling
Decouples spatial resolution compression into two complementary parallel paths — one optimized for local texture detail, the other for structural context — before fusing via channel concatenation. This dual-path design preserves disease-discriminative spatial cues that single-strategy downsampling (e.g., strided convolution or max pooling) would discard, improving multi-scale feature pyramid quality at no additional inference cost.

---

## Repository Structure

```
rice-disease-detection/
├── model/
│   └── DCA-YOLO.yaml      # Model architecture definition
├── weights/
│   └── best.pt            # Pretrained weights (~3.7 MB)
├── inference.py           # Run inference on images
├── requirements.txt
└── README.md
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
```
ultralytics>=8.0
torch>=2.0
```

---

## Inference

### Python

```python
from ultralytics import YOLO

model = YOLO("weights/best.pt")
results = model.predict("your_image.jpg", imgsz=640, conf=0.25)
results[0].save("output.jpg")
```

### Command-line

```bash
python inference.py --source your_image.jpg --weights weights/best.pt --conf 0.25
```

---

## Disease Classes

| ID | Class | Pathogen |
|---|---|---|
| 0 | Rice Blast | *Magnaporthe oryzae* |
| 1 | Brown Spot | *Bipolaris oryzae* |
| 2 | Bacterial Blight | *Xanthomonas oryzae* pv. *oryzae* |

---

## Citation

If you use this code, pretrained weights, or find this work helpful, **please cite the following manuscript**:

```bibtex
@article{yan2025dcayolo,
  title     = {Lightweight Visual Detection Framework for Real-Time Rice Leaf Disease
               Identification on Edge Mobile Robots},
  author    = {Yan, Xu and Liu, Yinan and Meng, Xiangchen and Yuan, Qing and
               Wang, Dazhong and Wu, Liyan and Yue, Xiang and
               Feng, Longlong and Liu, Cuihong},
  journal   = {The Visual Computer},
  year      = {2026},
  doi       = {10.1007/s00371-025-XXXXX},
  publisher = {Springer}
}
```

---

## Contact

**Liu Cuihong** — cuihongliu77@syau.edu.cn
College of Engineering, Shenyang Agricultural University, Shenyang 110866, China
