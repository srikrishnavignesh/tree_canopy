---
license: apache-2.0
title: tree_canopy
sdk: docker
emoji: 💻
colorFrom: blue
colorTo: yellow
short_description: tree segmentation
---
# 🌳 Tree Segmentation from Aerial Imagery (Solafune Platform)

## 📌 Overview
This project focuses on segmenting **individual trees** and **groups of trees** from high-resolution aerial imagery.  
The dataset comes from the **Solafune** platform and includes two segmentation targets:

- **`individual_tree`**  
- **`group_of_trees`**

Each image is **1204 × 1204 pixels**, captured at ground resolutions ranging from **20 cm to 80 cm** per pixel.  
This means one pixel corresponds to **0.2 m to 0.8 m** of ground area.

A key challenge in this dataset is the **extremely high object density**:  
➡️ **~4000 tree instances per image**, requiring robust segmentation techniques and efficient processing strategies.

---

## 🌱 Problem Description

Given aerial RGB images, the task is to **segment tree regions** into:

- Individual trees  
- Groups of trees  

Challenges include:
- Very high object density  
- Varying tree scales  
- Large variation in resolution (20 cm to 80 cm)  
- Shadows, illumination differences  
- Large image size (1204×1204)

---

## 🧪 Approaches Implemented

Two separate solutions were developed:

1. **YOLO11-Seg (with SAHI slicing)**  
2. **Mask2Former (with Swin-Tiny backbone + SAHI)**

Both solutions use **SAHI** (Slicing Aided Hyper Inference) to efficiently handle large images and to improve small-object performance.

---

# 🔷 1. Mask2Former Implementation

## 🛠 Model Details
- **Backbone:** Swin Transformer Tiny  
- **Architecture:** Mask2Former (query-based segmentation)  
- **Pretraining:** COCO-Segments  
- **Training Resolution:** **480 × 480** (sliced)  
- **Why sliced training?**  
  - Full 1204×1204 images have ~4000 tree instances  
  - Mask2Former uses **query-based detection**, where:
    - **#queries ≈ #objects**
  - Using thousands of queries significantly slows down cross-attention  
  - Training on smaller slices reduces both:
    - Compute cost  
    - GPU memory usage  
    - Attention complexity  

## 🌐 Data Augmentations
Standard Albumentations:
- Horizontal / vertical flip  
- Random rotate  
- Color jitter  
- Gaussian noise  

Custom augmentations:
- **Synthetic shadows** added to mimic aerial lighting variations  
- **Noise injection** for `individual_tree` class (high sample count)  
  - Avoids oversampling imbalance  
  - Preserves dataset richness  

## 📊 Performance
| Metric | Score |
|--------|--------|
| **mAP@75** | **0.40** |

Mask2Former performs significantly better because of its strong global reasoning and segmentation-focused architecture.

---

# 🔷 2. YOLO11-Seg Implementation

## 🛠 Model Details
- **Model:** YOLO11x-Seg  
- **Image Handling:** SAHI slicing  
- **Reason:**  
  - YOLO struggles with extremely dense small objects on large images  
  - Memory constraints prevent full-resolution training on 1204×1204  
  - SAHI allows per-slice detection, making training feasible  

## 📊 Performance
| Metric | Score |
|--------|--------|
| **mAP@75** | **0.11** |

YOLO11 underperforms in this scenario because:
- Dense small objects are harder for YOLO  
- Segmentation masks become noisy in crowded environments  
- Attention-free architecture limits instance separation  

---

# 🧠 Key Insights

- **High-density segmentation (>4000 objects)** requires query-based or transformer-style models.  
- **Mask2Former** handles dense tree structures far better than YOLO.  
- **SAHI** is essential for both training and inference:
  - Reduces GPU memory load  
  - Improves small-object detection  
  - Accelerates training on large images  
- Custom shadow and noise augmentations helped improve model robustness.  
- Lower-resolution images (20-80 cm) require scale-robust architectures.

---

# 🎉 Results Summary

| Model | Backbone | mAP@75 | Notes |
|-------|----------|--------|-------|
| **Mask2Former** | Swin-Tiny | **0.40** | Best performer, strong with dense segments |
| **YOLO11x-Seg** | CSP-based | **0.11** | Struggles with extremely dense small objects |

---

# 🚀 Future Improvements

- Use **Swin-Large** or **Swin-Base** for deeper feature representation  
- Train Mask2Former with **larger query capacity**  
- Add **class-specific weighting** to handle imbalance between `individual_tree` and `group_of_trees`