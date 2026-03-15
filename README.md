# Leaf-Pred

**Plant disease detection using vision models on the PlantVillage dataset.**

A comparative study of four deep learning architectures — from a custom CNN trained from scratch to state-of-the-art pretrained models — evaluated on 21 plant disease classes.

---

## Results

| Model | Test Accuracy | Macro F1 | Parameters |
|---|---|---|---|
| Custom CNN | 96.81% | 0.9550 | 4.9M |
| EfficientNet-B2 | 98.92% | 0.9866 | 8.4M |
| ViT-B/16 | 99.50% | 0.9923 | 86.2M |
| **ResNet50** | **99.79%** | **0.9967** | **24.6M** |

ResNet50 achieves the best overall accuracy and macro F1. EfficientNet-B2 offers the best accuracy-per-parameter trade-off. The consistently hardest class across all models is **Cercospora leaf spot**.

---

## Dataset

[PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — color images, 21-class subset.

- ~54,000 images across 21 disease categories
- Split: 70% train / 15% val / 15% test (stratified, seed=42)
- Covers apple, corn, grape, potato, strawberry, tomato, and more

The dataset is not included in this repository. Download it from Kaggle and place it at `data/raw/color/`.

---

## Models

### 04 — Custom CNN
Trained from scratch. Five convolutional blocks (32→64→128→256→512 filters), each with double Conv→BN→ReLU→MaxPool→Dropout, followed by Global Average Pooling into a 512→256→N classifier head.

### 05 — ResNet50
Pretrained on ImageNet. Two-phase fine-tuning: phase 1 trains the new classifier head only; phase 2 unfreezes `layer3` and `layer4` with differential learning rates.

### 06 — EfficientNet-B2
Pretrained on ImageNet. Same two-phase strategy as ResNet50. Phase 2 unfreezes the last three feature blocks with differential learning rates.

### 07 — ViT-B/16
Vision Transformer pretrained on ImageNet. Two-phase fine-tuning with phase 2 unfreezing the last four encoder layers. Also produces CLS-token attention maps showing what the model attends to when classifying a leaf.

---

## Project Structure

```
leaf-pred/
│
├── README.md
│
├── code/
│   ├── 04_model_custom_cnn.py
│   ├── 05_model_resnet50.py
│   ├── 06_model_efficientnet.py
│   ├── 07_model_vit.py
│   └── 08_results_comparison.ipynb
│
└── .gitignore
```

Scripts are numbered in execution order. Each is self-contained — they all read from the same preprocessed data and write their results independently.

---

## Requirements

```
torch>=2.2.0
torchvision>=0.17.0
numpy
pandas
scikit-learn
matplotlib
seaborn
tqdm
Pillow
```

---

## Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — Hughes & Salathé, 2015
- Pretrained weights from `torchvision.models` (ImageNet)
