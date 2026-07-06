# Leaf Species Classification using Deep Learning

A structured, concept-driven deep learning project that classifies **176 leaf species** from images using PyTorch. Built as a learning project — every cell is documented to explain not just *what* the code does, but *why* each decision was made.

## Dataset

- **12,000 labeled images** across 176 leaf species (Kaggle competition dataset)
- Significant class imbalance: 258 images (most common) vs 18 images (rarest)
- All images: 224x224 RGB

## Experiments & Results

| Experiment | Approach | Best Val Accuracy |
|-----------|----------|-------------------|
| 1 | Custom CNN from scratch (30 epochs) | 59.62% |
| 2 | Transfer Learning + Fine-tuning (ResNet-50) | 88.75% |
| 2b | Enhanced head + AdamW scheduler | 72.12% |
| 3 | Differential LR Fine-tuning (best approach) | **89.75%** |

## Concepts Covered

- **Data Pipeline** — Custom Dataset class, stratified train/val split, DataLoaders
- **Preprocessing** — Resize, ToTensor, ImageNet normalization
- **Data Augmentation** — RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter
- **CNN Architecture** — Conv2d, MaxPool2d, BatchNorm, Dropout, Flatten, Linear layers
- **Training Loop** — Forward pass, CrossEntropyLoss, backpropagation, Adam optimizer
- **Transfer Learning** — Pretrained ResNet-50, frozen backbone, classifier head replacement
- **Fine-Tuning** — Full model unfreezing, differential learning rates per layer group
- **Evaluation** — Validation accuracy, visual prediction inspection, confidence scores

## Key Takeaways

1. Transfer learning (88.75%) dramatically outperforms training from scratch (59.62%) — ResNet-50 already understands visual features from 1.2M ImageNet images
2. Fine-tuning requires careful learning rate selection — too high destroys pretrained features
3. Differential learning rates (lower for early layers, higher for later layers) is the correct approach for fine-tuning pretrained models
4. Data augmentation is essential for imbalanced datasets with rare classes

## Tech Stack

- Python 3.12
- PyTorch 2.x
- torchvision
- scikit-learn
- pandas, matplotlib, PIL

## How to Run

1. Add the dataset on Kaggle: `midterm-exam-classification-leaves`
2. Open `leaf-classification.ipynb` in a Kaggle notebook
3. Set accelerator to **GPU T4 x2** with **Internet enabled**
4. Run cells in order from top to bottom
