# Blood Cell Cancer Detection

A CNN (ResNet18 transfer learning) that classifies blood-smear microscopy
images as **cancerous** (leukemic blast cells) or **non-cancerous** (normal
cells), trained on the public **C-NMC 2019** leukemia dataset
([`dwb2023/cnmc-leukemia-2019`](https://huggingface.co/datasets/dwb2023/cnmc-leukemia-2019)
on Hugging Face — 10,661 images).

> **Not a medical device.** This is a research/educational project. Predictions
> must never be used for real diagnosis — always consult a qualified
> pathologist.

## Setup

```bash
pip install -r requirements.txt
```

(If you don't have a modern NVIDIA GPU set up for CUDA, PyTorch will run on
CPU automatically — training is still feasible on a few hundred images per
class.)

## 1. Prepare the data

Downloads the dataset from Hugging Face (~6.5GB, cached after the first run)
and writes a balanced, resized, train/val/test split to `data/processed/`:

```bash
python src/prepare_data.py --per-class 600
```

- `--per-class N` caps how many images per class are used (default 600 →
  ~1,200 images total, split 70/15/15). Good for fast CPU iteration.
- `--per-class 0` uses every available image for a stronger but much slower
  training run.

## 2. Train

```bash
python src/train.py --epochs 8
```

- Fine-tunes a ResNet18 pretrained on ImageNet (backbone frozen by default,
  only the classification head trains — fast on CPU). Pass
  `--unfreeze-backbone` to fine-tune the whole network (slower, needs a GPU
  to be practical).
- Saves the best checkpoint (by validation accuracy) to
  `checkpoints/best_model.pt`, plus `checkpoints/history.json`.

## 3. Evaluate

```bash
python src/evaluate.py --split test
```

Prints precision/recall/F1 per class and saves a confusion matrix plot to
`checkpoints/confusion_matrix_test.png`.

## 4. Predict on a single image

```bash
python src/predict.py path/to/image.jpg
```

## 5. Web demo

```bash
python app/app.py
```

Opens a local Gradio app — upload an image, get a prediction with class
probabilities.

## Project layout

```
src/
  config.py       # paths, hyperparameters
  prepare_data.py # download HF dataset -> data/processed/
  dataset.py      # ImageFolder + transforms + dataloaders
  model.py        # ResNet18 transfer-learning model
  train.py        # training loop
  evaluate.py     # test-set metrics + confusion matrix
  predict.py      # single-image inference
app/
  app.py          # Gradio web demo
```

## Notes on the dataset

C-NMC 2019 (Classification of Normal vs Malignant Cells) is derived from the
ISBI 2019 challenge for detecting Acute Lymphoblastic Leukemia (ALL) from
peripheral blood smear microscopy images. `label == "cancer"` (`class_label
== "all"`) are malignant lymphoblasts; `label == "healthy"` (`class_label ==
"hem"`) are normal cells.
