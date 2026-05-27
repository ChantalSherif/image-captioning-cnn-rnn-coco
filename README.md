# Image Captioning — CNN + RNN on MS COCO

An end-to-end deep learning pipeline that generates natural language captions for images. A CNN acts as the visual encoder; an RNN decoder then produces a caption word-by-word, conditioned on what the CNN saw.

> *"A boy in a red jacket is flying a kite in the park."*
> — the kind of sentence this model learns to generate from raw pixels.

---

## Overview

Image captioning sits at the intersection of computer vision and natural language processing. This project solves it by connecting two neural network families into a single trainable pipeline:

| Component | Role | Architecture |
|---|---|---|
| CNN Encoder | Extracts a compact visual representation | ResNet / VGG (pretrained) |
| RNN Decoder | Generates captions token-by-token | LSTM / GRU |
| Auxiliary Head | Image classification for sharper visual features | Dense + Softmax |

Trained and evaluated on a subset (≥50%) of the **MS COCO** benchmark dataset.

---

## Dataset

**[MS COCO — Common Objects in Context](https://www.kaggle.com/datasets/hariwh0/ms-coco-dataset)**

Over 120,000 everyday images, each paired with five independent human-written captions. The five-captions-per-image design reflects the fact that there is no single correct description of a scene.

| Split | Images | Captions |
|---|---|---|
| Train | (from 45,530 selected unique images)	 | 18,055 |
| Validation | 	(from 45,530 selected unique images) | 4,514 |
| Test | not used | not used |

---

## Project Structure

```
├── AML2.ipynb     # Full pipeline notebook
└── README.md
```

---

## Pipeline

### 1. Data Preprocessing
- **Caption cleaning & tokenization** — lowercasing, punctuation removal, `` / `` tokens
- **Vocabulary construction** — frequency-filtered word-to-index mapping
- **Sequence preparation** — padded caption tensors aligned to fixed max length
- **Image feature extraction** — CNN forward pass, feature vectors cached for training efficiency

### 2. Model Architecture

```
Image
  │
  ▼
CNN Encoder (ResNet/VGG, pretrained)
  │  → fixed-length feature vector
  ▼
RNN Decoder (LSTM/GRU)
  │  → hidden state initialized from CNN features
  │  → word embeddings fed step-by-step
  ▼
Dense + Softmax → predicted word at each timestep
  │
  ▼
Caption: ["A", "boy", "in", "a", "red", "jacket", ...]
```

- **Embedding layer** — learned word embeddings (or GloVe-initialized)
- **Auxiliary classification head** — extra supervision signal during training to sharpen visual features
- **Teacher forcing** during training; **greedy / beam search** at inference

### 3. Training
- Loss: cross-entropy over caption tokens
- Optimizer: Adam with learning rate scheduling
- Training & validation loss curves plotted per epoch

### 4. Evaluation
- Qualitative: sample images with generated vs. ground-truth captions
- Quantitative: BLEU score on the test set

---

## Sample Results

| Image | Image ID | Generated Caption |
|---|---|---|
| ![Truck in desert](http://images.cocodataset.org/val2017/000000403487.jpg) | 403487 | a truck is parked on a dirt road in a dry desert landscape. |
| ![Woman with umbrella](http://images.cocodataset.org/val2017/000000530867.jpg) | 530867 | a woman is holding a colorful umbrella while standing outdoors. |
| ![Man on motorcycle](http://images.cocodataset.org/val2017/000000154699.jpg) | 154699 | a man wearing a red shirt is riding a motorcycle on the street. |

> Populate after running the notebook end-to-end.

---

## Setup & Requirements

```bash
pip install torch torchvision transformers nltk matplotlib pillow pycocotools
```

Run on **Google Colab with GPU** (T4 or better strongly recommended — dataset and model are large).

---

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `MS COCO API` · `NLTK` · `Matplotlib` · `Pillow`

---

## Course Info

Advanced Machine Learning — Spring 2026
German International University of Applied Sciences
