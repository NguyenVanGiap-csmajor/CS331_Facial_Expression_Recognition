# TransFER - Facial Expression Recognition with Transformers (FER+ Re-implementation)

> PyTorch re-implementation of:  
> **TransFER: Learning Relation-aware Facial Expression Representations with Transformers**  
> [Paper link](https://arxiv.org/pdf/2108.11116)

---

## Introduction

This project is a re-implementation of the **TransFER** model for the facial expression recognition task. The model is fully re-trained on the **FERPlus** dataset. **TransFER** can learn context-aware local representations. It consists of three main components: **Multi Attention Dropping (MAD)**, **ViT-FER**, and **Multi-head Self-Attention Dropping (MSAD)**.  
![Model architecture](images/transfer_architecture.png)

**STEM CNN**: Used to extract feature maps from facial images, based on the IR-50 architecture pre-trained on Ms-Celeb-1M. [Original link](https://drive.google.com/drive/folders/1omzvXV_djVIW2A7I09DWMe9JR-9o_MYh)

---

## Directory Structure

```
TransFER/
├── code/
│ ├── transfer_model.py # TransFER model
│ ├── train.py # Training script
│ └── fer_realtime_demo.py # Real-time demo
│
├── dataset/
│ ├── FER_Image/ # Training images
│ └── Label/ # Training labels
│
├── images/
└── README.md
```

---

## Dataset

FER+ is an extended annotation version of FER2013, where each image is labeled by 10 annotators instead of a single one. This allows representing emotions as probability distributions or multi-labels instead of just one class.

Below are some examples from FER and FER+ (FER on top, FER+ on bottom):  
![FERvsFER+](images/FER+vsFER.png)

Here we have downloaded the original images and stored the new labels in the **FER_Image** and **Label** folders. You can also access the original dataset here: [FERPlus dataset](https://github.com/microsoft/FERPlus/tree/master)

---

## Results

The best performance achieved on the test set was **84.95% accuracy**.

We trained the model and saved the best weights based on validation performance. You can download them here: [Google Drive](https://drive.google.com/drive/u/4/folders/1DuqNhhV9suTmlCnYC9a5fAZ2cR_1NVNy)

To run the demo, place the weight files together with the `fer_realtime_demo.py`.

---

## Notice

This repository is a re-implementation based on the paper:  
**TransFER: Learning Relation-aware Facial Expression Representations with Transformers**.  

The original authors did not release their code or pre-processed dataset.  

This project re-creates the work following the ideas and methodology from the paper.  

It is intended for educational and research purposes. You may use this project as a reference or read the original paper for detailed insights.  
[Reference link](https://drive.google.com/drive/u/4/folders/1DuqNhhV9suTmlCnYC9a5fAZ2cR_1NVNy)

---

## Members

- Nguyễn Văn Giáp
- Đào Minh Hải
