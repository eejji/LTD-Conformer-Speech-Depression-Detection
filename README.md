# LTD-Conformer: Speech Depression Detection with Speaking and Listening Perspectives
<div align="center">

[![IEEE](https://img.shields.io/badge/IEEE-Published-blue?logo=ieee)](https://ieeexplore.ieee.org/abstract/document/11058748)
[![Conference](https://img.shields.io/badge/CBMS-2025-orange)](https://www.cbms2025.org/)

**Published at the 2025 IEEE 38th International Symposium on Computer-Based Medical Systems (CBMS)**   
June 18–20, 2025 · Madrid, Spain
DOI: [10.1109/CBMS65348.2025.00087](https://ieeexplore.ieee.org/abstract/document/11058748)

</div>

## Overview

This project presents a deep learning approach to **speech-based depression detection**, extending the Conformer architecture to jointly model the **local, global, and sequential** patterns of audio signals through a novel **Long-Term Dilated Conformer (LTD-Conformer)**.

Most prior work focused on either local acoustic features (CNN-based) or global context (Transformer-based) — but not the sequential dynamics that characterize depressed speech, such as slowed tempo, intonation drift, and frequent pauses. This model adds a dedicated Long-Term Module on top of a dilated Conformer backbone, and fuses **listening features (Mel-spectrogram)** with **speaking features (HuBERT)** to capture both perspectives of the audio signal.

## Key Contribution

> **Problem:** Conformer captures local and global patterns but ignores long-term sequential dynamics; single-perspective audio features (only listening *or* only speaking) miss complementary acoustic cues of depression.
> **Solution:** A dual-path architecture that (1) widens the Conformer's receptive field via **dilated depthwise convolution**, (2) adds a parallel **GRU-based Long-Term Module** for sequential modeling, and (3) fuses Mel-spectrogram and HuBERT features at the input to leverage both listening and speaking perspectives.

**Keywords:** Audio · Conformer · Depression · HuBERT · Long-Term Dilated-Conformer · Mel-spectrogram

| Item | Detail |
|------|--------|
| **Task** | 2-class depression classification (Non-Depression / Depression) |
| **Input** | Audio signal (Mel-spectrogram + HuBERT) |
| **Dataset** | [DAIC-WOZ](https://dcapswoz.ict.usc.edu/) |
| **Evaluation** | Train / Validation / Test split (169 / 43 / 54) |
| **Accuracy** | **87.04%** |
| **F1-Score** | **0.87** |

---

#### **Abstract:**
Depression is a pervasive mental health problem worldwide and requires quick and accurate diagnosis. Recently, machine learning and deep learning techniques have been actively applied to depression diagnosis research, especially as audio signals are attracting attention as non-invasive and cost-effective modality. This study proposes the Long-Term Dilated Conformer (LTD-Conformer), an extension of the existing Conformer model designed to utilize audio signals for more accurate depression detection. The LTD-Conformer employs dilated depthwise convolution to achieve a wide receptive field and integrates a Long-Term Module to capture sequential information in audio features. This model comprehensively captures and analyzes the local, global, and sequential patterns in audio signals. In addition, we combined listening features (Mel-spectrogram) and speaking features (HuBERT) to effectively analyze both perspectives of audio signal. The experiment was conducted using DAIC-WOZ dataset, and the LTD-Conformer model achieved an accuracy of 87.04% and an F1-score of 0.87, demonstrating a 4% improvement in accuracy and a 0.04 increase in the F1 score compared to the existing Conformer model. This study presents the possibility that the audio signal-based depression LTD-Conformer model can be effectively applied to depression diagnosis and will develop into a strong audio-based depression diagnosis model in the future.

##### **Keywords:** Audio, [Conformer](https://arxiv.org/abs/2005.08100), Depression, [HuBERT](https://arxiv.org/abs/2106.07447), Long-Term Dilated-Conformer, Mel-spectrogram


## Pipeline

![Workflow](https://github.com/eejji/LTD-Conformer-Speech-Depression-Detection/blob/main/Figure/Workflow.png)

---

## Dataset: DAIC-WOZ

The [DAIC-WOZ dataset](https://dcapswoz.ict.usc.edu/) (Distress Analysis Interview Corpus — Wizard of Oz) is an interview-based multimodal corpus for diagnosing psychological distress including depression, anxiety, and PTSD. Interviews were conducted between participants and a virtual interviewer (Ellie), lasting from 7 to 33 minutes (average ~16 min).

Class labels are assigned using the **PHQ-8 score**:

| Class | PHQ-8 Score | Participants |
|-------|-------------|--------------|
| **Non-Depression** | 0 – 9 | 133 |
| **Depression** | ≥ 10 | 56 |
| **Total** | — | 189 (82 M / 107 F) |

Audio was recorded at **16 kHz sampling frequency**. In this study, only the participant's speech was extracted from the dialogue using the provided transcripts, removing all interviewer turns.

---

## Preprocessing

### Data Augmentation — Pitch Shifting Only

Time-shifting and time-stretching were **deliberately excluded** because depression-relevant cues (stammering, slowed articulation, hesitation) live in the temporal structure. Pitch shifting alters tone without distorting timing.

1. **Pitch shift range** — randomly sampled from **[−0.55, −0.2] ∪ [0.2, 0.55]** semitones
2. **Applied to** — all 56 depressed participants + 21 randomly selected → **77 augmented samples**

### Feature Extraction

Each audio signal is divided using a **4-second window with 1-second overlap**, with each window further segmented into **25 ms frames with 10 ms stride** for precise analysis.

| Feature | Perspective | Dimension | Description |
|---------|-------------|-----------|-------------|
| **Mel-spectrogram** | Listening | 80 | Reflects human auditory perception via Mel-scale filter banks |
| **HuBERT** | Speaking | 768 | Self-supervised Transformer feature capturing utterance-level patterns |
| **Concatenated** | Multi-perspective | **848** | Input to the model |

Per-frame features are averaged within each window. The number of windows per subject is standardized to **151** (dataset average), zero-padded if shorter.

---

## Model Architecture

![Model](https://github.com/eejji/LTD-Conformer-Speech-Depression-Detection/blob/main/Figure/Model.png)

```
                              ┌──► D-Conformer Block (×N)  ─┐
Input (B, 151, 848) ──► Subsample ─┤                              ├──► ⊙ ──► Linear ──► Class
                              └──► Long-Term Module (GRU)   ─┘
```

**Convolutional Subsampling**
- Two 2D conv layers reduce the time dimension by 4× → (B, T/4, F)

**D-Conformer Block (Macaron-style)**
- ½ Feed-Forward Module → Multi-Head Self-Attention (with Transformer-XL relative positional encoding) → **Dilated Depthwise Convolution** → ½ Feed-Forward Module → LayerNorm
- Dilation = 2, kernel size = 33 → expanded receptive field over standard Conformer

**Long-Term Module**
- GRU + LayerNorm, applied in parallel to the D-Conformer branch
- Captures sequential dependencies missed by attention alone

**Fusion**
- Point-wise multiplication of `X_D ⊙ X_LT`, then a linear projection to class logits

### Mathematical Formulation

```
x̃ᵢ  = xᵢ + ½ · FFN(xᵢ)
x'ᵢ  = x̃ᵢ + MHSA(x̃ᵢ)
x''ᵢ = x'ᵢ + D_Conv(x'ᵢ)
X_D  = LayerNorm(x''ᵢ + ½ · FFN(x''ᵢ))
X_LT = LayerNorm(GRU(xᵢ))
Y    = Linear(X_D ⊙ X_LT)
```

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Encoder dim | 128 |
| Encoder layers | 1 |
| Attention heads | 2 |
| Dilation | 2 |
| Conv kernel size | 33 |
| Long-Term layers | 1 |
| Dropout | 0.2 |
| FFN expansion factor | 2 |
| Conv expansion factor | 2 |
| Optimizer | Adam (β₁=0.9, β₂=0.98, ε=1e-9) |
| Learning rate | 1e-5 |
| Batch size | 64 |
| Max epochs | 300 |
| Early stopping | patience=50 |
| Hardware | NVIDIA RTX 3060 Ti · CUDA 12.4 · PyTorch 2.4.0 |

---

## Results

Train / Validation / Test split on the DAIC-WOZ database:

### Main Comparison

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| Conformer (baseline) | 83.00% | 0.83 |
| **LTD-Conformer (proposed)** | **87.04%** | **0.87** |

![Confusion_matrix](https://github.com/eejji/LTD-Conformer-Speech-Depression-Detection/blob/main/Figure/Confusion_matrix.png)

Confusion matrix comparison between (A) the baseline Conformer and (B) the proposed LTD-Conformer.   
The depression-class recall improves from **0.74 → 0.81**, indicating that the Long-Term Module and dilated convolution help the model identify depressed patients more reliably.


### Ablation — Listening vs Speaking Features

| Model | Feature | Accuracy | F1-Score |
|-------|---------|----------|----------|
| Conformer | Listening only (Mel) | 69.00% | 0.68 |
| Conformer | Speaking only (HuBERT) | 81.00% | 0.81 |
| Conformer | Listening + Speaking | 83.00% | 0.83 |
| LTD-Conformer | Listening only (Mel) | 72.22% | 0.72 |
| LTD-Conformer | Speaking only (HuBERT) | 85.19% | 0.85 |
| **LTD-Conformer** | **Listening + Speaking** | **87.04%** | **0.87** |

### Comparison with Prior Work

| Method | Feature | Accuracy | F1-Score |
|--------|---------|----------|----------|
| Zhou et al. (2023) | MFCC + eGeMAPs + BoAW + Spectrogram | — | 0.77 |
| Du et al. (2023) | LPC + MFCC | 77.10% | 0.746 |
| Han et al. (2023) | vq-wav2vec | 80.00% | 0.80 |
| Rejaibi et al. (2022) | MFCC | 76.27% | 0.65 |
| Conformer | Mel + HuBERT | 83.00% | 0.83 |
| **LTD-Conformer (ours)** | **Mel + HuBERT** | **87.04%** | **0.87** |

- **+4.04% accuracy** and **+0.04 F1** over the vanilla Conformer
- **+10% accuracy** over Du et al.'s CNN-LSTM speech-chain model
- Depression-class recall improved from **0.74 → 0.83** (+9%)

Evaluation metrics reported: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

---

## Project Structure

```
.
├── Model.py               # LTD-Conformer architecture (D-Conformer Block + Long-Term Module)
├── requirements.txt       
├── Figure/                # Workflow and model architecture diagrams
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Dataset

Download the [DAIC-WOZ dataset](https://dcapswoz.ict.usc.edu/) (access requires application). Use the provided transcripts to extract only the participant's speech turns.

### 2. Extract Features

Extract Mel-spectrogram (80-d) and HuBERT (768-d) features per window (4 s window, 1 s overlap, 25 ms / 10 ms framing), then concatenate to obtain 848-d input vectors.

### 3. Train the Model

---

## Citation

```bibtex
@inproceedings{lee2025ltdconformer,
  title     = {LTD-Conformer: Speech Depression Detection with Speaking and Listening Perspectives},
  author    = {Lee, Jihun and Hong, Jisun and Choi, Daegil and Jung, Jaehyo},
  booktitle = {2025 IEEE 38th International Symposium on Computer-Based Medical Systems (CBMS)},
  pages     = {399--404},
  year      = {2025},
  address   = {Madrid, Spain},
  organization = {IEEE},
  doi       = {10.1109/CBMS65348.2025.00087}
}
```
