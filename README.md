# LTD-Conformer: Speech Depression Detection with Speaking and Listening Perspectives

## Conference / Publication
**Conference:** 38th IEEE International Symposium on Computer-Based Medical Systems (CBMS)   
**Date:** 18-20 June 2025   
**Location:** Madrid, Spain   
**DOI:** [10.1109/CBMS65348.2025.00087](https://ieeexplore.ieee.org/abstract/document/11058748)   

## Task- At A Glance:
The LTD-Conformer model for predicting depression detection by fusing the listening and speaking perspectives of audio signals
1. __Task__: Depression detection
2. __Input__: Audio signal
3. __Output__:  2 Class (Non-Depression, Depression)
4. __Database__: (1) [DAIC-WOZ](https://dcapswoz.ict.usc.edu/)
5. __Preprocessing__: Data Augmentation (Pitch shifting), Feature Extraction(Mel-spectrogram, HuBERT)
6. __Fusion method__: Concatenate (Early Fusion)
7. __Result__: Accuracy: **87.04 %**, F1-score **0.87** (3 class)

## Abstract
Depression is a pervasive mental health problem worldwide and requires quick and accurate diagnosis. Recently, machine learning and deep learning techniques have been actively applied to depression diagnosis research, especially as audio signals are attracting attention as non-invasive and cost effective modality. This study proposes the Long-Term Dilated Conformer (LTD-Conformer), an extension of the existing Conformer model designed to utilize audio signals for more accurate depression detection. The LTD-Conformer employs dilated depthwise convolution to achieve a wide receptive field and integrates a Long-Term Module to capture sequential information in audio features. This model comprehensively captures and analyzes the local, global, and sequential patterns in audio signals.  In addition, we combined listening features (Mel-spectrogram) and speaking features (HuBERT) to effectively analyze both perspectives of audio signal. The experiment was conducted using DAIC-WOZ dataset, and the LTD-Conformer model achieved an accuracy of 87.04% and an F1-score of 0.87, demonstrating a 4% improvement in accuracy and a 0.04 increase in the F1 score compared to the existing Conformer model.  This study presents the possibility that the audio signal-based depression LTD-Conformer model can be effectively applied to depression diagnosis and will develop into a strong audio-based depression diagnosis model in the future

##### **Keywords:**  Audio, [Conformer](https://arxiv.org/abs/2005.08100), Depression, [HuBERT](https://arxiv.org/abs/2106.07447), Long-Term Dilated-Conformer, Mel-spectrogram 

## Work flow
![image](https://github.com/eejji/LTD-Conformer-Speech-Depression-Detection/blob/main/Figure/Workflow.png)
