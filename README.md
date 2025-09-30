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
