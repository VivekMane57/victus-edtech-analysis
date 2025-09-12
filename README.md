<<<<<<< HEAD
# victus-edtech-analysis
IITB EdTech Internship – Group 18
=======
# Victus – Track 1 Educational Data Analysis

This repo follows the internship brief. Put your CSV files into `data/` (or change `base_path` in notebooks).

## Structure
```
project/
├─ data/
│  ├─ EEG.csv
│  ├─ GSR.csv
│  ├─ EYE.csv
│  ├─ IVT.csv
│  ├─ TIVA.csv
│  ├─ PSY.csv        # or ENG.csv
│  └─ externalEvents.csv (optional)
├─ notebooks/
│  ├─ 01_preprocessing.ipynb
│  ├─ 02_feature_cleaning.ipynb
│  ├─ 03_pca_analysis.ipynb
│  ├─ 04_lda_analysis.ipynb
│  ├─ 05_autoencoder_feature_learning.ipynb
│  └─ 06_feature_importance_analysis.ipynb
└─ models/
   ├─ pca_components.pkl
   ├─ lda_model.pkl
   └─ autoencoder_model.pt
```

## How to run
1. Create a Python env and install:
   ```bash
   pip install pandas numpy scikit-learn xgboost shap umap-learn torch torchvision matplotlib
   ```
2. Open notebooks **in order** (01 → 06). Each saves artifacts in `models/` and figures to `notebooks/figures`.

> If your data lives elsewhere (e.g., `D:\IITB\STData`), update `base_path` at the top of each notebook.
>>>>>>> e3fc313 (Initial scaffold commit for T1_G18_Victus (Group 18))
