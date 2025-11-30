# Human Activity Recognition (HAR) using PCA and Ensemble Learning in Julia

![Julia](https://img.shields.io/badge/-Julia-9558B2?style=flat&logo=julia&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.32%25-brightgreen)

## 📌 Project Overview
This project implements a Machine Learning pipeline to classify human activities (Standing, Sitting, Walking, etc.) using the **UCI HAR Dataset** (Samsung Galaxy S II sensor data). 

The primary goal was to investigate the effectiveness of **Dimensionality Reduction (PCA)** combined with **Ensemble Learning** to create a lightweight yet highly accurate model suitable for embedded environments.

### 🏆 Key Results
| Metric | Result | Details |
| :--- | :--- | :--- |
| **Cross-Validation Accuracy** | **97.31%** | 10-Fold Stratified CV |
| **Final Test Accuracy** | **93.32%** | On unseen test subjects |
| **Dimensionality Reduction** | **81.8%** | 561 Features → 102 Components |

---

## 🛠️ Methodology

### 1. Data Preprocessing
* **Standardization:** Applied **Z-Score Normalization** (Zero Mean, Unit Variance) to ensure optimal PCA performance.
* **Feature Extraction:** Hand-crafted features (561) provided by the original dataset were used as input.

### 2. Dimensionality Reduction (PCA)
* **Principal Component Analysis (PCA)** was applied to reduce the "Curse of Dimensionality."
* **Variance Retained:** 95%
* **Result:** Feature space reduced from **561** to **102** components.

### 3. Models & Experimentation
We trained and optimized four individual models using Julia's `MLJ` framework:
* **Artificial Neural Networks (ANN):** Best architecture `[200, 100]`.
* **Support Vector Machines (SVM):** Optimized with `Linear Kernel` and `C=1.0`.
* **k-Nearest Neighbors (kNN):** Tested various `k` values.
* **Decision Trees (DT):** Used as a baseline (Underperformed due to PCA rotation).

### 4. Ensemble Learning (The Solution)
To boost performance and generalization, we constructed a **Voting Ensemble** combining the top 3 models:
> **Ensemble = ANN + SVM + kNN**

This approach corrected misclassifications in dynamic activities where individual models struggled.

---

## 📊 Performance Evaluation

The model was evaluated on a strictly separated Test Set (30% of volunteers) to ensure subject independence.

| Model | Test Accuracy | F1-Score | Analysis |
| :--- | :--- | :--- | :--- |
| **Ensemble** | **93.32%** | **0.9329** | **Best Performance** |
| ANN | 93.04% | 0.9302 | Strong non-linear learning |
| SVM | 92.20% | 0.9216 | Highly efficient & robust |
| kNN | 84.39% | 0.8437 | Overfitting observed at k=1 |
| Decision Tree | 78.38% | 0.7836 | Struggled with rotated feature space |

### Discussion
The original study (Anguita et al.) achieved **96.4%** accuracy using all 561 features. Our approach achieved **93.32%** accuracy using only **102 features**. This **3% trade-off** in accuracy yields a **5x reduction** in computational complexity, making it ideal for mobile deployment.

---

## 📂 Dataset
The dataset used is the **Human Activity Recognition Using Smartphones Data Set**.
* **Source:** [UCI Machine Learning Repository & Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)
* **Input:** 561-feature vector with time and frequency domain variables.
* **Output:** 6 Activity Classes (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING).

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/brakcetin/Human-Activity-Recognition-Julia.git](https://github.com/brakcetin/Human-Activity-Recognition-Julia.git)
2. **Install Julia Dependencies: Open Julia REPL and run:**
    Launch Julia in the project directory and run the following commands to install all required packages.

    **Option A:** Using Pkg Mode (Recommended for interactive use) Press `]` in the Julia REPL to enter Pkg mode, then run:

    ```Julia
        activate .
        instantiate
    ```

    *(Note: This will automatically download all packages listed in `Project.toml` and `Manifest.toml` if you included them. If not, use **Option B**).*

    **Option B:** Manual Installation (Copy & Paste) If you are setting up from scratch, copy and paste this block into the Julia REPL:

    ```Julia
        using Pkg
        Pkg.activate(".")
        Pkg.add([
            "MLJ", 
            "MLJBase",
            "CategoricalArrays", 
            "MultivariateStats", 
            "MLJMultivariateStatsInterface",
            "LIBSVM", 
            "MLJLIBSVMInterface",
            "DecisionTree", 
            "MLJDecisionTreeInterface",
            "NearestNeighborModels", 
            "Flux", 
            "MLJEnsembles",
            "MLJLinearModels",
            "Plots", 
            "StableRNGs",
            "StatsBase"
        ])
    ```
3. **Download Data:** Download `train.csv` and `test.csv` from the Kaggle link above and place them in a folder named `dataset/`.

4. Run the Notebook: Open `notebooks/Burak_PCA_Approach.ipynb` in in Jupyter or VS Code and run all cells.

---

## 📚 References
1. Dataset: Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN.

2. Baseline: Anguita, D., et al. (2012). Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. IWAAL 2012.

---

*Created by Burak for the Machine Learning Final Project.*