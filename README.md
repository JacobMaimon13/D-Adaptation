# Extending D-Adaptation to ASGD Optimization 🧠

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c)
![Status](https://img.shields.io/badge/Status-Research-green)

**Authors:** Jacob Maimon, Bar Naor  
**Institution:** Ben-Gurion University of the Negev  
**Course:** Introduction to Deep Learning

---

## 📌 Overview

This project explores the integration of **D-Adaptation**—a method for automatic learning rate selection—into the **Averaged Stochastic Gradient Descent (ASGD)** optimizer. 

Traditionally, D-Adaptation has been proven effective for SGD and Adam. Our research aims to determine if ASGD can similarly benefit from an adaptive learning rate mechanism, removing the need for manual hyperparameter tuning while maintaining convergence stability.

## 🔬 Methodology

We modified the ASGD update rule to incorporate the adaptation parameter $d_k$, which estimates the distance to the solution. We experimented with four different mathematical formulations for the learning rate $\lambda_k$:

1.  **Denominator Integration (Best Performance):** $$\lambda_{k} = \frac{\gamma_{t}}{d_{k}}$$
2.  **Numerator Integration:**
    $$\lambda_{k} = \gamma_{t} \cdot d_{k}$$
3.  **Square Root Scaling:** Variations of $\sqrt{d_k}$ in numerator and denominator.

## 📊 Experiments & Results

We evaluated the modified optimizers on Accuracy, Loss, and Training Time.

### 1. The Most Promising Approach: $d$ in Denominator
Placing the adaptation parameter in the denominator yielded the fastest initial convergence. As seen below, the training loss (orange line) drops significantly faster than standard ASGD, though it requires careful handling to avoid overfitting in later epochs.

![Comparison of Loss and Accuracy](assets/graph_denominator.jpg)
*(Figure 1: Comparison of Loss and Accuracy when d is in the denominator)*

### 2. Alternative Approaches
Other formulations, such as placing $d$ in the numerator or using square roots, generally resulted in slower convergence or higher loss compared to the baseline ASGD.

| Method | Initial Convergence | Stability | Final Accuracy |
| :--- | :--- | :--- | :--- |
| **Denominator ($1/d$)** | ⭐⭐⭐ Fast | ⭐ Moderate | High |
| **Numerator ($*d$)** | ⭐ Slow | ⭐⭐ Stable | Low |
| **Sqrt Denom ($1/\sqrt{d}$)** | ⭐ Very Slow | ⭐ Stable | Low |

*(See full report for detailed breakdown of all experiments)*

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JacobMaimon13/D-Adaptation-ASGD.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch torchvision matplotlib numpy
    ```

3.  **Run the training script:**
    ```bash
    python train.py --optimizer dadapt_asgd --method denominator
    ```

## 📄 Conclusion

Our findings suggest that applying D-Adaptation to ASGD is viable, with the **denominator formulation** showing strong potential for accelerating the training process. Future work involves refining the stability of this method for long-duration training.

---
*Based on the final project report submitted to the Dept. of Industrial Engineering and Management, BGU, 2025.*

## 🙏 Acknowledgments

This repository is based on the official implementation of **D-Adaptation** by Meta Research.
* **Original Codebase:** [facebookresearch/dadaptation](https://github.com/facebookresearch/dadaptation)
* **Original Paper:** Defazio, A., & Mishchenko, K. (2023). *Learning-Rate-Free Learning by D-Adaptation*.
