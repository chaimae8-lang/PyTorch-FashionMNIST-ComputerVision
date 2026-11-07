# PyTorch-FashionMNIST-TinyVGG

A production-grade, modular Computer Vision pipeline built from scratch using **PyTorch** to evaluate and classify the **FashionMNIST** dataset. This repository compares multiple architectural paradigms—ranging from baseline linear layers to non-linear deep denses and optimized convolutional neural networks (CNN)—leveraging advanced matrix broadcasting, tensor auditing, and macroeconomic evaluation metrics.

---

## 🔬 Core Architectures Compared & Benchmarked

1. **`FashionMNISTModelV0` (Linear Baseline):** 
   A purely linear network (`nn.Flatten` followed by `nn.Linear` layers) reaching an immutable performance ceiling at **83.42% accuracy**. This benchmark mathematically demonstrates the limitations of hyperplanes when slicing overlapping spatial features.
2. **`FashionMNISTModelV1` (Non-Linear MLP):** 
   A deep feedforward network integrating non-linear `nn.ReLU()` activations. This model exposes the high sensitivity of dense layers to flattened spatial matrices under sub-optimal hyperparameter boundaries.
3. **`FashionMNISTModelV2` (TinyVGG CNN):** 
   An elite **TinyVGG-style Convolutional Neural Network** mimicking the Oxford Visual Geometry Group paradigm. It implements a homogeneous chain of `3x3` convolutions with dimension-preserving padding (`padding='same'`) and `nn.MaxPool2d` downsampling blocks. By preserving 2D geometric topology, it breaks the linear ceiling to achieve a stable **89.02% validation accuracy** with a minimal cross-entropy loss score of **0.2918**.

---

## 📊 Evaluation & Optimization Performance

| Architectural Paradigm | Optimization Policy | Validation Loss | Final Test Accuracy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Linear MLP (V0)** | SGD (`lr=0.1`) | 0.4760 | 83.42% | Stabilized |
| **Non-Linear MLP (V1)**| Adam (`lr=0.001`) | 0.6850 | 75.02% | Underfitted |
| **TinyVGG CNN (V2)** | Adam (`lr=0.001`) | 0.3201 | 88.49% | Converged |
| **TinyVGG CNN (V2)** | **SGD (`lr=0.1, momentum=0.9`)** | **0.2918** | **89.02%** | 🏆 **Best Generalization** |

---

## 🛠️ Key Implementation Highlights

* **Modular Step Abstraction:** Completely abstracted the mini-batch execution loop into distinct `train_step()` and `test_step()` modular functions, drastically reducing VRAM overhead and improving pipeline maintainability.
* **Tensor Spatial Tracking:** Mastered the spatial dimension reduction transformations from an initial shape of `[32, 1, 28, 28]` down to a `[32, 10, 7, 7]` tensor representation following successive un-padded max-pooling downsampling routines.
* **Numerical Divergence Debugging:** Successfully identified, diagnosed, and resolved the critical `nan` (Not a Number) loss explosion bug caused by an aggressive `0.1` learning rate on raw convolutional outputs, recovering full network learning metrics through gradient-smoothing optimizer adaptations.
* **Macroeconomic Diagnostic:** Implemented a full verification pipeline leveraging `torchmetrics.classification.MulticlassAccuracy` coupled with an interactive colored Confusion Matrix (`mlxtend`) to track and analyze morphological classification overlaps between structural boundaries (e.g., *Pullovers* vs. *Coats*).

---

## 📁 Repository Structure

* `FashionMNIST_Classification_Workflow.ipynb`: Core production notebook featuring dataset downloads, custom class architectures, isolated training steps, and plotting configurations.
* `README.md`: Architectural documentation and benchmarking summary.

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com
   ```
2. Install the necessary machine learning ecosystem dependencies:
   ```bash
   pip install torch torchvision torchmetrics mlxtend tqdm matplotlib pandas
   ```
3. Run the notebook inside Google Colab or an active local Jupyter/VS Code kernel accelerated via an active **NVIDIA CUDA GPU**.
