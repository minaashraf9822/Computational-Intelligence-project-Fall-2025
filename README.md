Build Your Own Neural Network Library & Advanced ApplicationsCourse:

CSE473s: Computational Intelligence (Fall 2025) 1Faculty: Engineering Ain Shams University

📌 Project OverviewThis project implements a modular deep learning library from scratch using only Python and NumPy. The goal is to demystify the mathematics behind neural networks by building the core components manually:
Forward & Backward Propagation (Matrix Calculus)
Layers: Dense (Fully Connected)
Activations: ReLU, Sigmoid, Tanh
Optimizers: Stochastic Gradient Descent (SGD)
Loss Functions: Mean Squared Error (MSE)

The library is validated through three distinct tasks:

XOR Problem: Solving non-linear classification
Autoencoder: Unsupervised image reconstruction on the MNIST dataset
Latent Space Classification: Using the trained Encoder as a feature extractor for an SVM classifier10.

📂 Repository StructureThe project follows the mandatory structure required by the course11:Plaintext/
├── lib/ # Custom Neural Network Library
│ ├── **init**.py
│ ├── layers.py # Dense Layer implementation
│ ├── activations.py # ReLU, Sigmoid, Tanh
│ ├── losses.py # MSE Loss & Prime
│ ├── optimizer.py # SGD Optimizer
│ ├── network.py # Network class (train/predict loops)
│ ├── encoder.py # Encoder Architecture helper
│ ├── decoder.py # Decoder Architecture helper
│ ├── data_loader.py # MNIST loading & preprocessing
│ └── training_history.py # Visualization tools
│
├── notebooks/
│ └── project_demo.ipynb # Main Demo (XOR, Autoencoder, SVM) [cite: 101]
│
├── report/
│ └── project_report.pdf # Detailed Analysis & Results [cite: 95]
│
├── .gitignore
├── README.md
└── requirements.txt

⚙️ Installation & RequirementsTo run this project,
you need Python installed along with the following libraries:

on Bash

pip install numpy matplotlib scikit-learn tensorflow

(Note: TensorFlow is only used for the baseline comparison and data loading, not for the core library logic.)

🚀 How to RunClone the repository

on Bash

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

Run the Main Demo Notebook:Open notebooks/project_demo.ipynb in Jupyter Lab or VS Code. This single notebook contains the code for all three sections of the project:Section

1: XOR Validation (Training & Loss Plots)Section
2: Autoencoder Training (Reconstruction Visualization)Section
3: SVM Classification (Confusion Matrix & Accuracy)

📊 Results Summary

1. XOR ValidationObjective: Solve the XOR logic gate problem.Result: The network successfully converged, predicting 0 for [0,0]/[1,1] and 1 for [0,1]/[1,0].

2. Autoencoder (MNIST)Architecture: 784 $\to$ 128 $\to$ 64 $\to$ 128 $\to$ 784.Compression: Images were compressed by 92% (to 64 latent features).
   Outcome: Reconstructed images are blurry but clearly recognizable digits.

3. SVM ClassificationMethod: Transfer Learning (Encoder Features $\to$ SVM).Performance: Achieved ~86% Accuracy on the test set.Insight: Proves that the custom Encoder successfully learned meaningful semantic features of the handwritten digits.

⚖️ Comparison with TensorFlowWe compared our custom library against an identical architecture built in Keras/TensorFlow12.My Library: ~0.070 Final MSE LossTensorFlow: ~0.068 Final MSE Loss (Optimized SGD)

📝 Authors Mina Ashraf Rizk - ID: 2100447
