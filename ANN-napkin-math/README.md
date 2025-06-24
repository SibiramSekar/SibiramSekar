# 🧠 Simple ANN from Scratch (Napkin Math Style)

This project demonstrates how to build a simple **Artificial Neural Network (ANN)** entirely from scratch using **NumPy**, without relying on any machine learning frameworks like TensorFlow or PyTorch.

> This is a "Napkin Math" implementation — raw math, minimal dependencies, and full control.

---

## 📘 What’s Inside

- ✅ Manual initialization of weights and biases
- ✅ Forward propagation implemented with matrix multiplication
- ✅ Softmax activation for output layer
- ✅ Cross-entropy loss function
- ✅ One-hot encoded labels
- ✅ A simple architecture for digit recognition (e.g., MNIST)
- ✅ Clean and beginner-friendly Python code

---

## 📁 Files

| File | Description |
|------|-------------|
| `sibiram-ann-math.ipynb` | Jupyter notebook containing all code for building and training the ANN |
| `README.md` | This file |

---

## 🛠️ How It Works

The model follows a basic 2-layer neural network structure:

- **Input layer:** 784 nodes (for 28x28 pixel images)
- **Hidden layer:** 10 nodes with ReLU or Sigmoid activation
- **Output layer:** 10 nodes (one for each digit class, using Softmax)

Key operations are implemented manually:
- `Z1 = W1.dot(X) + b1`
- `A1 = activation(Z1)`
- `Z2 = W2.dot(A1) + b2`
- `A2 = softmax(Z2)`

---

## 🧪 Dataset

This notebook assumes the use of a digit recognition dataset (e.g., MNIST). Data is flattened and normalized before being fed into the ANN.

---

## 🚀 How to Run

1. Clone the repository or download the notebook.
2. Run the notebook in:
   - Kaggle
   - Jupyter Notebook locally
3. All required packages are built-in:
   ```bash
   pip install numpy
