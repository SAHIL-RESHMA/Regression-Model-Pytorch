# 🔢 Linear Regression Model with PyTorch

This project implements a **linear regression model using PyTorch**, designed to predict continuous values from input data by learning a simple linear relationship. It demonstrates end-to-end model development, including data generation, training, evaluation, visualization, and model persistence.

---

## 📌 Key Features

- Developed with **PyTorch**, utilizing its `nn.Module`, `L1Loss`, and optimization APIs.
- Trains a custom **linear regression model** from scratch.
- Visualizes predictions and data splits using **Matplotlib**.
- Incorporates **GPU/CPU compatibility** for efficient training.
- Includes **model saving and loading** using `torch.save` and `torch.load`.

---

## 🧠 Project Objective

The goal of this project is to:
- Generate a **synthetic dataset** with a known linear relationship.
- Build a **PyTorch-based regression model**.
- Train and evaluate the model on custom data.
- Save and reload the trained model for inference.
- Visualize how well the model fits the data.

---

## 📂 Dataset Overview

This project does not rely on external datasets. It uses **synthetically generated data**:

- Input Range: `0 to 1` (incremented by `0.02`)
- Target Function: `y = 0.7x + 0.3`
- Dataset Split:
  - **Training Set**: 80%
  - **Testing Set**: 20%

---

## ⚙️ How It Works

1. **Data Preparation**
   - Generate data using `torch.arange` and apply a linear function with optional noise.
   - Split into training and test datasets.

2. **Model Building**
   - Define a custom `LinearRegressionModel` using PyTorch’s `nn.Module`.

3. **Training Loop**
   - Use `L1Loss` (Mean Absolute Error) for simplicity.
   - Apply stochastic gradient descent (SGD) for optimization.
   - Log loss values every 10 epochs.

4. **Evaluation**
   - Disable gradient computation with `torch.inference_mode()`.
   - Calculate final loss on training and testing data.

5. **Visualization**
   - Use Matplotlib to show:
     - Training data
     - Testing data
     - Model predictions

6. **Model Persistence**
   - Save the model using `state_dict`.
   - Load it later and perform predictions.

---

## 📊 Libraries Used

- [`torch`](https://pytorch.org/) — Deep learning framework
- [`matplotlib`](https://matplotlib.org/) — Visualization library

---

## 💼 Real-World Applications

- **Finance**: Predict stock prices, trends, or interest rates.  
- **Marketing**: Estimate campaign ROI or customer lifetime value.  
- **Engineering**: Forecast load, pressure, or efficiency metrics.  
- **Healthcare**: Predict patient risk scores, biometrics, or recovery time.
