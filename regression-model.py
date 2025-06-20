"""
📈 Regression Model using PyTorch

This script builds and trains a simple linear regression model using PyTorch
to predict continuous outcomes from synthetic data.

Key Features:
- Clean modular structure
- GPU/CPU device agnostic
- MAE loss and SGD optimizer
- Training, evaluation, and visualization
- Model saving and loading
"""

# ==============================================================================
# 📦 Imports
# ==============================================================================
import torch
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

# ==============================================================================
# ⚙️ Configuration
# ==============================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
torch.manual_seed(SEED)

# ==============================================================================
# 📊 Data Generation
# ==============================================================================
def generate_data(start=0, stop=1, step=0.02, slope=0.7, bias=0.3):
    """Generates synthetic linear data."""
    X = torch.arange(start, stop, step).unsqueeze(1)
    y = slope * X + bias
    return X, y

def split_data(X, y, train_ratio=0.8):
    """Splits data into training and test sets."""
    split_idx = int(train_ratio * len(X))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

# ==============================================================================
# 📉 Visualization Utility
# ==============================================================================
def plot_predictions(train_data, train_labels, test_data, test_labels, preds=None):
    """Visualizes training, test, and predicted values."""
    plt.figure(figsize=(8, 4))
    plt.scatter(train_data, train_labels, label="Train Data", s=10, c="blue")
    plt.scatter(test_data, test_labels, label="Test Data", s=10, c="green")
    if preds is not None:
        plt.scatter(test_data, preds, label="Predictions", s=10, c="red")
    plt.title("Model Predictions vs Actual")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 🧠 Linear Regression Model
# ==============================================================================
class LinearRegressionModel(nn.Module):
    """Single-layer linear regression model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# ==============================================================================
# 🔁 Training Function
# ==============================================================================
def train(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01):
    """Trains the regression model."""
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_train, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(y_test, test_pred)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03} | Train Loss: {loss:.4f} | Test Loss: {test_loss:.4f}")

    return model

# ==============================================================================
# 💾 Save & Load Model
# ==============================================================================
def save_model(model, path="model/regression_model.pth"):
    """Saves model state_dict to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to: {path}")

def load_model(path="model/regression_model.pth"):
    """Loads model from saved state_dict."""
    model = LinearRegressionModel()
    model.load_state_dict(torch.load(path))
    model.to(DEVICE)
    model.eval()
    return model

# ==============================================================================
# 🚀 Main Execution
# ==============================================================================
if __name__ == "__main__":
    # Generate and prepare data
    X, y = generate_data()
    X_train, y_train, X_test, y_test = split_data(X, y)

    # Move to device
    X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    # Initialize model
    model = LinearRegressionModel().to(DEVICE)

    # Train model
    model = train(model, X_train, y_train, X_test, y_test)

    # Evaluate final predictions
    with torch.inference_mode():
        final_preds = model(X_test)

    # Plot predictions
    plot_predictions(X_train.cpu(), y_train.cpu(), X_test.cpu(), y_test.cpu(), final_preds.cpu())

    # Save model
    save_model(model)

    # Load model and validate
    loaded_model = load_model()
    with torch.inference_mode():
        loaded_preds = loaded_model(X_test)

    assert torch.allclose(final_preds, loaded_preds), "Mismatch in predictions after loading model!"
