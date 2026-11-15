import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from optimizers import SGD, Momentum, NAG, AdaGrad, AdaDelta, RMSprop, Adam


def generate_synthetic_data(n_samples=600, random_state=42):
    """
    Generate a 2D binary classification synthetic dataset using np.random.randn.
    
    Creates two Gaussian clusters with some overlap:
      - Class 0: centered around (-1.5, -1.5)
      - Class 1: centered around (1.5, 1.5)
    
    Returns:
        X: ndarray of shape (n_samples, 2)
        y: ndarray of shape (n_samples, 1) with values in {0.0, 1.0}
    """
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2

    X_neg = rng.randn(n_per_class, 2) + np.array([-1.5, -1.5])
    y_neg = np.zeros((n_per_class, 1), dtype=np.float64)

    X_pos = rng.randn(n_per_class, 2) + np.array([1.5, 1.5])
    y_pos = np.ones((n_per_class, 1), dtype=np.float64)

    X = np.vstack([X_pos, X_neg])
    y = np.vstack([y_pos, y_neg])

    # Shuffle dataset
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using NumPy.
    Supports forward pass, cross-entropy loss, and backward pass for gradient computation.
    """
    def __init__(self, in_features=2, seed=123):
        self.in_features = in_features
        self.seed = seed
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights W and bias b with a fixed seed."""
        rng = np.random.RandomState(self.seed)
        self.W = rng.randn(self.in_features, 1).astype(np.float64) * 0.1
        self.b = np.zeros((1,), dtype=np.float64)
        self.params = [self.W, self.b]

    @staticmethod
    def sigmoid(z):
        """Numerically stable sigmoid function."""
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def forward(self, X):
        """
        Forward pass of logistic regression.
        z = X @ W + b
        y_hat = sigmoid(z)
        """
        z = X @ self.W + self.b
        return self.sigmoid(z)

    @staticmethod
    def compute_loss(y_true, y_pred, eps=1e-15):
        """
        Binary cross-entropy loss:
        L = - (1 / N) * sum(y * log(y_hat + eps) + (1 - y) * log(1 - y_hat + eps))
        """
        y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred_clipped) + (1.0 - y_true) * np.log(1.0 - y_pred_clipped))
        return float(loss)

    def backward(self, X, y_true, y_pred):
        """
        Compute gradients of binary cross-entropy loss w.r.t. parameters W and b.
        grad_W = (1 / N) * (X^T @ (y_pred - y_true))
        grad_b = (1 / N) * sum(y_pred - y_true) = mean(y_pred - y_true)
        """
        N = X.shape[0]
        error = y_pred - y_true
        grad_W = (X.T @ error) / N
        grad_b = np.mean(error, axis=0)
        return [grad_W, grad_b]

    def predict(self, X):
        """Predict binary class labels {0, 1}."""
        y_pred = self.forward(X)
        return (y_pred >= 0.5).astype(np.float64)

    def accuracy(self, X, y_true):
        """Compute classification accuracy percentage."""
        preds = self.predict(X)
        return float(np.mean(preds == y_true) * 100.0)


def run_benchmark(n_steps=500, plot_path="results/logistic_regression_convergence.png"):
    """
    Run all 7 optimizers for n_steps each with identical initialization,
    record loss curves, and save comparison plot.
    """
    # 1. Generate dataset
    X, y = generate_synthetic_data(n_samples=600, random_state=42)

    # 2. Reference model to obtain fixed initial weights
    ref_model = LogisticRegression(in_features=2, seed=123)
    init_W = ref_model.W.copy()
    init_b = ref_model.b.copy()

    # 3. Configure all 7 optimizers
    optimizers = {
        "SGD": SGD(lr=0.1),
        "Momentum": Momentum(lr=0.05, momentum=0.9),
        "NAG": NAG(lr=0.05, momentum=0.9),
        "AdaGrad": AdaGrad(lr=0.5, epsilon=1e-8),
        "AdaDelta": AdaDelta(rho=0.9, epsilon=1e-6),
        "RMSprop": RMSprop(lr=0.02, rho=0.9, epsilon=1e-8),
        "Adam": Adam(lr=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8),
    }

    results = {}

    print("=" * 70)
    print(" Running Logistic Regression Benchmark (All 7 Optimizers)")
    print("=" * 70)

    # 4. Train each optimizer
    for name, opt in optimizers.items():
        # Initialize model with identical weights
        model = LogisticRegression(in_features=2, seed=123)
        model.W[:] = init_W.copy()
        model.b[:] = init_b.copy()

        loss_history = []

        for step in range(n_steps):
            # Forward pass
            y_pred = model.forward(X)

            # Record loss at every step
            loss = model.compute_loss(y, y_pred)
            loss_history.append(loss)

            # Backward pass
            grads = model.backward(X, y, y_pred)

            # In-place parameter update
            opt.step(model.params, grads)

        # Final evaluation
        final_y_pred = model.forward(X)
        final_loss = model.compute_loss(y, final_y_pred)
        final_acc = model.accuracy(X, y)

        results[name] = {
            "losses": loss_history,
            "final_loss": final_loss,
            "accuracy": final_acc,
        }

        print(f"{name:12s} | Initial Loss: {loss_history[0]:.4f} | Final Loss: {final_loss:.4f} | Accuracy: {final_acc:.2f}%")

    print("=" * 70)

    # 5. Plot convergence curves on a single figure
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)

    plt.figure(figsize=(9, 5.5), dpi=100)

    # Color palette and line styles for distinct visibility
    styles = {
        "SGD": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.0},
        "Momentum": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 2.0},
        "NAG": {"color": "#2ca02c", "linestyle": "-.", "linewidth": 2.0},
        "AdaGrad": {"color": "#d62728", "linestyle": ":", "linewidth": 2.2},
        "AdaDelta": {"color": "#9467bd", "linestyle": "-", "linewidth": 1.8},
        "RMSprop": {"color": "#8c564b", "linestyle": "--", "linewidth": 2.0},
        "Adam": {"color": "#e377c2", "linestyle": "-", "linewidth": 2.2},
    }

    steps_range = np.arange(1, n_steps + 1)
    for name, data in results.items():
        style = styles.get(name, {})
        plt.plot(
            steps_range,
            data["losses"],
            label=f"{name} (Final: {data['final_loss']:.4f})",
            **style
        )

    plt.title("Convergence Comparison of 7 Optimizers on Logistic Regression", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10, loc="upper right", framealpha=0.95)
    plt.tight_layout()

    plt.savefig(plot_path)
    plt.close()
    print(f"Convergence plot successfully saved to: {plot_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
