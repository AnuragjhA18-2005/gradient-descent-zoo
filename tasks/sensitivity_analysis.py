import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.logistic_regression import generate_synthetic_data, LogisticRegression
from optimizers import Momentum, Adam


def run_lr_sensitivity(n_steps=500):
    """
    Sweep 3 learning rate values (0.1, 0.01, 0.001) for both Adam and Momentum (6 curves total).
    Returns a dictionary of loss histories and final metrics.
    """
    X, y = generate_synthetic_data(n_samples=600, random_state=42)

    # Identical initialization
    ref_model = LogisticRegression(in_features=2, seed=123)
    init_W = ref_model.W.copy()
    init_b = ref_model.b.copy()

    lr_values = [0.1, 0.01, 0.001]
    results = {}

    configs = [
        ("Momentum", Momentum, {"momentum": 0.9}),
        ("Adam", Adam, {"beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8}),
    ]

    for opt_name, OptClass, kwargs in configs:
        for lr in lr_values:
            key = f"{opt_name} (lr={lr})"
            model = LogisticRegression(in_features=2, seed=123)
            model.W[:] = init_W.copy()
            model.b[:] = init_b.copy()

            opt = OptClass(lr=lr, **kwargs)
            losses = []

            for _ in range(n_steps):
                y_pred = model.forward(X)
                losses.append(model.compute_loss(y, y_pred))
                grads = model.backward(X, y, y_pred)
                opt.step(model.params, grads)

            final_pred = model.forward(X)
            final_loss = model.compute_loss(y, final_pred)
            final_acc = model.accuracy(X, y)

            results[key] = {
                "losses": np.array(losses),
                "final_loss": final_loss,
                "final_acc": final_acc,
                "opt_name": opt_name,
                "lr": lr,
            }

    return results


def run_adam_beta1_sensitivity(n_steps=500, lr=0.05):
    """
    Sweep beta1 for Adam (0.85, 0.90, 0.99) at lr=0.05.
    Returns loss histories and metrics.
    """
    X, y = generate_synthetic_data(n_samples=600, random_state=42)

    ref_model = LogisticRegression(in_features=2, seed=123)
    init_W = ref_model.W.copy()
    init_b = ref_model.b.copy()

    beta1_values = [0.85, 0.90, 0.99]
    results = {}

    for b1 in beta1_values:
        key = f"Adam (β₁={b1})"
        model = LogisticRegression(in_features=2, seed=123)
        model.W[:] = init_W.copy()
        model.b[:] = init_b.copy()

        opt = Adam(lr=lr, beta1=b1, beta2=0.999, epsilon=1e-8)
        losses = []

        for _ in range(n_steps):
            y_pred = model.forward(X)
            losses.append(model.compute_loss(y, y_pred))
            grads = model.backward(X, y, y_pred)
            opt.step(model.params, grads)

        final_pred = model.forward(X)
        final_loss = model.compute_loss(y, final_pred)
        final_acc = model.accuracy(X, y)

        results[key] = {
            "losses": np.array(losses),
            "final_loss": final_loss,
            "final_acc": final_acc,
            "beta1": b1,
        }

    return results


def plot_lr_sensitivity(results, output_path="results/lr_sensitivity_analysis.png"):
    """Plot all 6 learning rate sensitivity curves on a single clear chart."""
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=100)

    # Styles for each of the 6 curves
    styles = {
        "Momentum (lr=0.1)": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 2.2},
        "Momentum (lr=0.01)": {"color": "#d95f02", "linestyle": "--", "linewidth": 2.0},
        "Momentum (lr=0.001)": {"color": "#a63603", "linestyle": ":", "linewidth": 2.0},
        "Adam (lr=0.1)": {"color": "#1f77b4", "linestyle": "-", "linewidth": 2.2},
        "Adam (lr=0.01)": {"color": "#4393c3", "linestyle": "--", "linewidth": 2.0},
        "Adam (lr=0.001)": {"color": "#92c5de", "linestyle": ":", "linewidth": 2.0},
    }

    n_steps = len(next(iter(results.values()))["losses"])
    steps = np.arange(1, n_steps + 1)

    for label, data in results.items():
        style = styles.get(label, {})
        final_l = data["final_loss"]
        ax.plot(steps, data["losses"], label=f"{label} [Final: {final_l:.4f}]", **style)

    ax.set_title("Hyperparameter Sensitivity: Learning Rate Sweep (Adam vs. Momentum)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Learning rate sensitivity plot saved to: {output_path}")


def plot_beta1_sensitivity(results, output_path="results/adam_beta1_sensitivity.png"):
    """Plot Adam beta1 sensitivity curves on a single chart."""
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=100)

    styles = {
        "Adam (β₁=0.85)": {"color": "#2b83ba", "linestyle": "--", "linewidth": 2.0},
        "Adam (β₁=0.9)": {"color": "#fdae61", "linestyle": "-", "linewidth": 2.2},
        "Adam (β₁=0.99)": {"color": "#d7191c", "linestyle": "-.", "linewidth": 2.0},
    }

    n_steps = len(next(iter(results.values()))["losses"])
    steps = np.arange(1, n_steps + 1)

    for label, data in results.items():
        style = styles.get(label, {})
        final_l = data["final_loss"]
        ax.plot(steps, data["losses"], label=f"{label} [Final: {final_l:.4f}]", **style)

    ax.set_title("Adam First-Moment Sensitivity: $\\beta_1$ Sweep (lr=0.05)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Optimization Step", fontsize=12)
    ax.set_ylabel("Binary Cross-Entropy Loss", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Adam beta1 sensitivity plot saved to: {output_path}")


def main():
    print("=" * 65)
    print(" Running Hyperparameter Sensitivity Analysis")
    print("=" * 65)

    # 1. Sweep learning rates
    lr_results = run_lr_sensitivity(n_steps=500)
    print("\n--- Learning Rate Sweep Results (500 Steps) ---")
    for name, data in lr_results.items():
        print(f"{name:25s} | Initial: {data['losses'][0]:.4f} | Final: {data['final_loss']:.4f} | Accuracy: {data['final_acc']:.2f}%")

    plot_lr_sensitivity(lr_results, output_path="results/lr_sensitivity_analysis.png")

    # 2. Sweep beta1 for Adam
    b1_results = run_adam_beta1_sensitivity(n_steps=500, lr=0.05)
    print("\n--- Adam Beta1 Sweep Results (500 Steps, lr=0.05) ---")
    for name, data in b1_results.items():
        print(f"{name:25s} | Step 25: {data['losses'][24]:.4f} | Step 50: {data['losses'][49]:.4f} | Final: {data['final_loss']:.4f}")

    plot_beta1_sensitivity(b1_results, output_path="results/adam_beta1_sensitivity.png")
    print("\nAll sensitivity analyses completed successfully!")


if __name__ == "__main__":
    main()
