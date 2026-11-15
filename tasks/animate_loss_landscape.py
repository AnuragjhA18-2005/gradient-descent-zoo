import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tasks.logistic_regression import generate_synthetic_data, LogisticRegression
from optimizers import SGD, Momentum, NAG, AdaGrad, AdaDelta, RMSprop, Adam


def compute_loss_grid(X, y, w1_range, w2_range, b_val=-0.26, grid_size=120):
    """Compute the binary cross-entropy loss surface over a 2D grid of (w1, w2)."""
    w1_vals = np.linspace(w1_range[0], w1_range[1], grid_size)
    w2_vals = np.linspace(w2_range[0], w2_range[1], grid_size)
    W1, W2 = np.meshgrid(w1_vals, w2_vals)

    eps = 1e-15
    Z = np.zeros_like(W1)

    for i in range(grid_size):
        for j in range(grid_size):
            w = np.array([[W1[i, j]], [W2[i, j]]], dtype=np.float64)
            z = X @ w + b_val
            y_hat = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))
            y_hat = np.clip(y_hat, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat))
            Z[i, j] = loss

    return W1, W2, Z


def record_optimizer_trajectories(X, y, n_steps=500):
    """
    Run all 7 optimizers from identical initial weights and record
    (w1, w2, b, loss) at every step.
    """
    ref_model = LogisticRegression(in_features=2, seed=123)
    init_W = ref_model.W.copy()
    init_b = ref_model.b.copy()

    optimizers = {
        "SGD": (SGD(lr=0.1), "#1f77b4", "-"),
        "Momentum": (Momentum(lr=0.05, momentum=0.9), "#ff7f0e", "--"),
        "NAG": (NAG(lr=0.05, momentum=0.9), "#2ca02c", "-."),
        "AdaGrad": (AdaGrad(lr=0.5, epsilon=1e-8), "#d62728", ":"),
        "AdaDelta": (AdaDelta(rho=0.9, epsilon=1e-6), "#9467bd", "-"),
        "RMSprop": (RMSprop(lr=0.02, rho=0.9, epsilon=1e-8), "#8c564b", "--"),
        "Adam": (Adam(lr=0.05, beta1=0.9, beta2=0.999, epsilon=1e-8), "#e377c2", "-"),
    }

    trajectories = {}

    for name, (opt, color, linestyle) in optimizers.items():
        model = LogisticRegression(in_features=2, seed=123)
        model.W[:] = init_W.copy()
        model.b[:] = init_b.copy()

        w1_hist = [float(model.W[0, 0])]
        w2_hist = [float(model.W[1, 0])]
        b_hist = [float(model.b[0])]
        loss_hist = [model.compute_loss(y, model.forward(X))]

        for _ in range(n_steps):
            y_pred = model.forward(X)
            grads = model.backward(X, y, y_pred)
            opt.step(model.params, grads)

            w1_hist.append(float(model.W[0, 0]))
            w2_hist.append(float(model.W[1, 0]))
            b_hist.append(float(model.b[0]))
            loss_hist.append(model.compute_loss(y, model.forward(X)))

        trajectories[name] = {
            "w1": np.array(w1_hist),
            "w2": np.array(w2_hist),
            "b": np.array(b_hist),
            "loss": np.array(loss_hist),
            "color": color,
            "linestyle": linestyle,
        }

    return trajectories


def create_static_contour_plot(W1, W2, Z, trajectories, output_path="results/loss_landscape_contours.png"):
    """Generate a high-resolution static contour plot with all full trajectories overlaid."""
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=100)

    # Contour levels
    levels = np.array([0.038, 0.045, 0.06, 0.08, 0.12, 0.2, 0.35, 0.6, 1.0, 1.5, 2.5, 3.8])
    cs = ax.contourf(W1, W2, Z, levels=levels, cmap="Blues_r", alpha=0.75, extend="max")
    cbar = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Binary Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.contour(W1, W2, Z, levels=levels, colors="gray", linewidths=0.6, alpha=0.5)

    # Initial start point
    start_w1 = trajectories["SGD"]["w1"][0]
    start_w2 = trajectories["SGD"]["w2"][0]
    ax.plot(start_w1, start_w2, "ko", markersize=8, zorder=6, label=f"Start ({start_w1:.2f}, {start_w2:.2f})")

    # Plot each trajectory
    for name, data in trajectories.items():
        w1, w2 = data["w1"], data["w2"]
        final_loss = data["loss"][-1]
        ax.plot(w1, w2, color=data["color"], linestyle=data["linestyle"],
                linewidth=2.0, label=f"{name} (Loss: {final_loss:.4f})", zorder=4)
        ax.plot(w1[-1], w2[-1], marker="x", markersize=7, color=data["color"], markeredgewidth=2.0, zorder=5)

    ax.set_title("Optimization Trajectories on Logistic Regression Loss Landscape", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Weight $w_1$", fontsize=12)
    ax.set_ylabel("Weight $w_2$", fontsize=12)
    ax.set_xlim(W1.min(), W1.max())
    ax.set_ylim(W2.min(), W2.max())
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9.5, loc="lower right", framealpha=0.95)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Static contour plot saved to: {output_path}")


def create_animation(W1, W2, Z, trajectories, n_steps=500, output_path="results/loss_landscape_trajectory.gif", fps=18):
    """
    Animate the trajectories of all 7 optimizers traversing the loss landscape.
    Saves as an animated GIF using matplotlib.animation and PillowWriter.
    """
    # Sample frame indices: high temporal resolution in early steps, moderate later
    frame_indices = (
        list(range(0, 50, 1)) +
        list(range(50, 150, 2)) +
        list(range(150, n_steps, 5)) +
        [n_steps]
    )

    fig, ax = plt.subplots(figsize=(10, 8), dpi=130)

    # Background contour
    levels = np.array([0.038, 0.045, 0.06, 0.08, 0.12, 0.2, 0.35, 0.6, 1.0, 1.5, 2.5, 3.8])
    cs = ax.contourf(W1, W2, Z, levels=levels, cmap="Blues_r", alpha=0.75, extend="max")
    cbar = fig.colorbar(cs, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Binary Cross-Entropy Loss", fontsize=11, fontweight="bold")
    ax.contour(W1, W2, Z, levels=levels, colors="gray", linewidths=0.6, alpha=0.5)

    # Initial start marker
    start_w1 = trajectories["SGD"]["w1"][0]
    start_w2 = trajectories["SGD"]["w2"][0]
    ax.plot(start_w1, start_w2, "ko", markersize=8, zorder=6, label=f"Start ({start_w1:.2f}, {start_w2:.2f})")

    # Lines and point heads for each optimizer
    lines = {}
    heads = {}
    for name, data in trajectories.items():
        line, = ax.plot([], [], color=data["color"], linestyle=data["linestyle"],
                        linewidth=2.0, label=name, zorder=4)
        head, = ax.plot([], [], marker="o", markersize=6, color=data["color"], zorder=5)
        lines[name] = line
        heads[name] = head

    # Dynamic step text box
    step_text = ax.text(
        0.03, 0.95, "", transform=ax.transAxes, fontsize=11,
        fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9, edgecolor="gray")
    )

    ax.set_title("Animated Trajectories of 7 Optimizers on Loss Landscape", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Weight $w_1$", fontsize=12)
    ax.set_ylabel("Weight $w_2$", fontsize=12)
    ax.set_xlim(W1.min(), W1.max())
    ax.set_ylim(W2.min(), W2.max())
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)

    def init():
        for name in trajectories:
            lines[name].set_data([], [])
            heads[name].set_data([], [])
        step_text.set_text("")
        return list(lines.values()) + list(heads.values()) + [step_text]

    def update(frame_idx):
        for name, data in trajectories.items():
            w1_sub = data["w1"][:frame_idx + 1]
            w2_sub = data["w2"][:frame_idx + 1]
            lines[name].set_data(w1_sub, w2_sub)
            heads[name].set_data([w1_sub[-1]], [w2_sub[-1]])

        step_text.set_text(f"Optimization Step: {frame_idx:3d} / {n_steps}")
        return list(lines.values()) + list(heads.values()) + [step_text]

    print(f"Rendering {len(frame_indices)} animation frames...")
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=frame_indices,
        init_func=init,
        blit=True,
        interval=1000 // fps
    )

    writer = animation.PillowWriter(fps=fps)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Animation successfully saved to: {output_path}")


def main():
    print("Generating synthetic data and computing trajectories...")
    X, y = generate_synthetic_data(n_samples=600, random_state=42)
    trajectories = record_optimizer_trajectories(X, y, n_steps=500)

    # Determine landscape bounding box from trajectories
    all_w1 = np.concatenate([t["w1"] for t in trajectories.values()])
    all_w2 = np.concatenate([t["w2"] for t in trajectories.values()])
    w1_range = (min(-0.8, float(np.min(all_w1)) - 0.5), max(3.8, float(np.max(all_w1)) + 0.5))
    w2_range = (min(-0.8, float(np.min(all_w2)) - 0.5), max(3.2, float(np.max(all_w2)) + 0.5))

    # Mean optimal bias
    mean_bias = float(np.mean([t["b"][-1] for t in trajectories.values()]))
    print(f"Computing loss grid over w1={w1_range}, w2={w2_range} with b={mean_bias:.3f}...")
    W1, W2, Z = compute_loss_grid(X, y, w1_range, w2_range, b_val=mean_bias, grid_size=100)

    # 1. Create static high-res contour plot
    create_static_contour_plot(W1, W2, Z, trajectories, output_path="results/loss_landscape_contours.png")

    # 2. Create animated GIF
    create_animation(W1, W2, Z, trajectories, n_steps=500, output_path="results/loss_landscape_trajectory.gif", fps=18)

    print("All visualizations created successfully in results/")


if __name__ == "__main__":
    main()
