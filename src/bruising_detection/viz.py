import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def show_augmentations(dataset, idx=0, band=99, n=6):
    old_aug = dataset.augment
    dataset.augment = None
    x0, label = dataset[idx]
    dataset.augment = old_aug

    band = min(band, x0.size(0) - 1)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.2 * (n + 1), 2.4))
    axes[0].imshow(x0[band].numpy(), cmap="gray")
    axes[0].set_title(f"original\ny={int(label)}")
    axes[0].axis("off")
    for ax in axes[1:]:
        x = dataset.augment(x0.clone()) if dataset.augment else x0
        ax.imshow(x[band].numpy(), cmap="gray")
        ax.set_title("augmented")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def cnn_architecture_steps(input_shape, conv_channels, kernel_size):
    h, w, bands = input_shape
    steps = [f"Input\n{h}x{w}x{bands}"]
    in_c = bands
    for i, out_c in enumerate(conv_channels, 1):
        h, w = h // 2, w // 2
        steps.append(f"Conv {i}\n{in_c}->{out_c}, k={kernel_size}")
        steps.append(f"MaxPool {i}\n{h}x{w}x{out_c}")
        in_c = out_c
    steps += ["Global AvgPool\n1x1", "Dropout", "Linear\n2 classes"]
    return steps


def plot_cnn_architecture(model, input_shape, config, save_path=None, show=True):
    steps = cnn_architecture_steps(input_shape, config.conv_channels, config.kernel_size)
    fig, ax = plt.subplots(figsize=(max(10, 1.45 * len(steps)), 3.2))
    ax.set_xlim(0, len(steps))
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, step in enumerate(steps):
        box = FancyBboxPatch(
            (i + 0.05, 0.30),
            0.82,
            0.40,
            boxstyle="round,pad=0.03",
            facecolor="white",
            edgecolor="black",
            linewidth=1.4,
        )
        ax.add_patch(box)
        ax.text(i + 0.46, 0.50, step, ha="center", va="center", fontsize=9)
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.02, 0.50), xytext=(i + 0.90, 0.50), arrowprops={"arrowstyle": "->", "lw": 1.3})

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ax.set_title(f"Shallow CNN architecture | parameters: {n_params:,} ({trainable:,} trainable)", loc="left")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Save the shallow CNN architecture graph.")
    parser.add_argument("--out", default="cnn_architecture.png", help="Output image path.")
    parser.add_argument("--height", type=int, default=160, help="Input crop height.")
    parser.add_argument("--width", type=int, default=160, help="Input crop width.")
    parser.add_argument("--bands", type=int, default=220, help="Number of spectral bands.")
    args = parser.parse_args()

    from .config import CNNConfig
    from .model import ShallowCNN

    config = CNNConfig()
    input_shape = (args.height, args.width, args.bands)
    model = ShallowCNN(
        in_channels=args.bands,
        conv_channels=config.conv_channels,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )
    out = Path(args.out)
    plot_cnn_architecture(model, input_shape, config, save_path=out, show=False)
    print(f"Saved {out.resolve()}")


# if __name__ == "__main__":
#     main()
