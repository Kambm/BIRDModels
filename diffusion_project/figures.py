"""Notebook previews (no implicit file writes)."""

import math
import matplotlib.pyplot as plt
import numpy as np


def sample_grid(samples, title="", columns=4):
    images = (samples.detach().cpu() * 0.5 + 0.5).clamp(0, 1)
    fig, axes = plt.subplots(
        math.ceil(len(images) / columns),
        columns,
        figsize=(3 * columns, 3 * math.ceil(len(images) / columns)),
        squeeze=False,
    )
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            image = images[i].permute(1, 2, 0).numpy()
            ax.imshow(
                image[:, :, 0] if image.shape[-1] == 1 else image,
                cmap="Greys_r",
                vmin=0,
                vmax=1,
            )
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return fig
