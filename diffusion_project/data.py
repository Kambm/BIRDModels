"""Explicit dataset paths, deterministic ordering, and [-1,1] image normalization."""

import importlib
from pathlib import Path
import torch
from torch.utils.data import Dataset, Subset, TensorDataset
from .io import load_tensor


class ImageDirectory(Dataset):
    """Flat or nested preprocessed image directory, sorted by relative POSIX path."""

    def __init__(self, directory, transform):
        self.root = Path(directory).expanduser()
        self.paths = sorted(
            (
                p
                for p in self.root.rglob("*")
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            ),
            key=lambda p: p.relative_to(self.root).as_posix(),
        )
        if not self.paths:
            raise FileNotFoundError(f"No images under {self.root}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image

        with Image.open(self.paths[index]) as im:
            return self.transform(im.convert("RGB")), 0


def get_dataset(
    name, root="./data", image_dir=None, tensor_file=None, provider=None, download=False
):
    """provider='module:function' selects a custom dataset loader.

    tensor_file must contain normalized NCHW images in the desired dataset order.
    CelebA folder input assumes the desired crop is already applied; only resize occurs.
    """
    name = "celeba32" if name == "celeba" else name
    if provider:
        module, function = provider.split(":")
        return getattr(importlib.import_module(module), function)(name, root=str(root))
    if name not in ("mnist", "fashion_mnist", "cifar10", "celeba32", "celeba64"):
        raise ValueError(f"Unknown dataset {name}; use a custom provider")
    c, size = (1 if name in ("mnist", "fashion_mnist") else 3), (
        64 if name == "celeba64" else 32
    )
    meta = dict(
        name=name, image_size=size, num_channels=c, mean=[0.5] * c, std=[0.5] * c
    )
    if tensor_file:
        x = load_tensor(tensor_file)
        if tuple(x.shape[1:]) != (c, size, size) or x.min() < -1.001 or x.max() > 1.001:
            raise ValueError(
                "Tensor images must match dataset shape and be normalized to [-1,1]"
            )
        return TensorDataset(x, torch.zeros(len(x), dtype=torch.long)), meta
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * c, [0.5] * c),
        ]
    )
    if name.startswith("celeba"):
        if image_dir is None:
            raise ValueError(
                "Set image_dir explicitly for CelebA, or use tensor_file/provider"
            )
        return ImageDirectory(image_dir, transform), meta
    cls = {
        "mnist": datasets.MNIST,
        "fashion_mnist": datasets.FashionMNIST,
        "cifar10": datasets.CIFAR10,
    }[name]
    return (
        cls(
            root=str(Path(root).expanduser()),
            train=True,
            download=download,
            transform=transform,
        ),
        meta,
    )


def subset_indices(length, size=10000, num_subsets=2):
    if num_subsets not in (1, 2) or size < 1 or length < size * num_subsets:
        raise ValueError("Choose one/two subsets, a positive size, and enough data")
    return [list(range(i * size, (i + 1) * size)) for i in range(num_subsets)]


def checkpoint_name(kind, size, subset, suffix=""):
    label = f"{size//1000}k" if size % 1000 == 0 else str(size)
    return f"{kind}_{label}_s{subset}{suffix}"
