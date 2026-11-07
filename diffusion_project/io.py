"""Small file helpers; artifacts are written atomically."""

import contextlib
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    os.close(fd)
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_tensor(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(value) or value.ndim != 4 or not torch.isfinite(value).all():
        raise ValueError(f"{path}: expected finite [N,C,H,W] tensor")
    return value.float()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


@contextlib.contextmanager
def lock(path):
    """OS advisory lock, released even when a process dies (Linux/macOS)."""
    import fcntl

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def device_for(name="auto"):
    return torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else name
    )
