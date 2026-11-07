"""Shared v-prediction training loop for DiT and UNet."""

import argparse
import csv
import json
import random
import re
import zlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from .models import build_model
from .data import get_dataset, subset_indices, checkpoint_name
from .diffusion import DiffusionSchedule, ddim_sample
from .io import atomic_torch_save, device_for, lock, write_json

DEFAULT_TRAIN = dict(
    diffusion_steps=1000,
    sample_steps=100,
    epochs=400,
    batch_size=32,
    num_workers=0,
    lr=2e-4,
    weight_decay=5e-2,
    grad_clip=1.0,
    amp=True,
    num_heads=8,
    dropout=0.0,
    seed=0,
)
DEFAULT_MODELS = {
    "dit": dict(patch_size=4, hidden_size=512, depth=12, mlp_ratio=4.0),
    "unet": dict(
        model_channels=128,
        channel_mult=[1, 2, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[16],
    ),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(
        torch=torch.get_rng_state(),
        numpy=np.random.get_state(),
        python=random.getstate(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def run_epoch(model, loader, optimizer, scaler, schedule, cfg, device):
    model.train()
    loss_sum, count = 0.0, 0
    for batch in loader:
        x = batch[0].to(device)
        t = torch.randint(schedule.T, (len(x),), device=device)
        noise = torch.randn_like(x)
        xt, target = schedule.q_sample(x, t, noise), schedule.v_target(x, t, noise)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=cfg["amp"] and device.type == "cuda",
        ):
            loss = F.mse_loss(model(xt, t), target)
        scaler.scale(loss).backward()
        if cfg["grad_clip"] is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(x)
        count += len(x)
    return loss_sum / count


def train(experiment, dataset=None, metadata=None, preview_callback=None):
    """Train selected runs. Existing runs resume; complete runs are skipped individually.

    Pass a dataset/metadata directly for custom data. Configuration and exact
    subset indices are stored with each checkpoint. Training resume requires the
    saved configuration, optimizer state, and RNG state.
    """
    exp = dict(experiment)
    kind = exp["model"]
    if kind not in DEFAULT_MODELS:
        raise ValueError("model must be dit or unet")
    if dataset is None:
        dataset, metadata = get_dataset(exp["dataset"], **exp.get("data", {}))
    cfg = dict(DEFAULT_TRAIN, **DEFAULT_MODELS[kind])
    cfg.update(exp.get("training", {}))
    cfg.update(exp.get("architecture", {}))
    cfg.update(image_size=metadata["image_size"], channels=metadata["num_channels"])
    size = exp.get("subset_size", 10000)
    splits = subset_indices(len(dataset), size, exp.get("num_subsets", 2))
    if size < cfg["batch_size"]:
        raise ValueError("subset_size must be at least batch_size (drop_last=True)")
    device = device_for(exp.get("device", "auto"))
    root = Path(exp.get("output_root", "./results")).expanduser()
    checkpoint_root = Path(
        exp.get("checkpoint_root", root / "model_checkpoints" / exp["dataset"])
    ).expanduser()
    histories = {}
    for suffix in exp.get("run_suffixes", ["", "_b", "_c"]):
        if not re.fullmatch(r"(?:_[A-Za-z0-9]+)?", suffix):
            raise ValueError(
                "Run suffix must be empty or an underscore followed by letters/numbers"
            )
        seed_offset = {"": 0, "_b": 1000, "_c": 2000}.get(
            suffix, zlib.crc32(suffix.encode())
        )
        for subset, indices in enumerate(splits, 1):
            run_seed = (cfg["seed"] + seed_offset + subset - 1) % (2**32)
            run_cfg = dict(cfg, seed=run_seed)
            directory = checkpoint_root / checkpoint_name(kind, size, subset, suffix)
            with lock(directory / ".train.lock"):
                files = sorted(directory.glob("epoch_*.pt"))
                if files and not exp.get("resume", True):
                    raise FileExistsError(
                        f"{directory} contains checkpoints; use another output root"
                    )
                set_seed(run_seed)
                model = build_model(kind, run_cfg).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=cfg["lr"],
                    betas=(0.9, 0.95),
                    weight_decay=cfg["weight_decay"],
                )
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, cfg["epochs"], eta_min=1e-6
                )
                scaler = torch.amp.GradScaler(
                    "cuda", enabled=cfg["amp"] and device.type == "cuda"
                )
                data_cfg = dict(
                    dataset=exp["dataset"],
                    subset_size=size,
                    subset=subset,
                    indices=indices,
                    loader=exp.get("data", {}),
                )
                preview_noise = torch.randn(
                    8,
                    cfg["channels"],
                    cfg["image_size"],
                    cfg["image_size"],
                    generator=torch.Generator().manual_seed(cfg["seed"]),
                )
                start, history = 1, []
                if files:
                    ck = torch.load(files[-1], map_location="cpu", weights_only=False)
                    if ck["config"] != run_cfg or ck.get("data_config") != data_cfg:
                        raise ValueError(
                            f"{directory}: resume configuration differs; choose a new run"
                        )
                    model.load_state_dict(ck["model_state_dict"])
                    optimizer.load_state_dict(ck["optimizer_state_dict"])
                    scheduler.load_state_dict(ck["scheduler_state_dict"])
                    scaler.load_state_dict(ck["scaler_state_dict"])
                    start, history = ck["epoch"] + 1, ck["history"]
                    restore_rng(ck["rng_state"])
                if start > cfg["epochs"]:
                    print(f"Complete; skipping {directory}", flush=True)
                    histories[str(directory)] = history
                    del model, optimizer, scheduler, scaler
                    continue
                loader = DataLoader(
                    Subset(dataset, indices),
                    batch_size=cfg["batch_size"],
                    shuffle=True,
                    num_workers=cfg["num_workers"],
                    drop_last=True,
                    pin_memory=device.type == "cuda",
                    persistent_workers=cfg["num_workers"] > 0,
                )
                schedule = DiffusionSchedule(cfg["diffusion_steps"], device)
                write_json(directory / "experiment.json", exp)
                for epoch in range(start, cfg["epochs"] + 1):
                    loss = run_epoch(
                        model, loader, optimizer, scaler, schedule, run_cfg, device
                    )
                    scheduler.step()
                    history.append(dict(epoch=epoch, loss=loss))
                    every = exp.get("preview_every", 10)
                    if every and epoch % every == 0:
                        samples = ddim_sample(
                            model, preview_noise, schedule, cfg["sample_steps"]
                        )
                        atomic_torch_save(samples, directory / "preview.pt")
                        if preview_callback:
                            preview_callback(
                                samples, f"{kind} s{subset}{suffix} epoch {epoch}"
                            )
                    atomic_torch_save(
                        dict(
                            epoch=epoch,
                            model_state_dict=model.state_dict(),
                            optimizer_state_dict=optimizer.state_dict(),
                            scheduler_state_dict=scheduler.state_dict(),
                            scaler_state_dict=scaler.state_dict(),
                            config=run_cfg,
                            model_type=kind,
                            data_config=data_cfg,
                            rng_state=rng_state(),
                            history=history,
                        ),
                        directory / f"epoch_{epoch:04d}.pt",
                    )
                    with (directory / "history.csv").open("w") as f:
                        writer = csv.DictWriter(f, fieldnames=["epoch", "loss"])
                        writer.writeheader()
                        writer.writerows(history)
                    print(
                        f'{kind} s{subset}{suffix} epoch {epoch}/{cfg["epochs"]}: {loss:.6f}',
                        flush=True,
                    )
                histories[str(directory)] = history
                del model, optimizer, scheduler, scaler, loader
    return histories


def main(model=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--suffix", help="Run one suffix; base means the empty suffix")
    args = parser.parse_args()
    exp = json.loads(Path(args.config).read_text())
    if model is not None:
        exp["model"] = model
    if args.suffix is not None:
        exp["run_suffixes"] = ["" if args.suffix == "base" else args.suffix]
    train(exp)


if __name__ == "__main__":
    main()
