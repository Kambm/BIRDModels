"""Unified, resumable calibration and generation for Wiener LS or boundary-broken ELS."""

import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import Subset
from ..data import get_dataset, subset_indices
from ..io import atomic_torch_save, device_for, lock, load_tensor, digest, write_json
from .common import images_from, cosine, sample
from .wiener import WienerLS, dataset_statistics
from .els import BoundaryBrokenELS
from .calibration import calibrate_step


def make_model(kind, dataset, config):
    if kind == "wiener":
        return WienerLS(
            dataset,
            source_batch_size=config.get("source_batch_size", 32),
            target_batch_size=config.get("target_batch_size", 64),
        )
    if kind == "els":
        return BoundaryBrokenELS(
            dataset,
            source_batch_size=config.get("source_batch_size", 8),
            candidate_batch_size=config.get("candidate_batch_size", 1024),
            query_batch_size=config.get("query_batch_size", 128),
        )
    raise ValueError("analytical_model must be wiener or els")


def run(experiment, dataset=None, metadata=None):
    config = dict(experiment)
    kind = config.get("analytical_model", "wiener")
    if dataset is None:
        dataset, metadata = get_dataset(config["dataset"], **config.get("data", {}))
    device = device_for(config.get("device", "auto"))
    size = config.get("subset_size", 10000)
    splits = subset_indices(len(dataset), size, config.get("num_subsets", 2))
    root = Path(config.get("output_root", "./results")).expanduser()
    saved = root / "saved_tensors" / config["dataset"]
    calibration_dir = root / "calibration" / config["dataset"] / kind
    cal = config.get("calibration", {})
    times = torch.tensor(
        cal.get("times", torch.linspace(0.005, 1.0, 20).tolist()), dtype=torch.float64
    )
    if (
        len(times) < 2
        or not torch.all(times[1:] > times[:-1])
        or not 0 < times[0] < times[-1] <= 1
    ):
        raise ValueError("Calibration times must ascend within (0,1]")
    seed = config.get("seed", 0)
    generator = torch.Generator().manual_seed(seed)
    source_size, val_size = cal.get("source_size", 5000), cal.get(
        "validation_size", 2000
    )
    if min(source_size, val_size) < 1 or source_size + val_size > len(dataset):
        raise ValueError("Calibration source/validation split exceeds dataset")
    ordering = (
        torch.randperm(len(dataset), generator=generator)
        if cal.get("randomize", False)
        else torch.arange(len(dataset))
    )
    source_ids = ordering[:source_size].tolist()
    validation_ids = ordering[source_size : source_size + val_size].tolist()
    count = cal.get("validation_count", val_size)
    if not 1 <= count <= val_size:
        raise ValueError("validation_count must be between 1 and validation_size")
    clean = torch.cat(list(images_from(Subset(dataset, validation_ids[:count]), 64)))
    calibration_model = make_model(kind, Subset(dataset, source_ids), config)
    shape = (metadata["num_channels"], metadata["image_size"], metadata["image_size"])
    # Calibration shares masks across generation subsets.
    calibration_config = {
        k: v
        for k, v in config.items()
        if k
        not in {"generation", "generate", "output_root", "num_subsets", "subset_size"}
    }
    provenance = dict(
        config=calibration_config,
        dataset_length=len(dataset),
        shape=list(shape),
        source_indices=source_ids,
        validation_indices=validation_ids[:count],
        times=times.tolist(),
    )
    with lock(calibration_dir / ".calibration.lock"):
        path = calibration_dir / "calibration.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state["provenance"] != provenance:
                raise ValueError(f"{path}: settings differ; choose a new output_root")
        else:
            state = dict(provenance=provenance, steps=[])
        covariance = None
        if kind == "wiener" and len(state["steps"]) < len(times):
            stat_path = calibration_dir / "statistics.pt"
            stat_config = dict(
                data=config.get("data", {}),
                dataset=config["dataset"],
                length=len(dataset),
                max_samples=cal.get("statistics_samples"),
            )
            if stat_path.exists():
                stats = torch.load(stat_path, map_location="cpu", weights_only=True)
                if stats["config"] != stat_config:
                    raise ValueError("Cached covariance settings differ")
            else:
                mean, cov = dataset_statistics(
                    dataset, device=device, max_samples=cal.get("statistics_samples")
                )
                stats = dict(mean=mean, covariance=cov, config=stat_config)
                atomic_torch_save(stats, stat_path)
            covariance = stats["covariance"]
        for i in range(len(state["steps"]), len(times)):
            t = times[i]
            noise = torch.randn(
                clean.shape, generator=torch.Generator().manual_seed(seed + 10000 + i)
            )
            a, b, _, _ = cosine(t.float())
            noisy = a * clean + b * noise
            previous = state["steps"][-1]["counts"] if state["steps"] else None
            step = calibrate_step(
                calibration_model,
                kind,
                clean,
                noisy,
                t,
                previous,
                covariance,
                minimum=cal.get("minimum", 10 if kind == "wiener" else 3),
                min_increment=cal.get("min_increment", 4),
                fraction_increment=cal.get("fraction_increment", 0.05),
                max_k=cal.get("max_k"),
                batch_size=cal.get("batch_size", 8),
                device=device,
            )
            state["steps"].append(step)
            atomic_torch_save(state, path)
            print(f"{kind} calibration {i+1}/{len(times)} t={float(t):.5f}", flush=True)
    if not config.get("generate", True):
        return state
    generation = config.get("generation", {})
    seed_file = Path(generation.get("seeds_file", saved / "seeds.pt")).expanduser()
    n = generation.get("num_samples", 100)
    with lock(seed_file.with_suffix(".lock")):
        if seed_file.exists():
            seeds = load_tensor(seed_file)
        else:
            seeds = torch.randn(
                (n,) + shape, generator=torch.Generator().manual_seed(seed)
            )
            atomic_torch_save(seeds, seed_file)
        if tuple(seeds.shape) != (n,) + shape:
            raise ValueError(
                f"{seed_file}: seeds must have shape {(n,)+shape}; never silently replace them"
            )
    template = generation.get(
        "filename_template",
        "subset{subset}.pt" if kind == "wiener" else "els_out_subset{subset}.pt",
    )
    outputs = {}
    for subset, indices in enumerate(splits, 1):
        out_path = saved / template.format(subset=subset)
        record = dict(
            analytical_model=kind,
            calibration_sha256=digest(path),
            seed_sha256=digest(seed_file),
            subset_indices=indices,
            method=generation.get("method", "euler"),
            times=times.tolist(),
        )
        meta_path = out_path.with_suffix(".json")
        with lock(out_path.with_suffix(".lock")):
            if out_path.exists():
                if (
                    not meta_path.exists()
                    or json.loads(meta_path.read_text()) != record
                ):
                    raise FileExistsError(
                        f"{out_path}: existing output has different/unknown provenance; choose another filename"
                    )
                outputs[subset] = load_tensor(out_path)
                continue
            model = make_model(kind, Subset(dataset, indices), config)
            progress_path = out_path.with_suffix(".progress.pt")
            if progress_path.exists():
                progress = torch.load(
                    progress_path, map_location="cpu", weights_only=True
                )
                if progress["record"] != record:
                    raise ValueError(f"{progress_path}: generation settings changed")
            else:
                progress = dict(record=record, parts=[])
            start = sum(len(part) for part in progress["parts"])
            batch_size = generation.get("batch_size", 8)
            for begin in range(start, len(seeds), batch_size):
                part = sample(
                    model,
                    seeds[begin : begin + batch_size],
                    times,
                    [s["parameter"] for s in state["steps"]],
                    method=generation.get("method", "euler"),
                    batch_size=batch_size,
                    device=device,
                )
                if not torch.isfinite(part).all():
                    raise ValueError("Nonfinite generated samples")
                progress["parts"].append(part)
                atomic_torch_save(progress, progress_path)
                print(
                    f"{kind} subset {subset}: {begin+len(part)}/{len(seeds)} samples",
                    flush=True,
                )
            result = torch.cat(progress["parts"])
            write_json(meta_path, record)
            atomic_torch_save(result, out_path)
            progress_path.unlink(missing_ok=True)
            outputs[subset] = result
            print(f"Saved {out_path}", flush=True)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))


if __name__ == "__main__":
    main()
