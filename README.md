# Neural and analytical diffusion models

Train diffusion transformers (DiT) and attention UNets, calibrate local analytical
denoisers, and compare their generated images throughout training.

The analytical models are **Wiener LS**, which selects spatial support using
Wiener-filter coefficients, and **boundary-broken ELS**, which uses square patches
with boundary-dependent translation sharing. Both compute empirical-posterior
denoising estimates from a dataset.

## Installation

Use Python 3.9 or newer and install a PyTorch build appropriate for the machine's
CPU or CUDA environment. From the repository root:

```bash
python -m pip install -e '.[notebooks]'
jupyter lab
```

The package requires PyTorch 2.6 or newer. File locking uses `fcntl`, so the
supported operating systems are Linux and macOS.

## Notebooks

Each notebook has its experiment and display configuration in the first cell.

| Notebook | Purpose |
| --- | --- |
| `01_Train.ipynb` | Train DiT or UNet on one or two disjoint subsets, with configurable repeated runs |
| `02_Calibrate_and_Generate.ipynb` | Calibrate Wiener LS or boundary-broken ELS and generate samples from shared noise |
| `03_Time_Course.ipynb` | Evaluate checkpoints and save per-image correlation/MSE trajectories and plots |
| `04_Results_Viewer.ipynb` | Display saved plots, numerical summaries, error estimates, and checkpoint configurations |
| `05_Checkpoint_Comparisons.ipynb` | Compare individual models, dataset subsets, training epochs, and architectures |

## Data

Supported datasets are MNIST, FashionMNIST, CIFAR10, CelebA32, and CelebA64.
MNIST, FashionMNIST, and CIFAR10 use torchvision's training split, resize to 32×32,
and normalize to [-1,1]. Set `data.download=True` to enable downloading.

For CelebA, supply a preprocessed `data.image_dir` or a normalized NCHW tensor via
`data.tensor_file`. The directory loader sorts relative paths, converts images to
RGB, resizes, and normalizes; apply the desired crop before loading. Custom datasets
can be supplied through `data.provider='module:function'`. The function must accept
`(name, root=...)` and return `(dataset, metadata)` with `image_size` and
`num_channels` in the metadata. Datasets and trained checkpoints are not bundled.

## Workflow

1. Configure the data and output directory in notebooks 01 and 02.
2. In notebook 01, select `model='dit'` or `'unet'`, set `subset_size`, and choose
   `num_subsets=1` or `2`. The default subset size is 10000; two subsets use
   consecutive blocks of indices. `run_suffixes=['']` trains one trial, while
   `['', '_b', '_c']` trains three independently seeded trials.
3. In notebook 02, select `analytical_model='wiener'` or `'els'`. Calibrate the
   model and generate samples for the selected subsets using a shared `seeds.pt`.
4. In notebook 03, select the datasets, architectures, subsets, and run suffixes
   to evaluate. Choose a shared analytical reference or one reference per subset.
   Evaluation saves each completed model/epoch and exports full-range and
   first-50-epoch plots.
5. Use notebook 04 to inspect saved results without repeating inference. Use
   notebook 05 for image comparisons; its cross-model defaults are UNet epoch 10
   and DiT epoch 15, averaged over up to five consecutive checkpoints.

The model definitions live in `diffusion_project/models/`, training code in
`diffusion_project/training.py`, and analytical implementations in
`diffusion_project/analytical/`.

## Command-line execution

Edit the JSON configurations in `configs/`, then run:

```bash
python -m diffusion_project.train_dit --config configs/training.json
python -m diffusion_project.train_unet --config configs/training.json
python -m diffusion_project.analytical.pipeline --config configs/analytical_wiener.json
python -m diffusion_project.analytical.pipeline --config configs/analytical_els.json
```

Training accepts `--suffix base`, `--suffix _b`, or another configured suffix to
run an individual trial. Trial seeds are tied to suffixes, including when jobs run
separately. Notebooks 01 and 02 can load a JSON configuration through `CONFIG_FILE`.

For batch evaluation:

```bash
MPLBACKEND=Agg BATCH_MODE=1 python scripts/run_notebook.py notebooks/03_Time_Course.ipynb
```

The runner writes scientific results to disk and leaves the input notebook
unmodified. Run and save the results viewer to retain embedded figures and tables.

## Saved outputs

With `output_root='./results'`:

- `model_checkpoints/<dataset>/<model>_10k_s<subset><suffix>/`: epoch checkpoints,
  experiment configuration, training losses, and sample previews.
- `calibration/<dataset>/<wiener|els>/`: calibration state and Wiener statistics.
- `saved_tensors/<dataset>/`: shared seeds, Wiener `subset1.pt` / `subset2.pt`,
  and ELS `els_out_subset1.pt` / `els_out_subset2.pt`, with generation metadata.
- `time_course/<dataset>/<model>/`: per-image trajectory arrays, summary and
  per-run CSVs, and missing-checkpoint reports.
- `time_course/plots/`: full-range and early-epoch correlation/MSE plots as PNG/PDF.

Training resumes individual runs, calibration resumes completed noise levels,
generation resumes completed sample batches, and evaluation resumes completed
model/epoch pairs. Use separate output directories for different experiments.

## Methods

[Methods](docs/methods.md) describes architectures, schedules, calibration, data
ordering, and metric definitions. See [acknowledgments](ACKNOWLEDGMENTS.md) for
related code. Distributed under the MIT license.
