# Methods and configuration

## Neural models

| Setting | DiT | UNet |
| --- | --- | --- |
| Architecture | patch size 4, hidden dimension 512, 12 blocks, 8 heads, MLP ratio 4 | channels 128/256/256/512, 2 residual blocks per scale, 8 attention heads at resolution 16 and in the bottleneck |
| Batch size | 32 | 32 |
| Epochs | 400 | 400 |
| Optimizer | AdamW, betas (0.9,0.95), weight decay 0.05 | same |
| Learning rate | 0.0002, epoch-wise cosine decay to 0.000001 | same |
| Diffusion | 1000-step discrete cosine schedule, s=0.008, v prediction | same |
| Sampling | deterministic 100-step DDIM, clean predictions clipped to [-1,1] | same |

Architecture and training parameters are configurable. The training loader uses
shuffle=True and drop_last=True. With 10000 samples and batch size 32, each epoch
contains 312 minibatches. Checkpoints contain model, optimizer, scheduler, AMP
scaler, RNG state, configuration, and subset indices. Sampling previews are
computed every ten epochs by default.

DiT uses PyTorch's default layer initialization; the optional custom initialization
method is not called by its constructor. UNet residual-output and attention-output
projections use zero initialization.

## Dataset subsets and repeatability

Subsets are consecutive disjoint index blocks of the configured size. Data order,
preprocessing, and dataset contents must remain fixed when resuming an experiment.
The image-directory loader sorts relative paths; tensor input preserves tensor
order. Tensor images must be normalized to [-1,1] and match the dataset dimensions.

Seeds are explicit and assigned per run suffix and subset. CPU restart equivalence
is tested with deterministic data and num_workers=0. Random transformations in
persistent worker processes can have RNG state that is not stored in a checkpoint.
Dataset configuration and indices are recorded; raw image contents are not hashed
at every restart.

## Analytical denoisers

Wiener LS ranks spatial locations by the sum of absolute Wiener-filter coefficients
across input and output channels. For each output location, the selected support
sets the distance used in an empirical posterior over dataset images. The posterior
mean of the source pixel at that same location is the denoised output. Covariance
is the population covariance; the filter is computed with a linear solve. At t=1,
ranking uses the covariance limit and the empirical posterior is uniform.

Boundary-broken ELS uses square patches. Interior locations share source candidates
across both spatial axes. Edge locations share along the edge at a fixed distance
from it. Corners retain their absolute location. Patch sizes greater than or equal
to the image size use same-position local patches. All channels share posterior
weights. Query and source candidates are processed in chunks with stable softmax
accumulation.

## Calibration and generation

Calibration compares denoising MSE on a source-disjoint validation split. Each
noise level uses a fixed corruption across candidate supports. The search starts
from the accepted support at the preceding noise level, increases support until
the error worsens, and retains the last accepted support. Wiener LS selects a
pixel count per output location; ELS selects a single odd patch side length.

The Wiener configuration uses 5000 source examples and 2000 validation examples.
Covariance uses the full dataset unless `statistics_samples` is set. Calibration
is shared across generation subsets. Consequently, calibration validation examples
are disjoint from the calibration source but are not held out from full-data
covariance estimation or necessarily from the later generation subsets.

The ELS example uses a randomized 3000/1000 source/validation split and evaluates
two validation images. Increase `validation_count` for a more stable estimate.
Changing source or validation counts changes the statistical experiment; chunk
sizes primarily control memory and floating-point rounding.

The analytical schedule is continuous cosine with s=0, evaluated at 20 ascending
times from 0.005 to 1. Generation performs 19 backward Euler updates and ends at
0.005 without an extra clean projection. `generation.method='ddim'` provides an
alternative using the same calibrated times. The neural schedule is configured
separately. Seeds are shared across generation subsets; generation metadata records
calibration and seed hashes, time points, integrator, and subset indices.

Dense covariance requires quadratic memory and cubic filter-solve work in the
number of scalar pixels. RGB 64×64 covariance occupies about 0.60 GB in float32,
with additional memory required for work buffers. ELS compares many candidate
patches. Use GPU execution for large calibration jobs.

## Evaluation

Correlation is signed per-image Pearson correlation, flattening all pixels and
channels. Constant-image correlations are NaN. Curves average each model's
per-image median correlation across runs. MSE is averaged over pixels, images,
and runs in normalized image units. Only displayed images are mapped to [0,1].

The error columns report `2 * pooled_per_image_std(ddof=1) / sqrt(n)`.
`ERROR_N` in evaluation and `CORR_ERROR_N` in the viewer set n for both metrics;
the default is 400. Actual finite image counts are reported separately. Six runs
with 100 images give 600 measurements, so select the denominator deliberately.
These descriptive error estimates are not confidence intervals for the median
and do not account for dependence between repeated seeds or trials.

References can be shared across runs/subsets or selected per subset. Wiener output
filenames are `subset1.pt` and `subset2.pt`; ELS outputs use
`els_out_subset1.pt` and `els_out_subset2.pt`. Reference samples must align with the
saved seeds image-for-image. Use a separate evaluation output directory for each
reference choice.

Checkpoint averaging uses the configured trailing epoch window. Missing results
remain NaN, with completion counts and missing-checkpoint reports. By default,
plots include an epoch only when every selected run has a valid metric.
