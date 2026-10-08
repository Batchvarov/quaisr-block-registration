# Nearest Neighbour Gaussian Process

This Block trains a scalable Gaussian Process model on 2D spatial data using the Nearest Neighbour GP (NNGP) variational approximation. The trained model can be passed to the **GP Inference** block to generate predictions and posterior samples over arbitrary grids.

## Model Overview

Standard GP inference is $O(N^3)$ in the number of training points, making it infeasible for large spatial datasets. This block uses the NNGP variational approximation, which factorises the GP prior over inducing points by conditioning each point on $k$ parents that precede it:

$$p(\mathbf{u}) \approx \prod_{i=1}^{N} p(u_i \mid \mathbf{u}_{N(i)})$$

This reduces the Cholesky cost to $O(Mk^2)$ in the number of inducing points $M$, and enables scalable stochastic training via mini-batch ELBO optimisation. The inducing points come from a regular grid masked to the data coverage, sized by the `ninducing` optimiser field.

The choice of parents decides which correlations the prior can hold, so it governs how far spatial structure carries. See [Vecchia Configuration](#vecchia-configuration).

### Mean Function

A linear mean is used:

$$\mu(x) = \mathbf{w}^\top x + b$$

### Kernel

The covariance is an additive sum of user-specified basis kernels:

$$k(x, x') = \sum_j k_j(x, x')$$

Supported kernels are RBF and Matérn ($\nu \in \{0.5, 1.5, 2.5\}$), each wrapped in a `ScaleKernel`. Lengthscale and output-scale constraints can be set per kernel.

If `kernels` is left unset, the block auto-constructs a multiscale additive kernel over three length scales:

| Component | Lengthscale bound | Outputscale bound |
| :--- | :--- | :--- |
| Matérn(2.5) | $\geq \ell_{\text{long}} = \frac{d_{\text{bbox}}}{4}$ | $[0.001, 2.0]$ |
| Matérn(1.5) | $\geq \ell_{\text{mid}} = \sqrt{\ell_{\text{long}} \ell_{\text{short}}}$ | $[0.001, 2.0]$ |
| Matérn(1.5) | $\geq \ell_{\text{short}} = s/4$ | $[0.001, 0.1]$ |

Here $d_{\text{bbox}}$ is the diagonal of the bounding box of the scaled training inputs, and $s$ is the median distance from an inducing point to its nearest neighbour. Every component is Matérn rather than RBF, every lengthscale starts at $1.5\times$ its own bound, and every outputscale starts at the midpoint of its interval.

The block writes $s$ and every component's bounds to `log`.

#### Amplitude modulation of the short component

The short component is stationary by default: one outputscale holds over the
whole survey. Set `modulation_control_count` in `kernel_options` to let its
amplitude vary in space instead:

$$\tilde k_{\text{short}}(x, x') = g(x)\, k_{\text{short}}(x, x')\, g(x')
\qquad
\log g(x) = \sum_c w_c \exp\!\left(-\frac{\lVert x - z_c \rVert^2}{2 h^2}\right)$$

The control points $z_c$ are an $n \times n$ grid over the training extent, with
$n$ = `modulation_control_count`, and $h$ is one cell of that grid. The weights
$w_c$ are trained, bounded to $[-a, a]$ with $a$ = `modulation_log_amplitude`,
and centred before use, so $g$ carries the contrast and the outputscale keeps
the overall level. The kernel stays positive definite for any $g$, because the
matrix is $D_g K D_g$.

Only the short component takes the field. The long and mid components carry the
regional trend, and a trend that fades in and out of regions is a different
model from the one this kernel states.

The control points are written into the checkpoint, so the GP Inference block
rebuilds the same field without seeing the training data.

### Training Objective

The model is trained by maximising the variational ELBO with a mean-field variational distribution over the inducing points:

$$\mathcal{L} = \mathbb{E}_{q(\mathbf{f})}[\log p(\mathbf{y} \mid \mathbf{f})] - \text{KL}[q(\mathbf{u}) \| p(\mathbf{u})]$$

Inputs are normalised (zero-mean, unit max-min range) and outputs are standardised (zero mean, unit variance) before training.

## Inputs

| Input | Type | Description |
| :--- | :--- | :--- |
| `training_samples` | Ordered Table | Training data. Must have a column `X` containing `[x, y]` coordinate pairs and a column `y` containing the scalar target value. |
| `test_samples` | Ordered Table | Hold-out data in the same format as `training_samples`, used for validation metrics. |
| `kernels` | Optional List of Records | Kernel definitions (see [Kernel Configuration](#kernel-configuration)). If omitted, a multiscale kernel is constructed automatically. |
| `kernel_options` | Optional Record | Options for that automatic kernel. See [Kernel Options](#kernel-options). Ignored when `kernels` is given. |
| `likelihood` | Record | Gaussian likelihood options. See [Likelihood Configuration](#likelihood-configuration). |
| `optimiser` | Record | Training hyperparameters. See [Optimiser Configuration](#optimiser-configuration). |
| `vecchia_options` | Optional Record | Structure of the nearest-neighbour DAG. See [Vecchia Configuration](#vecchia-configuration). If omitted, the hierarchical DAG is used. |
| `validation_options` | Optional Record | Point caps for the post-training metrics and plots. See [Validation Configuration](#validation-configuration). If omitted, both default to 100,000 points. |

### Kernel Configuration

Each element of `kernels` is a record of `kernel` (`"rbf"` or `"matern"`) and `options`:

| Key | Type | Description |
| :--- | :--- | :--- |
| `nu` | Float | Matérn only. Smoothness: `0.5`, `1.5` or `2.5`. |
| `ard_num_dims` | Integer | Number of dimensions for ARD lengthscales (typically `2`). |
| `lengthscale` | Constraint | Constraint on lengthscale, e.g. `["greaterthan", 1000]`. |
| `outputscale` | Constraint | Optional constraint on output scale. |

A constraint is `["greaterthan", v]`, `["lessthan", v]` or `["interval", [lo, hi]]`, with an optional third element giving the initial value: `["greaterthan", 1000, 1500]`. The initial value must lie strictly inside the bound, or the block raises. Omitted, it defaults to `v + softplus(0)` for `greaterthan`, half the bound for a positive `lessthan`, and the midpoint for `interval`.

Prefer `interval` for an outputscale. `lessthan` bounds one side only, so the fit can drive the outputscale negative and make the kernel matrix indefinite.

### Kernel Options

Options for the automatic kernel. They are read only when `kernels` is omitted:
a caller who passes `kernels` states every component in full.

| Field | Type | Description |
| :--- | :--- | :--- |
| `modulation_control_count` | Optional Integer | Cells across for the amplitude field on the short component. Omit or `0` to keep that component stationary. |
| `modulation_log_amplitude` | Optional Float | Symmetric bound on the log weight of each control point. Omit for `1.0`. The bumps overlap, so the amplitude contrast across the survey comes out wider than this bound. |

### Likelihood Configuration

| Field | Type | Description |
| :--- | :--- | :--- |
| `name` | String | Likelihood name (currently `"gaussian"`). |
| `noise` | Float | Initial noise level (noise variance is fixed during training). |

### Optimiser Configuration

| Field | Type | Description |
| :--- | :--- | :--- |
| `k` | Integer | Number of nearest neighbours used in the NNGP approximation. |
| `training_batch_size` | Integer | Mini-batch size for stochastic ELBO training. |
| `training_epochs` | Integer | Number of full passes over the training data. |
| `learning_rate` | Float | Initial Adam learning rate. |
| `milestones` | List of Integers | Epoch indices at which to reduce the learning rate by 10×. |
| `ninducing` | Integer | Target number of inducing points. They are laid out on a regular grid masked to the data coverage, so the realised count is at or below this. At or above the number of training points, every training point is used. |

### Vecchia Configuration

The NNGP prior conditions each inducing point on `k` parents, at a cost of $O(Mk^2)$. The hierarchical DAG spreads those parents across scales so the prior can hold long-range correlation; the flat DAG takes the `k` nearest predecessors, which is the better approximation when the correlation range is about one inducing spacing.

| Field | Type | Description |
| :--- | :--- | :--- |
| `use_hierarchical_vecchia` | Boolean | Use the multi-level DAG. Default `true`. Set `false` for the flat nearest-predecessor DAG. |
| `n_vecchia_levels` | Integer | Number of levels. Reduced automatically if the inducing count is too small. |
| `k_cross_ratio` | Float | Fraction of `k` spent on parents from coarser levels. The rest go to same-level neighbours. |
| `k_l0_fraction` | Float | Fraction of the cross-level budget reserved for the `L_0` skeleton. Raise it for longer-range kernels. |

The hierarchical DAG needs at least `2 * k` inducing points. Below that the block warns and falls back to a plain Hilbert sort.

### Validation Configuration

Validation metrics and plots run on the test set, diagnostics on the training set. Both are subsampled by default, because the metrics stop gaining accuracy well before the memory and time cost stops growing.

| Field | Type | Description |
| :--- | :--- | :--- |
| `max_validation_points` | Optional Integer | Cap on test points used for `validation_metrics` and `validation_plots`. Omit for the 100,000 default; set to `0` or less to use every test point. |
| `max_diagnostics_points` | Optional Integer | Cap on training points used for `diagnostics_metrics` and `diagnostics_plots`. Omit for the 100,000 default; set to `0` or less to use every training point. |

Subsampling is uniform without replacement at a fixed seed, so repeated runs on the same data pick the same points. The realised counts and the caps in force are written to `log`.

## Outputs

| Output | Type | Description |
| :--- | :--- | :--- |
| `loss_stream` | Stream of Floats | ELBO loss value after each mini-batch update, streamed during training. |
| `model_file` | File (`application/octet-stream`) | Serialised model state (`.pth`). Pass this to the **GP Inference** block. |
| `validation_metrics` | Dictionary of Floats | Metrics on the test set: `rmse`, `r2`, `rho`, `error_rate`. |
| `validation_plots` | Dictionary of HTML Files | Diagnostic plots on test data: `parity plot`, `error distribution`. |
| `diagnostics_metrics` | Dictionary of Floats | Same metrics computed on the training set. |
| `diagnostics_plots` | Dictionary of HTML Files | Same plots on training data, plus `convergence` (loss vs. iteration). |
| `time_left` | Stream of Integers | Streamed estimate of `[seconds_remaining, seconds_total]`, updated each epoch. |
| `log` | String | Training log. Contains error traces if training fails. |

### Validation Metrics

| Key | Description |
| :--- | :--- |
| `rmse` | Root mean squared error between posterior mean and measured values. |
| `r2` | Coefficient of determination. |
| `rho` | Slope of the linear regression of predicted on measured values. |
| `error_rate` | Fraction of measurements falling outside the 95% predictive interval. |

## When the block fails

The block checks the fit after training and writes no model file when the fit is
unusable. Two checks, both measured on a subsample:

- The predictive nugget is at or above half the variance of the training
  outputs. The fit misses more of the field than it explains.
- The `r2` on the training points is at or below -0.25. The fit is actively
  wrong on points it trained on. This one is a fallback: it applies when the test
  set is too small to measure the nugget on.

A failed check does not fail the run at once. The block retries training twice
from a new seed, because the order of the training mini-batches is the only
random part of a run, and a different order usually finds a different optimum.
The block fails only when every attempt is rejected.

The `log` output carries one line per attempt with the training `r2`, the
held-out `r2` and the nugget ratio, so a failure shows what each attempt reached.
The `time_left` estimate restarts on a retry.

A separate retry handles a NaN loss. That one raises the fixed likelihood noise
instead of reseeding, and the `noise_growth` and `max_noise_retries` fields on
the likelihood record control it.
