import os
import traceback
from os import PathLike
from typing import NamedTuple

import gpytorch
import torch
from quaisr import WireOutput

if torch.cuda.is_available():
    torch.backends.cuda.preferred_linalg_library("cusolver")

import numpy as np
import pandas as pd
import sklearn.pipeline
import sklearn.preprocessing
from nearest_neighbour_gaussian_process.kernel import (
    MIN_LENGTHSCALE_SPACINGS,
    PrecomputedInducing,
    construct_multiscale_kernel,
    describe_kernel,
    inducing_spacing,
    resolve_modulation,
)
from nearest_neighbour_gaussian_process.samples import (
    read_samples,
    resolve_nugget_control_count,
    resolve_point_cap,
    resolve_validation_batch_size,
    subsample,
)
from sklearn.preprocessing import StandardScaler
from upscaling_tools.calibration import (
    MIN_CALIBRATION_POINTS,
    cross_fitted_field_error_rate,
    fit_nugget_field,
)
from upscaling_tools.inducing import GridCountInducing
from upscaling_tools.kernels import clamp_keops_chunk, construct_kernel, control_grid
from upscaling_tools.nngp import (
    NaNTrainingError,
    construct_likelihood,
    model_topology,
    nngp_training,
)
from upscaling_tools.plotting import LineData, plot_lines, save_plot
from upscaling_tools.scaler import GlobalMinMaxScaler
from upscaling_tools.validation import nngp_validation

DEFAULT_VECCHIA_OPTIONS = {
    'use_hierarchical_vecchia': True,
    'n_vecchia_levels': 3,
    'k_cross_ratio': 0.5,
    'k_l0_fraction': 0.5,
}

# A training-set r2 at or below this means the fit is actively wrong on points it
# already saw. A backstop, not the main gate: it fires only when the held-out set
# is too small for the nugget ratio below, and it is set clearly negative on
# purpose. A model that cannot resolve the field shrinks towards its mean
# function and scores near zero, so a threshold at zero would fail an honest
# sparse fit -- and would still miss a collapsed run that happens to score just
# above it. Only a negative score says the fit is wrong rather than uninformative.
MIN_TRAINING_R2 = -0.25

# The predictive nugget over the variance of the training outputs, and the gate
# that does the work. At 1.0 the correction to the interval is the whole variance
# of the data and the fit explains nothing. 0.5 is where it explains less than it
# misses, which is where the two populations separate: measured over 17 runs on
# VAL_GRAV_VARIOUS.csv the healthy fits reached 0.389 and the collapsed ones
# started at 0.739. See docs/README.md.
MAX_NUGGET_RATIO = 0.5

# Retries after a fit fails the gate. Each one is a full training run.
MAX_FIT_RETRIES = 2

# Points per set for the health probe. Well above MIN_CALIBRATION_POINTS, and
# small enough that the probe costs a fraction of one training epoch.
PROBE_POINTS = 5_000

BASE_SEED = 0


class FitQualityError(RuntimeError):
    """Training converged, and the fit it converged to is not usable."""


class FitHealth(NamedTuple):
    """What the post-training probe measured, and which gate it failed.

    `nugget_ratio` is None when the held-out probe held fewer than
    MIN_CALIBRATION_POINTS, which is too few to fit a nugget on.
    """

    train_r2: float
    validation_r2: float
    nugget_ratio: float | None
    failure: str | None

    @property
    def ok(self) -> bool:
        return self.failure is None

    def describe(self) -> str:
        ratio = "n/a" if self.nugget_ratio is None else f"{self.nugget_ratio:.3f}"
        return (f"training r2 {self.train_r2:.3f}, held-out r2 "
                f"{self.validation_r2:.3f}, nugget/var(y) {ratio}")


def _resolve_noise_options(likelihood_opts: dict, log: WireOutput):
    """Read the NaN-recovery knobs off the likelihood record.

    Only an absent or None field takes the default; an explicit value is used as
    given, and a value that cannot work is reported rather than swapped in
    silence.
    """
    noise = likelihood_opts.get('noise')
    noise = 1e-4 if noise is None else float(noise)
    if noise <= 0:
        # construct_likelihood floors the noise at 1e-8: its Interval constraint
        # is degenerate at zero. Match that floor here so the saved noise is the
        # one the model trained with.
        log.write_string(f"noise={noise:g} is not positive; using 1e-8 instead")
        noise = 1e-8

    noise_growth = likelihood_opts.get('noise_growth')
    noise_growth = 10.0 if noise_growth is None else float(noise_growth)
    if noise_growth <= 1:
        log.write_string(
            f"noise_growth={noise_growth:g} would not raise the noise, so every "
            "retry would repeat the same run; using 10.0 instead")
        noise_growth = 10.0

    max_noise_retries = likelihood_opts.get('max_noise_retries')
    max_noise_retries = 5 if max_noise_retries is None else int(max_noise_retries)
    return noise, noise_growth, max(max_noise_retries, 0)


def _prepare_training_data(train_x, train_y):
    """Fit the input and output scalers and return the scaled float32 arrays."""
    input_scaler = sklearn.pipeline.Pipeline([
        ('tranform', sklearn.preprocessing.StandardScaler(with_std=False)),
        ('global_minmax', GlobalMinMaxScaler()),
    ])
    input_scaler.fit(train_x)
    # The sklearn pipeline returns float64 while the model trains in float32, so
    # narrow here: torch.tensor() on the float64 array would keep a second
    # full-size copy alive until construct_model's .to(float32).
    scaled_input = np.ascontiguousarray(
        input_scaler.transform(train_x), dtype=np.float32)

    output_scaler = StandardScaler()
    scaled_output = np.ascontiguousarray(
        output_scaler.fit_transform(train_y.reshape(-1, 1)), dtype=np.float32)
    return input_scaler, output_scaler, scaled_input, scaled_output


def _allocate_inducing(scaled_input, optimiser: dict, log: WireOutput):
    """Allocate the inducing set, then clamp k below the count actually returned.

    The coverage mask can push that count far below ninducing.
    """
    allocator = GridCountInducing(optimiser['ninducing'])
    inducing_points = allocator.initialise(torch.from_numpy(scaled_input))
    # Read after initialise(): the fallback flag is set during allocation.
    precomputed = PrecomputedInducing(inducing_points, allocator.returns_data_points)

    n_inducing_eff = len(inducing_points)
    if optimiser['k'] >= n_inducing_eff:
        log.write_string(
            f"k={optimiser['k']} is not below the {n_inducing_eff} inducing "
            f"points the allocator returned (ninducing={optimiser['ninducing']}); "
            f"lowering k to {n_inducing_eff - 1}")
        optimiser['k'] = n_inducing_eff - 1
    return precomputed


def _resolve_kernels(kernels, kernel_options, scaled_input, inducing_points,
                     log: WireOutput):
    """Use the given kernel records, or build the auto kernel from the spacing.

    Build it after the allocation: the short-scale bound comes from the realised
    inducing spacing, which the coverage mask can push well above the spacing
    implied by ninducing.

    kernel_options only reaches the auto kernel. A caller who passes `kernels`
    states every component in full, including any modulation.
    """
    if kernels is not None:
        # construct_kernel subscripts each entry; a record has no __getitem__.
        return [dict(kernel.__dict__) for kernel in kernels]

    spacing = inducing_spacing(inducing_points)
    min_lengthscale = MIN_LENGTHSCALE_SPACINGS * spacing
    modulation = resolve_modulation(kernel_options, scaled_input)
    _, kernels = construct_multiscale_kernel(
        scaled_input, min_lengthscale, modulation)
    log.write_string(
        f"Auto kernel: inducing spacing {spacing:g} (scaled units); "
        f"short-scale lengthscale floored at {min_lengthscale:g}. "
        f"Components: {describe_kernel(kernels)}")
    return kernels


def _validation_batch(trained_model, optimiser: dict, validation_options) -> int:
    """Points per posterior batch, derived from the dtype the model trains in."""
    dtype_bytes = 4 if trained_model.train_inputs.dtype == torch.float32 else 8
    return resolve_validation_batch_size(
        validation_options, optimiser['k'], dtype_bytes)


def _probe_fit(trained_model, scaled_input, train_y, scaled_test, test_y,
               output_scaler, optimiser, validation_options,
               y_variance) -> FitHealth:
    """Measure whether the fit is usable, on a subsample of each point set.

    Cheap enough to run after every training attempt: PROBE_POINTS points per
    set, and nngp_validation builds no figures when plots is off.
    """
    dtype = trained_model.train_inputs.dtype
    device = trained_model.train_inputs.device
    val_batch = _validation_batch(trained_model, optimiser, validation_options)

    def metrics(x, y, calibrate):
        return nngp_validation(
            trained_model,
            torch.tensor(x).to(dtype=dtype, device=device),
            torch.tensor(y).to(dtype=dtype, device=device),
            output_scaler=output_scaler,
            batch_size=val_batch,
            calibrate=calibrate,
            plots=False,
        )[0]

    probe_x, probe_y = subsample(scaled_input, train_y, max_points=PROBE_POINTS)
    train_metrics = metrics(probe_x, probe_y, calibrate=False)

    val_x, val_y = subsample(scaled_test, test_y, max_points=PROBE_POINTS)
    # The nugget needs held-out residuals and a set large enough to fit on.
    calibrate = len(val_y) >= MIN_CALIBRATION_POINTS
    val_metrics = metrics(val_x, val_y, calibrate) if len(val_y) else {}

    ratio = None
    if calibrate and y_variance > 0:
        ratio = float(val_metrics["predictive_nugget"] / y_variance)

    train_r2 = float(train_metrics["r2"])
    failure = None
    if ratio is not None and ratio >= MAX_NUGGET_RATIO:
        failure = (
            f"the predictive nugget is {ratio:.3f}x the variance of the training "
            f"outputs, at or above {MAX_NUGGET_RATIO:g}: the fit misses more of "
            "the field than it explains")
    elif train_r2 <= MIN_TRAINING_R2:
        failure = (
            f"training r2 {train_r2:.3f} is at or below {MIN_TRAINING_R2:g}: the "
            "fit is actively wrong on points it trained on")

    return FitHealth(
        train_r2, float(val_metrics.get("r2", float("nan"))), ratio, failure)


def _train_with_retries(scaled_input, scaled_output, kernels, likelihood_opts,
                        optimiser, vecchia_kwargs, inducing_allocator,
                        noise, noise_growth, max_noise_retries, probe_args,
                        loss_stream: WireOutput, time_left: WireOutput,
                        log: WireOutput):
    """Train the NNGP until it converges to a fit that passes the health gate.

    Two failure modes, two levers. A NaN run is retried with more fixed
    likelihood noise. A run that converges to a bad optimum is retried with a new
    seed instead: the noise sits inside the ELBO, so raising it there changes the
    fit and makes it worse.

    A reseed is enough because the minibatch order is the only randomness in a
    run -- the inducing grid and the hyperparameter starts are both
    deterministic -- and that order alone is what separated the healthy runs from
    the catastrophic one in the sweep in docs/README.md.
    """
    noise_retries = fit_retries = attempt = 0
    while True:
        torch.manual_seed(BASE_SEED + attempt)
        attempt += 1
        # Rebuild the likelihood every attempt: the noise value is baked into the
        # model at construction, so a retry with more noise needs a fresh
        # likelihood (and therefore a fresh model inside nngp_training).
        likelihood_object = construct_likelihood({**likelihood_opts, 'noise': noise})
        # Rebuild the kernel too. construct_model takes it by reference, so a
        # failed attempt leaves its hyperparameters wherever the diverging run
        # parked them and the retry would start from there, not from the priors.
        kernel_object = construct_kernel(kernels)
        try:
            # clamp_keops_chunk keeps the checkpoint chunk at or above
            # max_cholesky_size, below which KeOps switches off and the kernels
            # materialise a dense block.
            with gpytorch.settings.memory_efficient(True), \
                    gpytorch.beta_features.checkpoint_kernel(clamp_keops_chunk(500)):
                trained_model, losses = nngp_training(
                    torch.from_numpy(scaled_input),
                    torch.from_numpy(scaled_output).squeeze(-1),
                    kernel_object,
                    likelihood_object,
                    optimiser,
                    loss_stream=loss_stream,
                    time_left_stream=time_left,
                    mean_module=gpytorch.means.LinearMean(input_size=2),
                    inducing_points_allocator=inducing_allocator,
                    **vecchia_kwargs,
                )
        except NaNTrainingError as e:
            if noise_retries >= max_noise_retries:
                log.write_string(
                    f"NNGP training still NaN after {noise_retries} noise "
                    f"increases (noise={noise:g}): {e}")
                raise
            noise_retries += 1
            noise *= noise_growth
            log.write_string(
                f"NaN training ({e}); increasing fixed likelihood noise to "
                f"{noise:g} and retrying ({noise_retries}/{max_noise_retries})")
            continue
        except Exception:
            log.write_string(f"NNGP training failed: {traceback.format_exc()}")
            raise

        health = _probe_fit(trained_model, *probe_args)
        log.write_string(f"Attempt {attempt}: {health.describe()}")
        if health.ok:
            break
        if fit_retries >= MAX_FIT_RETRIES:
            log.write_string(
                f"Training converged to an unusable fit on every one of {attempt} "
                f"attempts. {health.failure}.")
            raise FitQualityError(
                f"{health.failure} ({health.describe()}). No checkpoint written.")
        fit_retries += 1
        log.write_string(
            f"Rejecting this fit: {health.failure}. Retrying from a new seed "
            f"({fit_retries}/{MAX_FIT_RETRIES}).")

    if noise >= 1.0:
        log.write_string(
            f"Fixed likelihood noise reached {noise:g} after {noise_retries} "
            "increases. The outputs are standardised to unit variance, so the "
            "noise is at or above the signal variance and the fit carries little "
            "signal.")
    else:
        log.write_string(
            f"Training converged with fixed likelihood noise {noise:g} "
            f"({noise_retries} noise increases, {fit_retries} rejected fits)")
    return trained_model, losses, noise


def _save_model(trained_model, train_x, train_y, kernels, likelihood_opts, noise,
                optimiser, vecchia_kwargs, input_scaler, output_scaler,
                predictive_nugget: float, predictive_nugget_source: str,
                predictive_nugget_field=None) -> str:
    """Write everything the inference block needs to rebuild this model."""
    mean = trained_model.mean_module
    mean_module_info = {"class": type(mean).__name__}
    if hasattr(mean, 'weights'):
        mean_module_info["input_size"] = mean.weights.shape[-2]

    savepath = 'results/model_state.pth'
    torch.save({
        "model" : trained_model.state_dict(),
        "train_x" : train_x,
        "train_y" : train_y,
        "kernel" : kernels,
        # The effective noise, not the requested one: construct_likelihood
        # derives the noise constraint from this value, and raw_noise is stored
        # relative to those bounds. Saving the requested noise after a retry
        # would make the inference block rebuild the constraint around the wrong
        # scale, and reloading raw_noise would silently give back the requested
        # noise instead of the trained one.
        "likelihood" : {**likelihood_opts, 'noise': noise},
        "likelihood_state" : trained_model.likelihood.state_dict(),
        "mean_state" : trained_model.mean_module.state_dict(),
        "mean_module" : mean_module_info,
        "optimiser" : optimiser,
        "vecchia" : vecchia_kwargs,
        # Inducing points, their order and the nearest-neighbour DAG. All
        # deterministic given train_x, so this is a cache, not new information:
        # it lets the inference block skip the KD-tree allocation, the Hilbert
        # sort, the Vecchia DAG and the M-k step nearest-neighbour build.
        "topology" : model_topology(trained_model),
        'input_scaler': input_scaler,
        'output_scaler': output_scaler,
        # Variance to add to the reported posterior variance, in data units.
        # The kernel cannot represent a measurement nugget or signal below the
        # resolvable floor, so both land in the residual and the interval comes
        # out too narrow. Fitted on the held-out test set, which is why this is
        # saved here rather than derived at inference time.
        'predictive_nugget': predictive_nugget,
        'predictive_nugget_source': predictive_nugget_source,
        # A nugget that varies in space, when the run asked for one. Absent on
        # every earlier checkpoint, and the scalar above stays the fallback, so
        # the inference block keeps working either way. Its control points are
        # in the model's SCALED input space, the space the residuals were
        # measured in.
        'predictive_nugget_field': predictive_nugget_field,
    }, savepath)
    return savepath


def _fit_nugget_field(calibration, scaled_input, count: int, log: WireOutput):
    """Fit one nugget per control point, or None when the block was not asked to.

    Off by default. The scalar nugget is what every existing checkpoint carries
    and what the reported interval has always been widened by, so turning a
    field on silently would change the uncertainty every current run reports.

    The control grid and the radius both come from the training extent, so the
    only choice the caller makes is how many cells across.
    """
    if not count or not calibration:
        return None

    control = control_grid(torch.from_numpy(scaled_input), count).numpy()
    spacing = float(min(scaled_input.max(axis=0) - scaled_input.min(axis=0))) / count
    field = fit_nugget_field(
        calibration["points"], calibration["residual"], calibration["variance"],
        control, radius=spacing)

    thin = int((field.counts < MIN_CALIBRATION_POINTS).sum())
    rate = cross_fitted_field_error_rate(
        calibration["points"], calibration["residual"], calibration["variance"],
        control, radius=spacing)
    log.write_string(
        f"Predictive nugget field on a {count}x{count} control grid "
        f"(radius {spacing:g} scaled): {len(field.values)} nodes, values "
        f"{field.values.min():g} to {field.values.max():g} against a survey-wide "
        f"{field.global_nugget:g}. {thin} nodes held fewer than "
        f"{MIN_CALIBRATION_POINTS} residuals and took the survey-wide value. "
        f"Cross-fitted error rate {rate:.3f}.")
    return field


def _resolve_predictive_nugget(validation_metrics, diagnostic_metrics,
                               n_validation: int, log: WireOutput):
    """Pick which residuals the reported interval is widened by.

    Held-out residuals are the right measure, but a caller can pass a test set
    too small to fit anything on. Training residuals are always available and,
    while the model fits them better, the gap is small whenever the inducing set
    is sparse -- which is the regime a production run is in, because the model
    cannot interpolate its own training points either. Measured on this block's
    two reference surveys the training residual understates the held-out one by
    about 1.3x when sparse, against 2.6x when every training point is an
    inducing point.
    """
    if n_validation >= MIN_CALIBRATION_POINTS:
        return validation_metrics["predictive_nugget"], "validation"

    nugget = diagnostic_metrics["predictive_nugget"]
    log.write_string(
        f"Only {n_validation} validation points, below the "
        f"{MIN_CALIBRATION_POINTS} needed to fit a predictive nugget on held-out "
        f"residuals; using the training residuals instead (nugget {nugget:g}). "
        "The model fits its training points better than new ones, so the "
        "reported interval is likely to stay a little too narrow.")
    return nugget, "diagnostics"


def _run_validation(trained_model, losses, scaled_input, train_y, scaled_test,
                    test_y, output_scaler, optimiser, validation_options,
                    log: WireOutput):
    """Metrics and plots on the held-out test set and on the training set."""
    model_dtype = trained_model.train_inputs.dtype
    model_device = trained_model.train_inputs.device
    val_batch = _validation_batch(trained_model, optimiser, validation_options)

    # Cap validation/diagnostics to bound memory/time. Both caps default to
    # MAX_VALIDATION_POINTS; validation_options can raise, lower or remove them.
    max_val = resolve_point_cap(validation_options, "max_validation_points")
    max_diag = resolve_point_cap(validation_options, "max_diagnostics_points")

    val_x, val_y = subsample(scaled_test, test_y, max_points=max_val)
    diag_x, diag_y = subsample(scaled_input, train_y, max_points=max_diag)

    log.write_string(
        f"Validation on {len(val_y):,}/{len(test_y):,} test points "
        f"(cap {'none' if max_val is None else f'{max_val:,}'}); "
        f"diagnostics on {len(diag_y):,}/{len(train_y):,} training points "
        f"(cap {'none' if max_diag is None else f'{max_diag:,}'}).")

    calibration = {}

    def validate(x, y, savepath, calibrate=False, calibration_out=None):
        return nngp_validation(
            trained_model,
            torch.tensor(x).to(dtype=model_dtype, device=model_device),
            torch.tensor(y).to(dtype=model_dtype, device=model_device),
            output_scaler=output_scaler,
            savepath=savepath,
            batch_size=val_batch,
            calibrate=calibrate,
            calibration_out=calibration_out,
        )

    validation_metrics, validation_plots = validate(
        val_x, val_y, "results/validation", calibrate=True,
        calibration_out=calibration)
    diagnostic_metrics, diagnostic_plots = validate(
        diag_x, diag_y, "results/diagnostics", calibrate=True)

    nugget, source = _resolve_predictive_nugget(
        validation_metrics, diagnostic_metrics, len(val_y), log)
    metrics = (validation_metrics if source == "validation"
               else diagnostic_metrics)
    log.write_string(
        f"Predictive nugget {nugget:g} (sd {nugget ** 0.5:g}) from the {source} "
        f"residuals brings the error rate from {metrics['error_rate']:.3f} to "
        f"{metrics['error_rate_calibrated']:.3f}. The reported posterior mean "
        "is unchanged.")

    line_data = LineData(
        "convergence",
        pd.DataFrame({"loss": losses, "iteration": list(range(0, len(losses)))}),
        "iteration",
        "loss")
    diagnostic_plots['convergence'] = save_plot(
        plot_lines([line_data]), "results/plots", "convergence")

    # The field is fitted on the held-out residuals only. Training residuals are
    # a usable fallback for one number, but a per-node fit on points the model
    # already saw would understate the correction node by node.
    field = _fit_nugget_field(
        calibration, scaled_input,
        resolve_nugget_control_count(validation_options), log)

    return (validation_metrics, validation_plots,
            diagnostic_metrics, diagnostic_plots, nugget, source, field)


def train_nngp(training_samples_file: PathLike, test_samples_file: PathLike, kernels, kernel_options, likelihood, optimiser, vecchia_options, validation_options, loss_stream: WireOutput, model_file: WireOutput, validation_metrics: WireOutput, validation_plots: WireOutput, diagnostics_metrics: WireOutput, diagnostics_plots: WireOutput, time_left: WireOutput, log: WireOutput):
    train_x, train_y = read_samples(training_samples_file)
    test_x, test_y = read_samples(test_samples_file)

    likelihood_opts = dict(likelihood.__dict__)
    optimiser = optimiser.__dict__
    noise, noise_growth, max_noise_retries = _resolve_noise_options(
        likelihood_opts, log)
    vecchia_kwargs = (DEFAULT_VECCHIA_OPTIONS if vecchia_options is None
                      else vecchia_options.__dict__)

    input_scaler, output_scaler, scaled_input, scaled_output = _prepare_training_data(
        train_x, train_y)
    inducing_allocator = _allocate_inducing(scaled_input, optimiser, log)
    kernels = _resolve_kernels(
        kernels, kernel_options, scaled_input, inducing_allocator.points, log)

    # Scaled once, for the health probe after every training attempt and for the
    # validation pass at the end.
    scaled_test = input_scaler.transform(test_x)
    probe_args = (scaled_input, train_y, scaled_test, test_y, output_scaler,
                  optimiser, validation_options, float(np.var(train_y)))

    trained_model, losses, noise = _train_with_retries(
        scaled_input, scaled_output, kernels, likelihood_opts, optimiser,
        vecchia_kwargs, inducing_allocator, noise, noise_growth,
        max_noise_retries, probe_args, loss_stream, time_left, log)

    os.makedirs("results", exist_ok=True)

    # Validate first: the predictive nugget is fitted on the held-out residuals
    # and has to be in the file the inference block loads.
    (val_metrics, val_plots, diag_metrics, diag_plots,
     nugget, nugget_source, nugget_field) = _run_validation(
        trained_model, losses, scaled_input, train_y, scaled_test, test_y,
        output_scaler, optimiser, validation_options, log)

    model_file.write_file(_save_model(
        trained_model, train_x, train_y, kernels, likelihood_opts, noise,
        optimiser, vecchia_kwargs, input_scaler, output_scaler,
        nugget, nugget_source, nugget_field))

    validation_metrics.write_dict(val_metrics)
    validation_plots.write_dict(val_plots)
    diagnostics_metrics.write_dict(diag_metrics)
    diagnostics_plots.write_dict(diag_plots)

    log.write_string("Model training completed")
