"""Score the auto kernel against the wavelengths present in the data.

The auto kernel's ladder and outputscale caps were chosen against ELBO loss and
held-out error. Neither measures whether the trained kernel's correlation
function matches the field's, which is what the model is for. This harness
measures that.

It trains one model per configuration and reports, for each, the measured
correlation function of the data against the trained kernel's own, the lag at
which each falls to 0.5 and to 0.1, and a scalar misfit between them. ELBO loss
and held-out error are reported alongside but do not decide anything.

Usage:

    OMP_NUM_THREADS=1 python docs/harness.py \
        --train train.parquet --test test.parquet \
        --ninducing 100000 --epochs 250 --out results.json

OMP_NUM_THREADS=1 is required on macOS: faiss and torch both load libomp, and
the first ELBO step segfaults otherwise.

The parquet files are the block's own input schema, {X: list<float>, y: float},
as written by the dataframe_to_samples block.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

# core.py imports quaisr for its WireOutput type, which is only installed inside
# the block image. The harness never writes a wire, so a stub is enough to reach
# the ladder helpers without standing up the whole block runtime.
if "quaisr" not in sys.modules:
    try:
        import quaisr  # noqa: F401
    except ImportError:
        stub = types.ModuleType("quaisr")
        stub.WireOutput = object
        sys.modules["quaisr"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gpytorch  # noqa: E402
import sklearn.pipeline  # noqa: E402
import sklearn.preprocessing  # noqa: E402

from nearest_neighbour_gaussian_process.kernel import (  # noqa: E402
    MAX_OUTPUTSCALE,
    MAX_ROUGH_OUTPUTSCALE,
    MIN_LENGTHSCALE_SPACINGS,
    MIN_OUTPUTSCALE,
    PrecomputedInducing,
    construct_multiscale_kernel,
    inducing_spacing,
    lengthscale_at,
    outputscale_upto,
)
from nearest_neighbour_gaussian_process.samples import read_samples  # noqa: E402
from upscaling_tools.inducing import GridCountInducing  # noqa: E402
from upscaling_tools.kernels import clamp_keops_chunk, construct_kernel  # noqa: E402
from upscaling_tools.nngp import construct_likelihood, nngp_training  # noqa: E402
from upscaling_tools.nngp_fft_prior import GridSpec  # noqa: E402
from upscaling_tools.scaler import GlobalMinMaxScaler  # noqa: E402
from upscaling_tools.spectral import (  # noqa: E402
    bootstrap_ladder,
    correlation_range,
    empirical_correlation,
    fit_ladder,
    kernel_correlation,
    spectral_misfit,
)
from upscaling_tools.validation import compute_interp_perf  # noqa: E402


def prepare(train_path, test_path, ninducing, k):
    """Reproduce train_nngp's scaling and inducing allocation, up to the kernel."""
    train_x, train_y = read_samples(train_path)
    test_x, test_y = read_samples(test_path)

    input_scaler = sklearn.pipeline.Pipeline([
        ('tranform', sklearn.preprocessing.StandardScaler(with_std=False)),
        ('global_minmax', GlobalMinMaxScaler()),
    ])
    input_scaler.fit(train_x)
    scaled_input = np.ascontiguousarray(input_scaler.transform(train_x), dtype=np.float32)

    output_scaler = sklearn.preprocessing.StandardScaler()
    scaled_output = np.ascontiguousarray(
        output_scaler.fit_transform(train_y.reshape(-1, 1)), dtype=np.float32)

    allocator = GridCountInducing(ninducing)
    inducing_points = allocator.initialise(torch.from_numpy(scaled_input))
    allocator = PrecomputedInducing(inducing_points, allocator.returns_data_points)

    spacing = inducing_spacing(inducing_points)
    return {
        "scaled_input": scaled_input,
        "scaled_output": scaled_output,
        "input_scaler": input_scaler,
        "output_scaler": output_scaler,
        "allocator": allocator,
        "n_inducing": len(inducing_points),
        "k": min(k, len(inducing_points) - 1),
        "spacing": spacing,
        "min_lengthscale": MIN_LENGTHSCALE_SPACINGS * spacing,
        "bbox_diagonal": float(np.linalg.norm(
            scaled_input.max(axis=0) - scaled_input.min(axis=0))),
        "test_x": np.ascontiguousarray(input_scaler.transform(test_x), dtype=np.float32),
        "test_y": test_y,
        "train_y": train_y,
    }


def measure_data(setup, n_pairs, n_bins, seed):
    """Correlation function of the training residual, in scaled units."""
    return empirical_correlation(
        setup["scaled_input"].astype(np.float64),
        setup["scaled_output"].astype(np.float64),
        min_lag=setup["min_lengthscale"],
        max_lag=setup["bbox_diagonal"],
        n_pairs=n_pairs,
        n_bins=n_bins,
        seed=seed,
    )


def config_a(setup, curve, ladder, args):
    """The shipped ladder: bbox_diagonal / 4 down a geometric mean, hardcoded caps."""
    _, kernels = construct_multiscale_kernel(setup["scaled_input"], setup["min_lengthscale"])
    return kernels


def config_b(setup, curve, ladder, args):
    """The previously rejected ladder: long scale from the measured range."""
    short = setup["min_lengthscale"]
    long = correlation_range(curve.lags, curve.rho, 0.1)
    if not np.isfinite(long):
        long = setup["bbox_diagonal"] / 4
    long = float(np.clip(long, short, setup["bbox_diagonal"]))
    mid = float(np.sqrt(long * short))

    scales = (long, mid, short)
    caps = (MAX_OUTPUTSCALE, MAX_OUTPUTSCALE, MAX_ROUGH_OUTPUTSCALE)
    return [
        {
            'kernel': 'matern',
            'options': {
                'nu': nu,
                'ard_num_dims': 1,
                'lengthscale': lengthscale_at(scale),
                'outputscale': outputscale_upto(cap),
                'keops': True,
            },
        }
        for nu, scale, cap in zip((2.5, 1.5, 1.5), scales, caps)
    ]


def _interval_around(value, low, high):
    """An interval bound with value strictly inside it.

    parse_constraint rejects an initial value sitting on a bound, because it
    inverts to a non-finite raw parameter. Widen by a hair rather than fail when
    the fit lands on the resolvable floor, which it routinely does.
    """
    low, high = min(low, value), max(high, value)
    if not low < value:
        low = value * 0.99
    if not value < high:
        high = value * 1.01
    return ['interval', [low, high], value]


def config_c(setup, curve, ladder, args):
    """Ladder and variance budget both fitted to the measured correlation.

    Both bounds are intervals, not floors. Every floor in the shipped kernel
    binds or is fled; an interval holds each component at the wavelength the
    data puts it at.

    The lengthscale band can come from the data. A block bootstrap reproduces
    each lengthscale to within a factor of its own, so --band-sigma sets each
    component's band from its own spread.

    The outputscale band cannot, and is a fixed factor around the fitted split
    whichever way the lengthscales are bounded. The bootstrap shows the split is
    not identified: components at adjacent lengthscales are close to collinear,
    so replicates shuffle variance between them while tracing the same total
    curve. Bounding only the total and letting the fit choose the split was
    tried and is worse -- it hands the choice to the ELBO, which is monotone in
    the carrying component's outputscale, and the fit parks the variance on the
    smoothest component it is still allowed. At ninducing 12,000 that raised the
    misfit from 0.022 to 0.095. Constraining the split works because it
    overrides the ELBO, not because the data pins it.
    """
    uncertainty = setup.get("uncertainty")

    kernels = []
    for rank, (lengthscale, variance, nu) in enumerate(
            zip(ladder.lengthscales, ladder.variances, ladder.nus)):
        lengthscale = float(max(lengthscale, setup["min_lengthscale"]))
        if uncertainty is not None:
            factor = float(np.exp(args.band_sigma * uncertainty.lengthscale_log_sd[rank]))
        else:
            factor = args.band
        low = max(lengthscale / factor, setup["min_lengthscale"] * 0.99)
        high = min(lengthscale * factor, setup["bbox_diagonal"])

        variance = float(np.clip(variance, 2 * MIN_OUTPUTSCALE, MAX_OUTPUTSCALE))
        outputscale = _interval_around(
            variance,
            max(variance / args.band, MIN_OUTPUTSCALE),
            min(variance * args.band, MAX_OUTPUTSCALE))

        kernels.append({
            'kernel': 'matern',
            'options': {
                'nu': float(nu),
                'ard_num_dims': 1,
                'lengthscale': _interval_around(lengthscale, low, high),
                'outputscale': outputscale,
                'keops': True,
            },
        })
    return kernels


CONFIGS = {"A": config_a, "B": config_b, "C": config_c}


def trained_components(kernel):
    """Lengthscale, outputscale and nu of every component, after training."""
    components = []
    for scale_kernel in kernel.kernels:
        components.append({
            "nu": float(scale_kernel.base_kernel.nu),
            "lengthscale": float(scale_kernel.base_kernel.lengthscale.detach().reshape(-1)[0]),
            "outputscale": float(scale_kernel.outputscale.detach()),
        })
    total = sum(component["outputscale"] for component in components)
    for component in components:
        component["variance_share"] = component["outputscale"] / total if total else float("nan")
    return components


def sample_range(model, setup, n_samples, grid_points, chunk):
    """Min, max and standard deviation of realisations, in the data's own units.

    Exercises the FFT prior that production uses, so the numbers are comparable
    to the range MAX_OUTPUTSCALE was introduced to protect.
    """
    mins = setup["scaled_input"].min(axis=0).tolist()
    maxs = setup["scaled_input"].max(axis=0).tolist()
    grid = GridSpec(mins=mins, maxs=maxs, shape=(grid_points, grid_points))

    axes = [torch.linspace(lo, hi, grid_points, dtype=torch.float32)
            for lo, hi in zip(mins, maxs)]
    xs = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)

    samples = model.sample_gp(
        xs, n_samples, output_scaler=setup["output_scaler"],
        output_grid=grid, batch_size=chunk).values
    fields = samples.T.reshape(n_samples, grid_points, grid_points)
    return fields, {
        "n_samples": n_samples,
        "grid_points": grid_points,
        "sample_min": float(np.min(samples)),
        "sample_max": float(np.max(samples)),
        "sample_std": float(np.std(samples)),
        "observed_min": float(np.min(setup["train_y"])),
        "observed_max": float(np.max(setup["train_y"])),
        "observed_std": float(np.std(setup["train_y"])),
    }


def run_config(name, setup, curve, ladder, args):
    kernels = CONFIGS[name](setup, curve, ladder, args)
    kernel = construct_kernel(kernels)
    likelihood = construct_likelihood({"name": "gaussian", "noise": args.noise})

    optimiser = {
        "k": setup["k"],
        "training_batch_size": args.batch_size,
        "training_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "milestones": args.milestones,
        "ninducing": args.ninducing,
    }

    with gpytorch.settings.memory_efficient(True), \
            gpytorch.beta_features.checkpoint_kernel(clamp_keops_chunk(500)):
        model, losses = nngp_training(
            torch.from_numpy(setup["scaled_input"]),
            torch.from_numpy(setup["scaled_output"]).squeeze(-1),
            kernel,
            likelihood,
            optimiser,
            mean_module=gpytorch.means.LinearMean(input_size=2),
            inducing_points_allocator=setup["allocator"],
            use_hierarchical_vecchia=True,
            n_vecchia_levels=3,
            k_cross_ratio=0.5,
            k_l0_fraction=0.5,
        )

    rho_model = kernel_correlation(kernel, curve.lags)
    predicted = model.inference(
        torch.from_numpy(setup["test_x"]).to(model.train_inputs.device),
        output_scaler=setup["output_scaler"],
        batch_size=args.validation_batch_size)
    predicted["measured"] = setup["test_y"]

    result = {
        "config": name,
        "kernels": kernels,
        "trained": trained_components(kernel),
        "lags": curve.lags.tolist(),
        "rho_data": curve.rho.tolist(),
        "rho_model": rho_model.tolist(),
        "spectral_misfit": spectral_misfit(curve.lags, curve.rho, rho_model),
        "range_data_0.5": correlation_range(curve.lags, curve.rho, 0.5),
        "range_data_0.1": correlation_range(curve.lags, curve.rho, 0.1),
        "range_model_0.5": correlation_range(curve.lags, rho_model, 0.5),
        "range_model_0.1": correlation_range(curve.lags, rho_model, 0.1),
        # Recorded, not used to decide. The ladder was previously rejected on
        # exactly these numbers.
        "final_loss": float(losses[-1]) if len(losses) else float("nan"),
        "held_out": compute_interp_perf(predicted, "measured", "GP_mean"),
    }
    fields = None
    if args.sample_grid:
        fields, result["samples"] = sample_range(
            model, setup, args.n_samples, args.sample_grid, args.validation_batch_size)
    return result, fields


def summarise(results, curve, ladder, setup):
    lines = [
        f"inducing points   {setup['n_inducing']:,} (spacing {setup['spacing']:.6g} scaled)",
        f"min lengthscale   {setup['min_lengthscale']:.6g} scaled",
        f"bbox diagonal     {setup['bbox_diagonal']:.6g} scaled",
        f"data range 0.5    {correlation_range(curve.lags, curve.rho, 0.5):.6g}",
        f"data range 0.1    {correlation_range(curve.lags, curve.rho, 0.1):.6g}",
        f"fitted ladder     {np.array2string(ladder.lengthscales, precision=5)}",
        f"fitted variances  {np.array2string(ladder.variances, precision=4)}",
        f"unrepresentable   {ladder.unrepresentable:.3f}",
        "",
        f"{'config':<8}{'misfit':>10}{'model 0.5':>12}{'model 0.1':>12}"
        f"{'loss':>10}{'rmse':>10}{'r2':>8}",
    ]
    for result in results:
        lines.append(
            f"{result['config']:<8}{result['spectral_misfit']:>10.4f}"
            f"{result['range_model_0.5']:>12.5g}{result['range_model_0.1']:>12.5g}"
            f"{result['final_loss']:>10.1f}{result['held_out']['rmse']:>10.3f}"
            f"{result['held_out']['r2']:>8.3f}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--configs", nargs="+", default=["A", "B", "C"], choices=list(CONFIGS))
    parser.add_argument("--ninducing", type=int, default=100_000)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--milestones", type=int, nargs="*", default=[100, 200])
    parser.add_argument("--noise", type=float, default=1e-4)
    parser.add_argument("--validation-batch-size", type=int, default=8192)
    parser.add_argument("--n-pairs", type=int, default=1_000_000)
    parser.add_argument("--n-bins", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--band", type=float, default=1.5,
        help="Fixed half-width factor for config C's bounds. Ignored when "
             "--band-sigma is given.")
    parser.add_argument(
        "--band-sigma", type=float, default=None,
        help="Set config C's LENGTHSCALE bounds from a spatial block bootstrap "
             "instead of a fixed factor: each gets exp(SIGMA * its own log sd). "
             "Outputscales keep the --band factor either way, because the "
             "bootstrap shows the variance split is not identified by the data.")
    parser.add_argument("--bootstrap-replicates", type=int, default=40)
    parser.add_argument("--bootstrap-blocks", type=int, default=8)
    parser.add_argument(
        "--sample-grid", type=int, default=0,
        help="Side of the square grid to draw realisations on. 0 skips sampling.")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--out", default="harness_results.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    setup = prepare(args.train, args.test, args.ninducing, args.k)
    curve = measure_data(setup, args.n_pairs, args.n_bins, args.seed)
    ladder = fit_ladder(curve, setup["min_lengthscale"], setup["bbox_diagonal"])

    if args.band_sigma is not None:
        setup["uncertainty"] = bootstrap_ladder(
            setup["scaled_input"].astype(np.float64),
            setup["scaled_output"].astype(np.float64),
            min_lengthscale=setup["min_lengthscale"],
            max_lengthscale=setup["bbox_diagonal"],
            min_lag=setup["min_lengthscale"],
            max_lag=setup["bbox_diagonal"],
            n_blocks=args.bootstrap_blocks,
            n_replicates=args.bootstrap_replicates,
            seed=args.seed,
            n_pairs=args.n_pairs,
            n_bins=args.n_bins,
        )
        u = setup["uncertainty"]
        print(f"bootstrap ({u.n_replicates} replicates): lengthscale bands "
              f"{np.array2string(np.exp(args.band_sigma * u.lengthscale_log_sd), precision=2)}"
              f", total variance {u.total_variance:.3f}")

    results, fields = [], {}
    for name in args.configs:
        result, drawn = run_config(name, setup, curve, ladder, args)
        results.append(result)
        if drawn is not None:
            fields[name] = drawn

    if fields:
        np.savez_compressed(
            Path(args.out).with_suffix(".samples.npz"),
            mins=setup["scaled_input"].min(axis=0),
            maxs=setup["scaled_input"].max(axis=0),
            **fields)

    payload = {
        "arguments": vars(args),
        "setup": {key: setup[key] for key in
                  ("n_inducing", "k", "spacing", "min_lengthscale", "bbox_diagonal")},
        "data_curve": {
            "lags": curve.lags.tolist(),
            "rho": curve.rho.tolist(),
            "counts": curve.counts.tolist(),
            "variance": curve.variance,
        },
        "uncertainty": None if setup.get("uncertainty") is None else {
            "lengthscale_log_sd": setup["uncertainty"].lengthscale_log_sd.tolist(),
            "total_variance": setup["uncertainty"].total_variance,
            "total_variance_log_sd": setup["uncertainty"].total_variance_log_sd,
            "n_replicates": setup["uncertainty"].n_replicates,
        },
        "fitted_ladder": {
            "lengthscales": ladder.lengthscales.tolist(),
            "variances": ladder.variances.tolist(),
            "nus": [float(nu) for nu in ladder.nus],
            "unrepresentable": ladder.unrepresentable,
            "misfit": ladder.misfit,
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(summarise(results, curve, ladder, setup))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
