"""Measure whether the short-scale amplitude varies across the survey.

The auto kernel is stationary: one lengthscale and one output scale for the
whole domain (`kernel.py`), and one `predictive_nugget` for the whole domain
(`core.py`). Both assume the field is equally rough everywhere. This script
measures that assumption rather than arguing about it.

It splits the domain into tiles and reports, per tile:

- the variance held below one inducing spacing, in absolute units -- the
  quantity the nugget replaces with a single number;
- a fitted Matern ladder, so a moving amplitude can be told apart from a moving
  lengthscale;
- with --model, the held-out coverage and the nugget the tile would have been
  given on its own.

Usage:

    OMP_NUM_THREADS=1 python docs/tile_sweep.py \
        --train train.parquet --test test.parquet \
        --ninducing 12000 --tiles 3 4 5 --out sweep.json

    OMP_NUM_THREADS=1 python docs/tile_sweep.py ... --shuffle    # null test

OMP_NUM_THREADS=1 is required on macOS, for the reason docs/harness.py gives.

Read the maps in this order. If `band_variance` is flat, the premise fails and
nothing below it matters. If it varies while `short_lengthscale` holds still,
the amplitude moves and the short component wants a spatially varying scale. If
the lengthscale moves too, a warp is needed instead, and this script does not
settle which.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.spatial

# harness.py sits beside this file. Importing it also installs the quaisr stub
# and puts the block package on sys.path, so the imports below resolve.
from harness import prepare  # noqa: E402

from upscaling_tools.calibration import (  # noqa: E402
    MIN_CALIBRATION_POINTS,
    coverage,
    fit_predictive_nugget,
)
from upscaling_tools.spectral import (  # noqa: E402
    MIN_BIN_PAIRS,
    bootstrap_ladder,
    detrend,
    empirical_correlation,
    fit_ladder,
)
from upscaling_tools.validation import compute_interp_perf  # noqa: E402


def tile_index(points: np.ndarray, count: int) -> np.ndarray:
    """Tile id per point, on a count x count grid over the point bounding box.

    Same construction as the one inside `bootstrap_ladder`, which is nested and
    so cannot be imported.
    """
    ids = np.zeros(len(points), dtype=int)
    for axis in range(points.shape[1]):
        column = points[:, axis]
        spread = np.ptp(column)
        scaled = (column - column.min()) / (spread if spread > 0 else 1.0)
        ids = ids * count + np.clip((scaled * count).astype(int), 0, count - 1)
    return ids


def band_semivariance(points, values, low, high, n_pairs, rng):
    """Variance held below `high`, from pairs drawn in the [low, high) annulus.

    Returns the semivariogram at that lag band in the values' own units, which
    is `variance * (1 - rho)`. Absolute, not a fraction of the tile: a quiet
    tile and a rough one are the comparison this whole script exists to make,
    and `empirical_correlation` divides that difference out.

    The lag band is fixed across tiles. A nearest-neighbour difference would
    instead measure each tile at its own point spacing, so a densely sampled
    tile would look smooth for a reason that has nothing to do with the field.

    The pair geometry copies `empirical_correlation`: step a random angle and a
    radius uniform in area, then take the nearest real point to where that
    lands, and keep the pair only if its realised lag is still inside the band.
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    values = np.ascontiguousarray(values, dtype=np.float64).ravel()
    residual = detrend(points, values)
    variance = float(residual.var())
    if variance <= 0 or len(points) < 2:
        return float("nan"), 0, variance

    tree = scipy.spatial.cKDTree(points)
    left = rng.integers(0, len(points), n_pairs)
    radius = np.sqrt(rng.uniform(low ** 2, high ** 2, n_pairs))
    angle = rng.uniform(0.0, 2.0 * np.pi, n_pairs)
    targets = points[left] + np.column_stack(
        [radius * np.cos(angle), radius * np.sin(angle)])

    _, right = tree.query(targets, workers=-1)
    lag = np.linalg.norm(points[left] - points[right], axis=1)
    keep = (right != left) & (lag >= low) & (lag < high)
    if keep.sum() < MIN_BIN_PAIRS:
        return float("nan"), int(keep.sum()), variance

    rho = float((residual[left[keep]] * residual[right[keep]]).mean() / variance)
    return variance * (1.0 - rho), int(keep.sum()), variance


def point_spacing(points) -> float:
    """Median distance from a point to its nearest neighbour."""
    if len(points) < 2:
        return float("nan")
    tree = scipy.spatial.cKDTree(np.ascontiguousarray(points))
    distances, _ = tree.query(points, k=2, workers=-1)
    return float(np.median(distances[:, 1]))


def tile_ladder(points, values, min_lengthscale, max_lengthscale, args):
    """Fitted ladder for one tile, or None when the tile holds too few pairs."""
    try:
        curve = empirical_correlation(
            points, values,
            min_lag=min_lengthscale,
            max_lag=max_lengthscale,
            n_pairs=args.n_pairs,
            n_bins=args.n_bins,
            seed=args.seed,
        )
    except ValueError:
        return None
    ladder = fit_ladder(curve, min_lengthscale, max_lengthscale)
    # variances are fractions of this tile's own variance. Multiply back, or a
    # uniformly quiet tile reads the same as a rough one.
    absolute = ladder.variances * curve.variance
    return {
        "lengthscales": ladder.lengthscales.tolist(),
        "variance_fractions": ladder.variances.tolist(),
        "variances": absolute.tolist(),
        "short_lengthscale": float(ladder.lengthscales[-1]),
        "short_variance": float(absolute[-1]),
        "unrepresentable_fraction": ladder.unrepresentable,
        "unrepresentable": ladder.unrepresentable * curve.variance,
        "misfit": ladder.misfit,
    }


def sweep_field(setup, values, count, args):
    """Every field measurement, tile by tile, at one tile count."""
    points = setup["scaled_input"].astype(np.float64)
    ids = tile_index(points, count)
    extent = points.max(axis=0) - points.min(axis=0)
    # A tile cannot measure a lag longer than itself. Half its short side is the
    # longest lag its pairs can cover in every direction.
    max_lengthscale = float(min(extent) / count / 2)
    rng = np.random.default_rng(args.seed)

    tiles = []
    for tile in range(count * count):
        mask = ids == tile
        if mask.sum() < args.min_tile_points:
            tiles.append({"tile": tile, "n": int(mask.sum()), "skipped": True})
            continue
        gamma, pairs, variance = band_semivariance(
            points[mask], values[mask],
            low=args.band_low or setup["min_lengthscale"],
            high=args.band_high or setup["spacing"],
            n_pairs=args.n_pairs // args.n_bins,
            rng=rng)
        tiles.append({
            "tile": tile,
            "n": int(mask.sum()),
            "skipped": False,
            "centre": points[mask].mean(axis=0).tolist(),
            "point_spacing": point_spacing(points[mask]),
            "variance": variance,
            "band_variance": gamma,
            "band_pairs": pairs,
            "ladder": tile_ladder(
                points[mask], values[mask].astype(np.float64),
                setup["min_lengthscale"], max_lengthscale, args),
        })
    return {"count": count, "max_lengthscale": max_lengthscale, "tiles": tiles}


def sweep_model(setup, predicted, nugget, count, args):
    """Held-out coverage and the nugget each tile would fit for itself."""
    points = np.ascontiguousarray(setup["test_x"], dtype=np.float64)
    ids = tile_index(points, count)
    residual = np.asarray(setup["test_y"], dtype=float) - predicted["GP_mean"].values
    variance = predicted["GP_variance"].values

    tiles = []
    for tile in range(count * count):
        mask = ids == tile
        if mask.sum() < MIN_CALIBRATION_POINTS:
            tiles.append({"tile": tile, "n": int(mask.sum()), "skipped": True})
            continue
        frame = predicted[mask].copy()
        frame["measured"] = np.asarray(setup["test_y"], dtype=float)[mask]
        tiles.append({
            "tile": tile,
            "n": int(mask.sum()),
            "skipped": False,
            "coverage_raw": coverage(residual[mask], variance[mask]),
            "coverage_global_nugget": coverage(residual[mask], variance[mask] + nugget),
            "nugget": fit_predictive_nugget(residual[mask], variance[mask]),
            **compute_interp_perf(frame, "measured", "GP_mean"),
        })
    return {"count": count, "global_nugget": nugget, "tiles": tiles}


def load_predictions(setup, model_path, args):
    """Predict the test points with a trained checkpoint. Returns (frame, nugget)."""
    import sys

    block = Path(__file__).resolve().parents[2] / "upscaling_gaussian_process_inference"
    sys.path.insert(0, str(block))
    import torch
    from upscaling_gaussian_process_inference.model_io import load_trained_model

    loaded = load_trained_model(model_path)
    predicted = loaded.model.inference(
        torch.from_numpy(setup["test_x"]).to(loaded.model.train_inputs.device),
        output_scaler=loaded.output_scaler,
        batch_size=args.validation_batch_size)
    return predicted, loaded.predictive_nugget


def _spread(tiles, key, path=None):
    """min, max and max/min of one measurement over the tiles that survived."""
    values = []
    for tile in tiles:
        if tile.get("skipped"):
            continue
        value = tile[path][key] if path else tile[key]
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return None
    low, high = float(min(values)), float(max(values))
    return {"min": low, "max": high, "ratio": high / low if low > 0 else float("inf"),
            "n_tiles": len(values)}


def summarise(payload):
    lines = [
        f"inducing points   {payload['setup']['n_inducing']:,} "
        f"(spacing {payload['setup']['spacing']:.6g} scaled)",
        f"min lengthscale   {payload['setup']['min_lengthscale']:.6g} scaled",
        f"lag band          [{payload['band'][0]:.6g}, {payload['band'][1]:.6g}]",
    ]
    if payload.get("bootstrap"):
        boot = payload["bootstrap"]
        lines.append(
            f"bootstrap spread  total variance {boot['total_variance']:.4g} "
            f"(log sd {boot['total_variance_log_sd']:.3f}, "
            f"{boot['n_replicates']} replicates)")
    if payload.get("shuffled"):
        lines.append("NULL TEST: values shuffled across locations; "
                     "every spread below should collapse to 1")

    for sweep in payload["field"]:
        lines.append("")
        lines.append(f"{sweep['count']}x{sweep['count']} tiles")
        header = (f"  {'tile':>5}{'n':>9}{'spacing':>10}{'band var':>12}"
                  f"{'short ls':>10}{'unrep':>10}")
        lines.append(header)
        for tile in sweep["tiles"]:
            if tile.get("skipped"):
                lines.append(f"  {tile['tile']:>5}{tile['n']:>9}   (skipped)")
                continue
            ladder = tile["ladder"]
            lines.append(
                f"  {tile['tile']:>5}{tile['n']:>9}{tile['point_spacing']:>10.4g}"
                f"{tile['band_variance']:>12.4g}"
                + (f"{ladder['short_lengthscale']:>10.4g}"
                   f"{ladder['unrepresentable']:>10.4g}"
                   if ladder else f"{'-':>10}{'-':>10}"))
        for label, key, path in (("band variance", "band_variance", None),
                                 ("unrepresentable", "unrepresentable", "ladder"),
                                 ("short lengthscale", "short_lengthscale", "ladder"),
                                 ("point spacing", "point_spacing", None)):
            spread = _spread(sweep["tiles"], key, path)
            if spread:
                lines.append(f"  {label:<20} min {spread['min']:.4g}  "
                             f"max {spread['max']:.4g}  ratio {spread['ratio']:.3g}")
        coarse = [tile["tile"] for tile in sweep["tiles"]
                  if not tile.get("skipped") and tile["point_spacing"] >= payload["band"][0]]
        if coarse:
            lines.append(f"  WARNING tiles {coarse} are sampled no finer than the "
                         "band's short end, so their band variance also carries "
                         "whatever sits between their points")

    for sweep in payload.get("model", []):
        lines.append("")
        lines.append(f"{sweep['count']}x{sweep['count']} tiles, held out "
                     f"(global nugget {sweep['global_nugget']:.4g})")
        lines.append(f"  {'tile':>5}{'n':>9}{'cover raw':>11}{'cover +nug':>12}"
                     f"{'own nugget':>12}{'rmse':>10}")
        for tile in sweep["tiles"]:
            if tile.get("skipped"):
                lines.append(f"  {tile['tile']:>5}{tile['n']:>9}   (skipped)")
                continue
            lines.append(
                f"  {tile['tile']:>5}{tile['n']:>9}{tile['coverage_raw']:>11.3f}"
                f"{tile['coverage_global_nugget']:>12.3f}{tile['nugget']:>12.4g}"
                f"{tile['rmse']:>10.4g}")
        spread = _spread(sweep["tiles"], "nugget")
        if spread:
            lines.append(f"  {'own nugget':<20} min {spread['min']:.4g}  "
                         f"max {spread['max']:.4g}  ratio {spread['ratio']:.3g}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--ninducing", type=int, default=12_000)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--tiles", type=int, nargs="+", default=[3, 4, 5],
        help="Tile counts per axis. A conclusion that does not survive the "
             "sweep is not a conclusion.")
    parser.add_argument(
        "--min-tile-points", type=int, default=500,
        help="Tiles below this are skipped rather than measured on noise.")
    parser.add_argument(
        "--band-low", type=float, default=None,
        help="Short end of the lag band. Defaults to min_lengthscale.")
    parser.add_argument(
        "--band-high", type=float, default=None,
        help="Long end of the lag band. Defaults to one inducing spacing, so "
             "the band holds exactly what the model cannot resolve.")
    parser.add_argument(
        "--model", default=None,
        help="Trained .pth checkpoint. Adds the held-out coverage measurement.")
    parser.add_argument("--validation-batch-size", type=int, default=8192)
    parser.add_argument("--n-pairs", type=int, default=1_000_000)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=40)
    parser.add_argument("--bootstrap-blocks", type=int, default=8)
    parser.add_argument(
        "--shuffle", action="store_true",
        help="Null test: shuffle the values across locations, keeping the "
             "coordinates. Every spread must collapse to 1, or the maps are "
             "measuring point density rather than the field.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="tile_sweep.json")
    args = parser.parse_args()

    setup = prepare(args.train, args.test, args.ninducing, args.k)
    values = setup["scaled_output"].astype(np.float64).ravel()
    if args.shuffle:
        np.random.default_rng(args.seed).shuffle(values)

    band = (args.band_low or setup["min_lengthscale"],
            args.band_high or setup["spacing"])

    # The reference spread, not a measurement. A shuffled field has no ladder to
    # fit, so this fails on exactly the null test the script has to survive.
    try:
        uncertainty = bootstrap_ladder(
            setup["scaled_input"].astype(np.float64), values,
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
    except ValueError as error:
        print(f"bootstrap skipped: {error}")
        uncertainty = None

    payload = {
        "arguments": vars(args),
        "shuffled": args.shuffle,
        "band": list(band),
        "setup": {key: setup[key] for key in
                  ("n_inducing", "k", "spacing", "min_lengthscale", "bbox_diagonal")},
        "bootstrap": None if uncertainty is None else {
            "lengthscale_log_sd": uncertainty.lengthscale_log_sd.tolist(),
            "total_variance": uncertainty.total_variance,
            "total_variance_log_sd": uncertainty.total_variance_log_sd,
            "n_replicates": uncertainty.n_replicates,
        },
        "field": [sweep_field(setup, values, count, args) for count in args.tiles],
    }

    if args.model:
        predicted, nugget = load_predictions(setup, args.model, args)
        payload["model"] = [sweep_model(setup, predicted, nugget, count, args)
                            for count in args.tiles]

    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(summarise(payload))
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
