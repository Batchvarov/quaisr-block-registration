"""Read the training/test parquets and cap the point sets used for metrics."""

from os import PathLike

import numpy as np
import pyarrow.parquet as pq

from upscaling_tools.memory import nngp_posterior_batch_size

# Default cap on the number of points used for validation/diagnostics. Metrics
# on much larger sets add no accuracy but blow up memory and time. Override per
# run through the validation_options input; a value <= 0 disables the cap.
MAX_VALIDATION_POINTS = 100_000


def read_samples(path: PathLike):
    """Load a train/test parquet with schema {X: list<float>, y: float}.

    Extract flat (n, k) X and (n,) y arrays via pyarrow without Python-boxing the
    list column. Skipping the DataFrame materialisation avoids allocating
    millions of small list objects.
    """
    tbl = pq.read_table(path)
    x_col = tbl["X"].combine_chunks()
    n = len(x_col)
    if n == 0:
        X = np.zeros((0, 0), dtype=np.float64)
    else:
        k = len(x_col[0])
        X = np.asarray(x_col.values.to_numpy(zero_copy_only=False)).reshape(n, k)
    y = tbl["y"].to_numpy(zero_copy_only=False)
    return X, y


def _options_dict(validation_options) -> dict:
    return {} if validation_options is None else dict(validation_options.__dict__)


def resolve_point_cap(validation_options, field: str) -> int | None:
    """Read one point cap off the validation_options record.

    An absent record, an absent field or an explicit None all fall back to
    MAX_VALIDATION_POINTS. A value <= 0 returns None, meaning "use every point".
    """
    value = _options_dict(validation_options).get(field)
    if value is None:
        value = MAX_VALIDATION_POINTS
    value = int(value)
    return value if value > 0 else None


def resolve_validation_batch_size(validation_options, k: int,
                                  dtype_bytes: int = 4) -> int:
    """Points per posterior batch during validation/diagnostics.

    Absent or non-positive, derive it from k and free GPU memory.
    """
    value = _options_dict(validation_options).get("validation_batch_size")
    if value is not None and int(value) > 0:
        return int(value)
    return nngp_posterior_batch_size(k, dtype_bytes)


def subsample(X, y, max_points=MAX_VALIDATION_POINTS, seed=0):
    """Randomly subsample (X, y) to at most max_points rows (no replacement).

    max_points of None keeps every point.
    """
    n = len(y)
    if max_points is None or n <= max_points:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_points, replace=False)
    return X[idx], y[idx]


# Control cells across the survey for the predictive nugget field. Zero keeps
# the single survey-wide nugget, which is what every earlier run reported and
# what every existing checkpoint carries.
DEFAULT_NUGGET_CONTROL_COUNT = 0


def resolve_nugget_control_count(validation_options) -> int:
    """Cells across for the nugget field. Zero keeps the single nugget."""
    value = _options_dict(validation_options).get("nugget_control_count")
    if value is None:
        value = DEFAULT_NUGGET_CONTROL_COUNT
    return max(int(value), 0)
