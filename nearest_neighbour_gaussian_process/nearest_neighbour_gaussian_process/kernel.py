"""The auto kernel: an additive Matern ladder bounded by the inducing spacing.

docs/README.md carries the full derivation of every constant below. Each comment
here states only the live reason.
"""

import numpy as np
import scipy.spatial

from upscaling_tools.inducing import InducingPointsAllocation
from upscaling_tools.kernels import construct_kernel, control_grid


class PrecomputedInducing(InducingPointsAllocation):
    """Hand construct_model an inducing set that is already allocated."""

    def __init__(self, points, returns_data_points: bool) -> None:
        self.points = points
        self._returns_data_points = returns_data_points

    @property
    def returns_data_points(self) -> bool:
        return self._returns_data_points

    def initialise(self, X):
        return self.points


# Not a slack safety rail: the ELBO is monotone in the outputscale of whichever
# component carries the signal, so the fit pins it on whatever value this takes.
# Changing it changes the model.
MAX_OUTPUTSCALE = 2.0

# Tighter, for an unrelated reason: the model cannot resolve structure below the
# inducing spacing, so an outputscale near 1 here lets the roughest component
# carry the whole signal and the fit turns to speckle.
MAX_ROUGH_OUTPUTSCALE = 0.1

# Floor under every outputscale interval, keeping it off LessThan's unbounded
# side where the fit can go negative and make the kernel matrix indefinite.
# Absolute, not a fraction of the cap: components that are not carrying the
# signal pin on this floor, so a relative one moved them with the cap.
MIN_OUTPUTSCALE = 1e-3

# Lower bound on the short-scale lengthscale, in inducing spacings, and -- via
# lengthscale_at -- where that component starts. The start is the load-bearing
# half: a lengthscale travels only a bounded distance in a training run, so this
# places the component rather than merely constraining it.
MIN_LENGTHSCALE_SPACINGS = 0.25

# Cells across the survey for the short-scale amplitude field. Zero keeps the
# stationary kernel, which is what every earlier run trained and what every
# existing checkpoint carries.
DEFAULT_MODULATION_CONTROL_COUNT = 0

# Symmetric bound on the log weight of each control point. The bumps overlap, so
# the realised log amplitude is a sum of several weights and the contrast across
# the survey is wider than this bound alone: at 1.0 a fitted field reaches about
# a factor of 50 between its strongest and weakest region. Widen it only with
# evidence: the ELBO rewards prior variance wherever it can be put, which is the
# same failure the outputscale caps exist to stop.
DEFAULT_MAX_LOG_AMPLITUDE = 1.0


def inducing_spacing(inducing_points) -> float:
    """Median distance from an inducing point to its nearest neighbour."""
    Z = inducing_points
    if hasattr(Z, "detach"):
        Z = Z.detach().cpu().numpy()
    Z = np.ascontiguousarray(Z)
    if len(Z) < 2:
        return 0.0
    tree = scipy.spatial.cKDTree(Z)
    nn_dists, _ = tree.query(Z, k=2, workers=-1)
    return float(np.median(nn_dists[:, 1]))


def lengthscale_at(scale):
    """Bound a lengthscale below by `scale`, and start it there too.

    gpytorch starts every bounded lengthscale at `bound + softplus(0)`, so
    without an initial value terms with small bounds all start near 0.69 and
    model the same wavelength. The 1.5 is headroom: an initial value sitting on
    the bound inverts to raw = -inf.
    """
    return ['greaterthan', scale, scale * 1.5]


def outputscale_upto(cap):
    """Bound an outputscale into (MIN_OUTPUTSCALE, cap), starting at the midpoint."""
    return ['interval', [MIN_OUTPUTSCALE, cap]]


def modulation_options(x_points, control_count: int,
                       max_log_amplitude: float = DEFAULT_MAX_LOG_AMPLITUDE):
    """Amplitude field for the short-scale component, or None when it is off.

    The field is a control_count x control_count grid of RBF bumps over the
    training extent, so the short component can be strong in one region and
    weak in another. The lengthscale is one cell, which keeps neighbouring bumps
    overlapping.

    The control points go into the kernel definition rather than being derived
    at load time, because `construct_kernel` has to rebuild the same module from
    a saved checkpoint without seeing the data again.
    """
    if control_count <= 0:
        return None

    extent = x_points.max(axis=0) - x_points.min(axis=0)
    spacing = float(np.min(extent)) / control_count
    return {
        'control_points': control_grid(x_points, control_count).numpy(),
        'lengthscale': spacing,
        'log_amplitude': ['interval', [-max_log_amplitude, max_log_amplitude]],
    }


def resolve_modulation(kernel_options, x_points):
    """Read the amplitude field off the kernel_options input.

    An absent record, an absent field or an explicit None all leave the short
    component stationary.
    """
    options = {} if kernel_options is None else dict(kernel_options.__dict__)

    count = options.get('modulation_control_count')
    if count is None:
        count = DEFAULT_MODULATION_CONTROL_COUNT

    limit = options.get('modulation_log_amplitude')
    if limit is None or float(limit) <= 0:
        limit = DEFAULT_MAX_LOG_AMPLITUDE

    return modulation_options(x_points, int(count), float(limit))


def construct_multiscale_kernel(x_points, min_lengthscale, modulation=None):
    """Additive Matern kernel over three length scales.

    min_lengthscale bounds the short-scale component, and starts it at 1.5x
    that. Below about a quarter of the inducing spacing the component collapses
    towards zero lengthscale and acts as a nugget, which is the failure the old
    2.0 setting avoided by staying well clear of it. See docs/README.md.

    modulation, when given, makes the short-scale component non-stationary: its
    amplitude then varies over the survey instead of holding one value
    everywhere. Only that component takes it. The long and mid components carry
    the regional trend, and a trend that fades in and out of regions is a
    different model from the one this kernel states.

    Matern throughout, not RBF. The NNGP conditions each point on its k nearest
    neighbours, which relies on those neighbours screening off the rest of the
    field. Screening fails for analytic sample paths, so an RBF term is
    approximated far worse than a Matern one at the same lengthscale.
    """
    bbox_diagonal = float(np.linalg.norm(x_points.max(axis=0) - x_points.min(axis=0)))
    long_lengthscale = bbox_diagonal / 4
    mid_lengthscale = float(np.sqrt(long_lengthscale * min_lengthscale))

    scales = (long_lengthscale, mid_lengthscale, min_lengthscale)
    smoothness = (2.5, 1.5, 1.5)
    caps = (MAX_OUTPUTSCALE, MAX_OUTPUTSCALE, MAX_ROUGH_OUTPUTSCALE)

    modulations = (None, None, modulation)

    kernel_opts = []
    for nu, scale, cap, component in zip(smoothness, scales, caps, modulations):
        options = {
            'nu': nu,
            'ard_num_dims': 1,
            'lengthscale': lengthscale_at(scale),
            'outputscale': outputscale_upto(cap),
            'keops': True,
        }
        if component is not None:
            options['modulation'] = component
        kernel_opts.append({'kernel': 'matern', 'options': options})

    return construct_kernel(kernel_opts), kernel_opts


def describe_kernel(kernel_opts) -> str:
    """One line naming each component's smoothness, bounds and amplitude field."""
    return "; ".join(
        f"Matern({opts['options']['nu']}) "
        f"lengthscale>={opts['options']['lengthscale'][1]:g} "
        f"outputscale<={opts['options']['outputscale'][1][1]:g}"
        f"{_describe_modulation(opts['options'].get('modulation'))}"
        for opts in kernel_opts)


def _describe_modulation(modulation) -> str:
    if not modulation:
        return ""
    limit = modulation['log_amplitude'][1][1]
    return (f" modulated over {len(modulation['control_points'])} control points "
            f"(lengthscale {modulation['lengthscale']:g}, |log g|<={limit:g})")
