"""Exact Bayesian binning of rates (Endres, Oram, Schindelin & Földiák 2008)."""

from bayesbin.core import (
    BernoulliModel,
    BinningResult,
    PoissonModel,
    best_prior,
    bin_posterior,
    boundary_positions,
    bin_posterior_for_m,
    credible_m_range,
    fit,
    fit_cyclic,
    latency_posterior,
    separation_level,
    spike_counts,
)
from bayesbin.online import OnlineBinning
from bayesbin.stream import ChangePointMixture, ChangePointStream

__all__ = [
    "BernoulliModel",
    "BinningResult",
    "ChangePointMixture",
    "ChangePointStream",
    "OnlineBinning",
    "PoissonModel",
    "best_prior",
    "bin_posterior",
    "boundary_positions",
    "bin_posterior_for_m",
    "credible_m_range",
    "fit",
    "fit_cyclic",
    "latency_posterior",
    "separation_level",
    "spike_counts",
]
__version__ = "0.3.0"
