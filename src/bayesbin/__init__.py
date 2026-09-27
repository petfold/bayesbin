"""Exact Bayesian binning of rates (Endres, Oram, Schindelin & Földiák 2008)."""

from bayesbin.core import (
    BernoulliModel,
    BinningResult,
    PoissonModel,
    bin_posterior,
    bin_posterior_for_m,
    credible_m_range,
    fit,
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
    "bin_posterior",
    "bin_posterior_for_m",
    "credible_m_range",
    "fit",
    "spike_counts",
]
__version__ = "0.3.0"
