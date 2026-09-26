"""Exact Bayesian binning of rates (Endres, Oram, Schindelin & Földiák 2008)."""

from bayesbin.core import (
    BernoulliModel,
    BinningResult,
    PoissonModel,
    bin_posterior_for_m,
    credible_m_range,
    fit,
    spike_counts,
)

__all__ = [
    "BernoulliModel",
    "BinningResult",
    "PoissonModel",
    "bin_posterior_for_m",
    "credible_m_range",
    "fit",
    "spike_counts",
]
__version__ = "0.1.0"
