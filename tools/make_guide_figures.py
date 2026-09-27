#!/usr/bin/env python3
"""The figures of docs/USER_GUIDE.md, light and dark (needs matplotlib, a
documentation-only dependency):

    tools/make_guide_figures.py        # writes docs/img/*.png, prints the numbers quoted
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bayesbin import BernoulliModel, fit, spike_counts

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
THEMES = {  # the reference palette's roles (surface, text, muted, grid, axis, series)
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", blue="#2a78d6", orange="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", blue="#3987e5", orange="#d95926"),
}


def data():
    """Tutorial 1's data: 30 trials, 1-ms intervals, a sharp onset at 80 ms, a 50-ms
    transient, a long plateau, and background before and after."""
    rng = np.random.default_rng(1)
    t = np.arange(-100, 500)
    p_true = np.select([t < 80, t < 130, t < 380], [0.01, 0.08, 0.05], 0.01)
    trials = [t[rng.random(t.size) < p_true] for _ in range(30)]
    s, g = spike_counts(trials, t_start=-100, t_end=499)
    return t, p_true, s, g


def histogram(s, n_trials, width):
    """Firing probability per ms in fixed bins of `width` ms, spread back over the ms."""
    return np.repeat(s.reshape(-1, width).sum(axis=1) / (n_trials * width), width)


def style(ax, th, title):
    ax.set_facecolor(th["surface"])
    ax.set_title(title, loc="left", fontsize=10, color=th["text"], pad=6)
    ax.grid(axis="y", color=th["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(th["axis"])
    ax.tick_params(colors=th["muted"], labelsize=8, length=0, pad=4)


def figure_bins(th, name):
    t, p_true, s, g = data()
    r = fit(BernoulliModel(s, g), max_boundaries=20)
    panels = [("1-ms bins: too small, mostly noise", histogram(s, 30, 1)),
              ("50-ms bins: too large, onset and transient smeared", histogram(s, 30, 50)),
              ("10-ms bins: a compromise, still noisy and still smearing", histogram(s, 30, 10)),
              ("bayesbin: bin sizes and positions from the data, ± 1 sd", r.rate)]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.8), sharex=True,
                             facecolor=th["surface"])
    for i, (ax, (title, est)) in enumerate(zip(axes, panels)):
        style(ax, th, title)
        ax.plot(t, p_true, color=th["text2"], linewidth=1.2, linestyle=(0, (4, 3)), label="true rate")
        if i < 3:
            ax.step(t, est, where="mid", color=th["orange"], linewidth=1.3, label="histogram")
        else:
            ax.fill_between(t, r.rate - r.rate_std, r.rate + r.rate_std, color=th["blue"],
                            alpha=0.22, linewidth=0)
            ax.plot(t, est, color=th["blue"], linewidth=1.6, label="bayesbin")
        ax.set_ylim(0, 0.22 if i == 0 else 0.11)  # the 1-ms panel on its own scale: its noise is taller
        rms = np.sqrt(np.mean((est - p_true) ** 2))
        ax.text(0.995, 0.93, f"RMS error {rms:.4f}", transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=th["text2"])
        print(f"  {title}: RMS error {rms:.4f}")
        leg = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=8,
                        labelcolor=th["text2"], handlelength=2.2)
    axes[-1].set_xlabel("time from stimulus (ms)", color=th["text2"], fontsize=9)
    fig.supylabel("firing probability per ms", color=th["text2"], fontsize=9)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(OUT / f"bins-{name}.png", dpi=150, facecolor=th["surface"])
    plt.close(fig)


def figure_boundaries(th, name):
    t, p_true, s, g = data()
    r = fit(BernoulliModel(s, g), max_boundaries=20)
    fig, ax = plt.subplots(figsize=(7.2, 2.4), facecolor=th["surface"])
    style(ax, th, "Where does the rate change? P(a bin ends at t | data)")
    for tc in (79, 129, 379):
        ax.axvline(tc, color=th["text2"], linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(79, 0.97, " true changes", color=th["text2"], fontsize=8, va="top",
            transform=ax.get_xaxis_transform())
    ax.plot(t[:-1], r.boundary_posterior, color=th["blue"], linewidth=1.6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("time from stimulus (ms)", color=th["text2"], fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"boundaries-{name}.png", dpi=150, facecolor=th["surface"])
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, th in THEMES.items():
        print(name)
        figure_bins(th, name)
        figure_boundaries(th, name)
