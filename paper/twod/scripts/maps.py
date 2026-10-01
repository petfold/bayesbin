"""World maps of the binnings: figures/globe_{bins,zoom,rates}.pdf for the paper (vector text,
rasterised cells) and build/globe_*.png for the web page. Reads build/globe_<name>.npz and
data/globe_<name>.json; coastlines from Natural Earth (public domain, 1:110m)."""

import json

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from common import BUILD, DATA, DATASETS, FIGURES, read_json
from h3tree import ALL

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
ORANGE = ["#ffdfcd", "#ffc8af", "#ffb291", "#ff9d75", "#fd885c", "#f07544", "#e16430",
          "#d05520", "#bb4915", "#a54013", "#8e3916", "#75341a", "#5c2f1e"]
SIZE = LinearSegmentedColormap.from_list("size", BLUE[::-1])  # large bins light, fine bins dark
RATE = LinearSegmentedColormap.from_list("rate", ORANGE)
NORM_SIZE = LogNorm(1e4, 3e7)
WORLD, EUROPE, AMERICA = (-180, 180, -60, 85), (-12, 48, 28, 62), (-170, -100, 18, 66)

INDEX = {c: i for i, c in enumerate(ALL)}
POLYS = []
for c in ALL:
    b = np.array(h3.cell_to_boundary(c))[:, ::-1]  # (lng, lat)
    if b[:, 0].max() - b[:, 0].min() > 180:
        b[b[:, 0] < 0, 0] += 360  # unwrap cells across the antimeridian
    POLYS.append(b)
WRAP = np.array([p[:, 0].max() > 180 for p in POLYS])
COAST = [np.array(f["geometry"]["coordinates"])
         for f in json.loads((DATA / "ne_110m_coastline.geojson").read_text())["features"]]


def edges(assign):
    """Segments between neighbouring cells in different bins."""
    seg = []
    for c in ALL:
        a = assign[INDEX[c]]
        for g in h3.grid_disk(c, 1):
            if g > c and assign[INDEX[g]] != a:
                b = np.array(h3.directed_edge_to_boundary(h3.cells_to_directed_edge(c, g)))[:, ::-1]
                if b[:, 0].max() - b[:, 0].min() > 180:
                    b[b[:, 0] < 0, 0] += 360
                seg.append(b[:2])
    return np.array(seg)


def draw(ax, values, cmap, norm, seg=None, box=WORLD, title=""):
    zoom = box[1] - box[0] < 180
    ax.set_facecolor(SURFACE)
    pp = POLYS + [p - [360, 0] for p, w in zip(POLYS, WRAP) if w]
    pc = PolyCollection(pp, array=np.concatenate([values, values[WRAP]]), cmap=cmap, norm=norm,
                        edgecolors="face", linewidths=0.1, rasterized=True)
    ax.add_collection(pc)
    if seg is not None and len(seg):
        extra = seg[(seg[:, :, 0] > 180).any(1)] - [360, 0]
        ax.add_collection(LineCollection(np.concatenate([seg, extra]), colors=SURFACE,
                                         linewidths=0.45 if zoom else 0.15, alpha=0.85, rasterized=True))
    cw = 0.6 if zoom else 0.3  # coastlines: a light halo and a dark core
    ax.add_collection(LineCollection(COAST, colors=SURFACE, linewidths=cw * 3, alpha=0.9))
    ax.add_collection(LineCollection(COAST, colors=INK, linewidths=cw, alpha=0.85))
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    ax.set_aspect(1 / np.cos(np.radians((box[2] + box[3]) / 2)) if zoom else "equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#c3c2b7")
        s.set_linewidth(0.5)
    ax.set_title(title, loc="left", fontsize=7.5, color=INK, pad=3)
    return pc


def size_bar(fig, pc, rect):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(pc, cax=cax, orientation="horizontal", extend="max")
    cb.outline.set_linewidth(0.4)
    cb.set_ticks([1.24e4, 8.7e4, 6.1e5, 4.3e6, 3e7])
    cb.set_ticklabels(["12,000 km²\nres 3", "87,000\nres 2", "610,000\nres 1", "4.3 M\nres 0", "30 M or more"])
    cb.ax.tick_params(labelsize=6.5, width=0.4, length=2, colors=INK2)
    cb.ax.minorticks_off()


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", dpi=300, facecolor=SURFACE)
    fig.savefig(BUILD / f"{name}.png", dpi=220, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf and build/{name}.png")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})
    d = {n: np.load(BUILD / f"globe_{n}.npz") for n in DATASETS}
    meta = {n: read_json(f"globe_{n}.json") for n in DATASETS}
    seg = {(n, k): edges(d[n][f"{k}_bin"]) for n in DATASETS for k in ("tree", "merged")}
    label = {"gdelt": "News events", "usgs": "Earthquakes"}

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 3.95), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.28, "wspace": 0.03, "bottom": 0.17, "top": 0.94,
                                          "left": 0.005, "right": 0.995})
    for row, n in enumerate(DATASETS):
        nb = {"tree": meta[n]["best_partition"]["bins"], "merged": meta[n]["merged"]["regions"]}
        for col, k in enumerate(("tree", "merged")):
            what = f"tree, best partition: {nb[k]:,} bins" if k == "tree" else f"then greedy merging: {nb[k]:,} bins"
            pc = draw(axes[row, col], d[n][f"{k}_area"], SIZE, NORM_SIZE, seg[(n, k)], title=f"{label[n]}: {what}")
    size_bar(fig, pc, [0.25, 0.085, 0.5, 0.025])
    save(fig, "globe_bins")

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 6.1), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.14, "wspace": 0.04, "bottom": 0.1, "top": 0.96,
                                          "left": 0.005, "right": 0.995})
    boxes = {"gdelt": (EUROPE, "News events, Europe"), "usgs": (AMERICA, "Earthquakes, Alaska and the western US")}
    for row, n in enumerate(DATASETS):
        box, where = boxes[n]
        for col, k in enumerate(("tree", "merged")):
            pc = draw(axes[row, col], d[n][f"{k}_area"], SIZE, NORM_SIZE, seg[(n, k)], box,
                      f"{where}: {'tree' if k == 'tree' else 'merged'}")
    size_bar(fig, pc, [0.25, 0.045, 0.5, 0.016])
    save(fig, "globe_zoom")

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), facecolor=SURFACE,
                             gridspec_kw={"wspace": 0.03, "bottom": 0.3, "top": 0.91, "left": 0.005, "right": 0.995})
    for col, (n, lo, hi) in enumerate([("gdelt", 1e-3, 1e3), ("usgs", 1e-5, 10)]):
        pc = draw(axes[col], np.clip(d[n]["pm_rate"], lo, hi), RATE, LogNorm(lo, hi),
                  title=f"{label[n]}: posterior mean rate")
        cax = fig.add_axes([0.06 + 0.5 * col, 0.17, 0.38, 0.035])
        cb = fig.colorbar(pc, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(labelsize=6.5, width=0.4, length=2, colors=INK2)
        cb.set_label("events per 10,000 km² per day", fontsize=6.5, color=INK2, labelpad=1)
    save(fig, "globe_rates")


if __name__ == "__main__":
    main()
