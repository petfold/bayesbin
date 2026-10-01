"""Every number the paper quotes, as LaTeX macros (data/numbers.tex), the generated tables
(data/*_table.tex) and the plotted series (data/plot_*.csv), from the pipeline's results.

The text's qualitative claims are checked here too. A claim the new data no longer support is
printed and listed in a box at the top of the PDF (macro \\claimwarnings), so that a rerun on more
data shows at once which sentences need rewriting."""

import csv
import math

from common import DATA, PLACES, read_json

PREFIX = {"gdelt": "news", "usgs": "quake"}
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six"]
macros, warnings = {}, []


def m(name, value):
    assert name.isalpha(), name
    macros[name] = value


def num(x, digits=0):
    return f"{x:,.{digits}f}"


def sig(x, n=2):
    """x to n significant figures, as text."""
    if x == 0:
        return "0"
    d = max(n - 1 - int(math.floor(math.log10(abs(x)))), 0)
    return f"{x:,.{d}f}"


def pct(x):
    return f"{100 * x:.0f}\\%"


def sec(s):
    return f"{s * 1e3:.1f}~ms" if s < 0.1 else f"{s:.2g}~s" if s < 10 else f"{s:.0f}~s"


def sci(x):
    """x as LaTeX math, e.g. 4\\times10^{-14}."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.0e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def signed(x, digits):
    """A signed number as math, so that its minus is a minus sign."""
    return f"${x:+.{digits}f}$"


def claim(ok, text):
    if not ok:
        warnings.append(text)
        print(f"CLAIM NO LONGER HOLDS: {text}")


# --- the globe -----------------------------------------------------------------------------------
for name, p in PREFIX.items():
    g = read_json(f"globe_{name}.json")
    m(f"{p}Events", num(g["events"]))
    m(f"{p}Days", f"{g['days']:.1f}")
    m(f"{p}First", g["first"])
    m(f"{p}Last", g["last"])
    m(f"{p}Cells", num(g["cells_with_events"]))
    m("allCells", num(g["cells"]))
    m("finestRes", str(g["res"]))
    m(f"{p}Nodes", num(g["nodes"]))
    m(f"{p}EmptyChildren", num(g["empty_children"]))
    pr = g["prior"]
    m(f"{p}Alpha", f"{pr['alpha']:g}")
    m(f"{p}Rho", f"{pr['rho']:g}")
    m(f"{p}PriorMean", f"{pr['mean_vs_global']:g}")
    bp = g["best_partition"]
    m(f"{p}TreeBins", num(bp["bins"]))
    m(f"{p}EmptyBins", num(bp["empty_bins"]))
    for r, v in bp["by_res"].items():
        word = "World" if r == "-1" else "Res" + WORDS[int(r)]
        m(f"{p}Bins{word}", num(v["bins"]))
        m(f"{p}Share{word}", pct(v["area_share"]))
    mg = g["merged"]
    m(f"{p}Regions", num(mg["regions"]))
    m(f"{p}EmptyShare", pct(mg["empty_share"]))
    m(f"{p}LargestRegion", sig(mg["largest_km2"] / 1e6, 3))
    s = g["seconds"]
    m(f"{p}Pass", sec(s["pass"]))
    m(f"{p}Search", sec(s["prior_search"]))
    m("priorSettings", num(s["prior_settings"]))
    m(f"{p}Merge", sec(s["merge"]))
    claim(g["places"]["N Pacific (30°N 150°W)"]["res"] == 0, f"{name}: the North Pacific is one resolution-0 bin")
    claim(g["coverage_error"] < 1e-9, f"{name}: posterior bin probabilities cover every cell once")
    edge = (pr["alpha"] in (0.003, 3.0), pr["rho"] in (0.001, 0.7),
            any(math.isclose(pr["mean_vs_global"], v, rel_tol=1e-6) for v in (1000.0, 0.01)))
    claim(not any(edge), f"{name}: the evidence-chosen prior lies inside its grid")
m("newsKansas", num(read_json("globe_gdelt.json")["places"]["Kansas (centre of the US)"]["events"]))
claim(read_json("globe_gdelt.json")["places"]["London"]["res"] == read_json("globe_gdelt.json")["res"],
      "news: London gets the finest cells")
claim(read_json("globe_usgs.json")["places"]["Los Angeles"]["res"] == read_json("globe_usgs.json")["res"],
      "earthquakes: Los Angeles gets the finest cells")

# places table
gn, gq = read_json("globe_gdelt.json")["places"], read_json("globe_usgs.json")["places"]
with open(DATA / "places_table.tex", "w") as f:
    for place in PLACES:
        a, b = gn[place], gq[place]
        f.write(f"{place} & {a['res']} & {num(a['area_km2'])} & {num(a['events'])} & "
                f"{b['res']} & {num(b['area_km2'])} & {num(b['events'])} \\\\\n")

# --- held-out prediction ---------------------------------------------------------------------------
rows = list(csv.DictReader(open(DATA / "heldout.csv")))
by = {(r["dataset"], r["key"]): r for r in rows}
labels = {"world": "one rate for the whole Earth", "grid0": "fixed grid, resolution 0",
          "grid1": "fixed grid, resolution 1", "grid2": "fixed grid, resolution 2",
          "grid3": "fixed grid, resolution 3", "tree": "tree, averaged over partitions",
          "best": "tree, best partition only", "merged0.1": "greedy merging, bonus 2.3 nats",
          "merged1": "greedy merging, no bonus", "merged10": "greedy merging, cost 2.3 nats"}
with open(DATA / "heldout_table.tex", "w") as f:
    for key, label in labels.items():
        cells = []
        for name in PREFIX:
            r = by[(name, key)]
            cells += [r["bins"] and num(float(r["bins"])) or "--", signed(float(r["per_event"]), 3)]
        f.write(f"{label} & " + " & ".join(cells) + " \\\\\n")
for name, p in PREFIX.items():
    h = read_json(f"heldout_{name}.json")
    m(f"{p}TrainDays", f"{h['days_train']:.1f}")
    m(f"{p}TestDays", f"{h['days_test']:.1f}")
    m(f"{p}TrainEvents", num(h["events_train"]))
    m(f"{p}TestEvents", num(h["events_test"]))
    m(f"{p}TestChange", signed(100 * (h["events_test"] / h["events_train"] - 1), 0) + "\\%")
    per = {k: float(by[(name, k)]["per_event"]) for k in labels}
    total = {k: float(by[(name, k)]["vs_best"]) for k in labels}
    m(f"{p}GainGrid", num(-total["grid3"] + total["tree"]))
    m(f"{p}GainGridPerEvent", f"{per['tree'] - per['grid3']:.3f}")
    m(f"{p}BestPerEvent", signed(per["best"], 3))
    m(f"{p}MergedPerEvent", signed(per["merged0.1"], 2))
    m(f"{p}MergedNoBonusPerEvent", signed(per["merged1"], 2))
    m(f"{p}GridTwoPerEvent", signed(per["grid2"], 2))
    m(f"{p}WorldPerEvent", signed(per["world"], 1))
    claim(max(per, key=per.get) == "tree", f"{name}: the tree average predicts best")
    claim(per["tree"] > per["grid3"], f"{name}: the tree average beats the finest fixed grid")
    claim(per["merged0.1"] < per["grid3"] and per["merged1"] < per["grid3"],
          f"{name}: greedy merging predicts worse than the finest fixed grid")
    with open(DATA / f"plot_heldout_{name}.csv", "w") as f:  # semicolons: the labels hold commas
        f.write("label;per_event\n")
        for k in ["merged0.1", "merged1", "grid3", "best", "tree"]:
            f.write(f"{labels[k]};{per[k]}\n")

# --- 1-D ----------------------------------------------------------------------------------------
t = read_json("tree1d.json")
m("oneDT", num(t["T"]))
m("oneDEvents", num(t["events"]))
m("oneDMmap", str(t["bayesbin"]["m_map"]))
m("oneDBinaryGain", signed(t["binary"]["log_evidence"] - t["bayesbin"]["log_evidence"], 1))
m("oneDDyadicLoss", f"{t['bayesbin']['log_evidence'] - t['dyadic']['log_evidence']:.0f}")
m("oneDBayesbinSeconds", sec(t["bayesbin"]["seconds"]))
m("oneDBinarySeconds", sec(t["binary"]["seconds_per_rho"]))
m("oneDDyadicSeconds", sec(t["dyadic"]["seconds_per_rho"]))
m("oneDBinaryRho", f"{t['binary']['rho']:g}")
claim(abs(t["binary"]["log_evidence"] - t["bayesbin"]["log_evidence"]) < 5,
      "1-D: the binary-cut tree and bayesbin are within 5 nats (a tie)")
claim(t["dyadic"]["log_evidence"] < t["bayesbin"]["log_evidence"], "1-D: the dyadic tree's evidence is lower")
claim(not (t["binary"]["rho_at_edge"] or t["dyadic"]["rho_at_edge"]), "1-D: the trees' best rho lies inside its grid")

# --- rectangles -----------------------------------------------------------------------------------
rect = read_json("rectangles.json")
chk = rect["two_level_checks"]
m("checkPartitions", ", ".join(num(c["partitions"]) for c in chk))
m("checkEvidenceError", sci(max(abs(c["dp"] - c["enumeration"]) for c in chk)))
m("checkRateError", sci(max(c["rate_error"] for c in chk)))
m("checkRecursiveError", sci(max(abs(c["dp"] - c["reference"]) for c in rect["guillotine_checks"])))
claim(all(abs(c["dp"] - c["enumeration"]) < 1e-9 and c["rate_error"] < 1e-12 for c in chk),
      "two-level: agrees with enumeration")
claim(all(abs(c["dp"] - c["reference"]) < 1e-9 for c in rect["guillotine_checks"]),
      "recursive cuts: agree with the reference recursion")
m("callOverhead", f"{rect['call_overhead_ms']:.1f}")
c44 = rect["counts"][-1]
m("countGrid", c44["grid"].replace("x", "\\times"))
for k, mac in [("tilings", "countTilings"), ("guillotine", "countGuillotine"), ("two_level", "countTwoLevel"),
               ("product", "countProduct"), ("binary_trees", "countBinaryTrees"),
               ("multiway_trees", "countMultiwayTrees"), ("singles_multiplicity", "countSingles")]:
    m(mac, num(c44[k]))
m("binaryPerPartition", f"{c44['binary_trees'] / c44['guillotine']:.0f}")
m("multiwayPerPartition", f"{c44['multiway_trees'] / c44['guillotine']:.1f}")
with open(DATA / "counts_table.tex", "w") as f:
    for c in rect["counts"]:
        grid = c["grid"].replace("x", "\\times")
        f.write(f"${grid}$ & " + " & ".join(
            num(c[k]) for k in ["tilings", "guillotine", "two_level", "product", "binary_trees", "multiway_trees"])
            + " \\\\\n")
timings = list(csv.DictReader(open(DATA / "timings.csv")))
for method in ["two_level_evidence", "two_level_rates", "recursive_evidence"]:
    with open(DATA / f"plot_{method}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "seconds"])
        for r in timings:
            if r["method"] == method:
                w.writerow([r["side"], r["seconds"]])
for r in timings:
    word = {"32": "ThirtyTwo", "64": "SixtyFour", "100": "Hundred", "128": "OneTwentyEight", "200": "TwoHundred"}.get(r["side"])
    if word:
        key = {"two_level_evidence": "twoLevelEvidence", "two_level_rates": "twoLevelRates",
               "recursive_evidence": "recursive"}[r["method"]]
        m(f"{key}{word}", sec(float(r["seconds"])))
        if r["method"] == "recursive_evidence":
            m(f"recursiveMB{word}", f"{float(r['table_mb']):.0f}")
        if r["method"] == "two_level_evidence":
            m(f"twoLevelM{word}", r["m_map"])

# --- streaming over the tree (Worldwatch's replay, scripts/stream.py) -----------------------------
st = read_json("stream.json")
ALONE, RES2, TREE = "cell alone (Layer 0 today)", "the resolution-2 cell", "tree, memory 3 days"
m("streamCells", num(st["cells"]))
m("streamEvents", num(st["events"]))
m("streamDays", f"{st['days']:.0f}")
m("streamWindows", num(st["windows"]))
m("streamWindowMinutes", f"{st['window_seconds'] / 60:.0f}")
m("streamRho", f"{st['rho']:g}")
gain = st["log_score_gain"]
m("streamGain", num(gain[TREE]))
m("streamGainResTwo", num(gain[RES2]))
m("streamGainMonth", num(gain["tree, memory 30 days"]))
m("streamGainAll", num(gain["tree, memory all"]))
ap = st["alarms_per_day"]
m("streamAlarmsAlone", f"{ap[ALONE]:.0f}")
m("streamAlarmsTree", f"{ap[TREE]:.0f}")
m("streamAlarmsResTwo", f"{ap[RES2]:.0f}")
big = st["big"]
m("streamBigQuakes", num(big["quakes"]))
for key, mac in [(ALONE, "Alone"), (RES2, "ResTwo"), (TREE, "Tree"), ("tree, memory all", "TreeAll")]:
    m(f"streamBig{mac}", num(big["in_window"][key]))
    m(f"streamBig{mac}Pct", pct(big["in_window"][key] / big["quakes"]))
cls = st["classes"]
for key in cls:
    cap = key.capitalize()
    c = cls[key]
    m(f"stream{cap}Cells", num(c["cells"]))
    m(f"stream{cap}Self", pct(c["weight_by_res"][3]))
    for name, mac in [(ALONE, "Alone"), (RES2, "ResTwo"), (TREE, "Tree")]:
        m(f"stream{cap}Tail{mac}", f"{1000 * c['variants'][name]['p999']:.2f}")
m("streamRareBase", pct(cls["rare"]["weight_by_res"][0]))
m("streamBusyResTwoLoss", f"{-cls['busy']['variants'][RES2]['log_score_per_cell_day']:.1f}")
with open(DATA / "stream_table.tex", "w") as f:
    for key in ["busy", "medium", "sparse", "rare"]:
        c = cls[key]
        v = c["variants"]
        f.write(f"{c['label']} & {num(c['cells'])} & " + " & ".join(
            f"{1000 * v[k]['p999']:.2f}" for k in (ALONE, RES2, TREE)) + " & " + " & ".join(
            signed(v[k]["log_score_per_cell_day"], 2) for k in (RES2, TREE))
            + f" & {pct(c['weight_by_res'][3])} \\\\\n")
for key in ("sparse", "rare"):
    v = cls[key]["variants"]
    claim(v[ALONE]["p999"] < 0.0007 and 0.00085 < v[TREE]["p999"] < 0.00115,
          f"streaming, {key} cells: alone the upper tail is too thin, pooled it is nominal")
claim(cls["busy"]["weight_by_res"][3] > 0.9, "streaming: busy cells keep most of the weight on themselves")
claim(cls["busy"]["variants"][RES2]["log_score_per_cell_day"] < -1, "streaming: a fixed resolution-2 bin costs busy cells")
claim(gain[TREE] > max(gain[RES2], gain["tree, memory 30 days"], gain["tree, memory all"]) > 0,
      "streaming: the tree with a 3-day forgetting time scores best")
claim(big["in_window"][TREE] > big["in_window"][ALONE], "streaming: more large quakes alarm in their window pooled")

# --- write ----------------------------------------------------------------------------------------
with open(DATA / "numbers.tex", "w") as f:
    f.write("% Generated by scripts/macros.py from the pipeline's results; do not edit.\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    f.write("\\newcommand{\\claimwarnings}{" + "".join(f"\\item {w}" for w in warnings) + "}\n")
    f.write(f"\\newcommand{{\\claimcount}}{{{len(warnings)}}}\n")
print(f"wrote data/numbers.tex ({len(macros)} numbers, {len(warnings)} claims to revisit)")
