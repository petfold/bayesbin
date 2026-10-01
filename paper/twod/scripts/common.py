"""Configuration and data loading shared by the pipeline's scripts.

Everything the paper reports is computed from two event sources, read where Worldwatch keeps
them (override with environment variables):

- WW_ARCHIVE: Worldwatch's Parquet archive (the daily backup); news events are the
  `gdelt_events` stream of its `bins` table, counts per H3 cell and time bin.
- WW_USGS: the USGS catalogue as CSV files (Worldwatch's research/replay_changepoint/fetch_usgs.py).

TWOD_RES sets the finest H3 resolution (default 3, the archive's); TWOD_QUICK=1 runs smaller
prototype sizes for a fast check of the pipeline.
"""

import csv
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # paper/twod
DATA = ROOT / "data"  # results the paper reads (committed)
BUILD = ROOT / "build"  # intermediate arrays (not committed)
FIGURES = ROOT / "figures"
ARCHIVE = Path(os.environ.get("WW_ARCHIVE", Path.home() / "worldwatch-archive"))
USGS = Path(os.environ.get("WW_USGS", Path.home() / ".cache" / "worldwatch-research" / "usgs"))
RES = int(os.environ.get("TWOD_RES", "3"))
QUICK = os.environ.get("TWOD_QUICK", "") not in ("", "0")

DATASETS = {
    "gdelt": {"label": "News events (GDELT)", "short": "news events"},
    "usgs": {"label": "Earthquakes (USGS, M ≥ 1)", "short": "earthquakes"},
}

# places looked up in the best partition (lat, lng)
PLACES = {
    "N Pacific (30°N 150°W)": (30.0, -150.0),
    "central Pacific (0° 160°W)": (0.0, -160.0),
    "Sahara (23°N 10°E)": (23.0, 10.0),
    "Siberia (65°N 100°E)": (65.0, 100.0),
    "London": (51.51, -0.13),
    "Delhi": (28.61, 77.21),
    "Lagos": (6.52, 3.38),
    "Kansas (centre of the US)": (39.83, -98.58),
    "Los Angeles": (34.05, -118.25),
    "Aleutians (52°N 176°W)": (51.9, -176.6),
    "Hawaii (Kīlauea)": (19.4, -155.3),
    "Tokyo": (35.68, 139.69),
}


def load_events(name):
    """[(time in s, H3 cell at RES, count)], one row per record."""
    import h3

    if name == "gdelt":
        import duckdb

        rows = duckdb.sql(f"SELECT cell, bin_start, n FROM '{ARCHIVE}/bins/*.parquet' "
                          "WHERE stream_id = 'gdelt_events'").fetchall()
        if not rows:
            raise SystemExit(f"no gdelt_events rows under {ARCHIVE}/bins")
        res = h3.get_resolution(rows[0][0])
        if RES > res:
            raise SystemExit(f"the archive's news cells are H3 resolution {res}; TWOD_RES={RES} is finer")
        to = (lambda c: c) if RES == res else (lambda c: h3.cell_to_parent(c, RES))
        return [(float(t), to(c), float(n)) for c, t, n in rows]
    if name == "usgs":
        out = []
        files = sorted(glob.glob(str(USGS / "*.csv")))
        if not files:
            raise SystemExit(f"no USGS CSV files under {USGS}")
        for f in files:
            for r in csv.DictReader(open(f)):
                ts = datetime.fromisoformat(r["time"].replace("Z", "+00:00")).timestamp()
                out.append((ts, h3.latlng_to_cell(float(r["latitude"]), float(r["longitude"]), RES), 1.0))
        return out
    raise ValueError(name)


def span(events):
    """(first, last) time and the span in days."""
    t = [e[0] for e in events]
    return min(t), max(t), (max(t) - min(t)) / 86400


def date(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def write_json(name, obj):
    DATA.mkdir(exist_ok=True)
    path = DATA / name
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def read_json(name):
    return json.loads((DATA / name).read_text())
