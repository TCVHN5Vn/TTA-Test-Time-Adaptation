"""
Parser for CASAS smart-home data, producing the same binary sensor matrix the rest of
this pipeline expects.

CASAS releases vary in format across testbeds and years. The common denominator is one
event per line, whitespace-separated, roughly:

    2010-11-04 00:03:50.209589  M003  ON
    2010-11-04 00:03:57.399391  M005  ON   Sleeping begin
    2010-11-04 00:15:00.000000  M003  OFF  Sleeping end

Some releases use a single `Activity="..."` column, some annotate only transitions with
`begin`/`end`, some carry an extra sensor-type column. This parser accepts all of those
and reduces each to:

    time (s since midnight), year_day, activity, sensor_id_<X> ... (binary, 0/1)

sampled on a fixed grid so the result is interchangeable with the Tapia matrices.

Binary sensors (motion M*, door D*, item I*) are held at their last state between events,
which is the convention the existing feature code assumes. Analogue channels (temperature
T*, light L*) are ignored: the layout-invariant features are defined over binary
activations, and mixing modalities would change what is being compared.

Usage
-----
    python src/casas_parse.py ~/Desktop/casas --out data/raw --resample 30
"""
import os, re, sys, argparse, glob
import numpy as np, pandas as pd

TS = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
BINARY_PREFIX = ("M", "D", "I", "MA", "DO", "AD1-A", "E")   # motion, door, item
ON = {"ON", "OPEN", "PRESENT", "1", "TRUE", "ABSENT"}       # ABSENT handled below
OFF = {"OFF", "CLOSE", "CLOSED", "ABSENT", "0", "FALSE"}


def _state(v):
    v = v.strip().upper()
    if v in ("ON", "OPEN", "PRESENT", "1", "TRUE"):  return 1
    if v in ("OFF", "CLOSE", "CLOSED", "0", "FALSE"): return 0
    if v == "ABSENT":  return 0
    try:                                              # numeric analogue channel
        float(v); return None
    except ValueError:
        return None


def read_events(path):
    """Yield (timestamp, sensor, state, activity_text) for one CASAS file."""
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip(): continue
            m = TS.match(line)
            if not m: continue
            rest = line[m.end():].split()
            if len(rest) < 2: continue
            ts = pd.to_datetime(m.group(1) + " " + m.group(2), errors="coerce")
            if pd.isna(ts): continue
            sensor, value = rest[0], rest[1]
            act = " ".join(rest[2:]).strip()
            rows.append((ts, sensor, value, act))
    return rows


def activity_series(rows):
    """Resolve CASAS's begin/end annotation style into a per-event activity label."""
    acts, current = [], ""
    for _, _, _, a in rows:
        if not a:
            acts.append(current); continue
        low = a.lower()
        # 'Activity="X"' or 'X begin' / 'X end'
        m = re.search(r'activity\s*=\s*"?([^"\s]+)', low)
        if m:
            acts.append(m.group(1)); current = m.group(1); continue
        if low.endswith("begin"):
            current = a[: -len("begin")].strip().lower(); acts.append(current)
        elif low.endswith("end"):
            acts.append(current); current = ""
        else:
            current = low.split()[0]; acts.append(current)
    return acts


def to_matrix(path, resample_s=30, min_count=50):
    rows = read_events(path)
    if not rows:
        return None
    acts = activity_series(rows)
    df = pd.DataFrame({"ts": [r[0] for r in rows],
                       "sensor": [r[1] for r in rows],
                       "value": [r[2] for r in rows],
                       "activity": acts}).sort_values("ts").reset_index(drop=True)
    df["state"] = df.value.map(_state)
    df = df[df.state.notna()]                           # binary channels only
    df = df[df.sensor.str.upper().str.startswith(BINARY_PREFIX)]
    if df.empty: return None

    # fixed grid, last-state-carried-forward
    t0, t1 = df.ts.min().floor("s"), df.ts.max().ceil("s")
    grid = pd.date_range(t0, t1, freq=f"{resample_s}s")
    wide = (df.pivot_table(index="ts", columns="sensor", values="state", aggfunc="last")
              .reindex(grid.union(df.ts.unique())).sort_index().ffill()
              .reindex(grid).fillna(0).astype(np.int8))
    act = (df.set_index("ts").activity
             .reindex(grid.union(df.ts.unique())).sort_index().ffill()
             .reindex(grid).fillna(""))

    keep = act.astype(str).str.len() > 0
    wide, act = wide[keep.values], act[keep.values]
    if len(wide) < 100: return None

    vc = act.value_counts()
    common = set(vc[vc >= min_count].index) - {"", "other", "none"}
    mask = act.isin(common).values
    wide, act = wide[mask], act[mask]
    if len(wide) < 100: return None

    idx = wide.index
    out = pd.DataFrame({"time": idx.hour * 3600 + idx.minute * 60 + idx.second,
                        "year_day": idx.dayofyear,
                        "activity": act.values})
    for c in wide.columns:
        out[f"sensor_id_{c}"] = wide[c].values
    return out.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder containing one subfolder per CASAS home")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    ap.add_argument("--resample", type=int, default=30)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    cands = []
    for p in sorted(glob.glob(os.path.join(a.root, "**", "*"), recursive=True)):
        if os.path.isfile(p) and os.path.basename(p).lower() in (
                "data", "data.txt", "rawdata.txt", "events.txt") or p.endswith(".txt"):
            cands.append(p)
    if not cands:
        print(f"no candidate files under {a.root}"); sys.exit(1)

    print(f"{'file':<48}{'rows':>8}{'sensors':>9}{'activities':>12}")
    made = 0
    for p in cands:
        try:
            m = to_matrix(p, a.resample)
        except Exception as e:
            print(f"{os.path.basename(p)[:46]:<48}  skipped ({type(e).__name__})"); continue
        if m is None:
            continue
        home = os.path.basename(os.path.dirname(p)) or os.path.splitext(os.path.basename(p))[0]
        f = os.path.join(a.out, f"casas_{home}.csv")
        m.to_csv(f, index=False); made += 1
        ns = sum(c.startswith("sensor_id_") for c in m.columns)
        print(f"{home[:46]:<48}{len(m):>8}{ns:>9}{m.activity.nunique():>12}   -> {os.path.basename(f)}")
    print(f"\nwrote {made} home(s) to {a.out}")
    if made:
        print("next: python src/casas_pairs.py   (runs every ordered pair)")
