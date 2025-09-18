
"""
Set 1) Cross-well (A1..D6) with subsites collapsed  -> absolute + per-well relative
Set 2) Single well (16 electrodes relabeled 1–16)   -> absolute + per-electrode relative

Dependencies: pandas, numpy, matplotlib, scipy (optional)

Author: Tracy Huang
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm


# SETTINGS
CSV_PATH = Path("/Users/tracy/Downloads/Bai Lab Downloads/spike_counts.csv")
BIN_SEC  = 180
OUTDIR   = Path("heatmap_outputs"); OUTDIR.mkdir(exist_ok=True)

# Color & contrast
CMAP = "turbo"                 # 'turbo'|'viridis'|'magma'|'inferno'
USE_PERCENTILE_CLIP = True
PCTL_MIN, PCTL_MAX   = 5, 99.5
USE_LOG_SCALE        = False
LOG_EPS              = 1.0     # added before log if using log scale

# Layout & labels
THIN_X_LABELS_STEP   = 0       # 0/1 show all; 2 shows every other, etc.
WIDTH_SCALE          = 0.8     # column width scaling
INVERT_TIME_AXIS     = True    # True => earliest bin at bottom (matches your past figures)

# Single-well which to plot for Set 2
WELL_CHOICE          = "A1"    # change to any of A1..D6; fallback kicks in if not found


# Helpers
def _clean_colname(c: str) -> str:
    c = str(c).strip()
    c = c.replace("–", "-").replace("—", "-")          # normalize dashes
    c = re.sub(r"[\u200b\u200c\u200d\u2060]", "", c)   # zero-width chars
    c = re.sub(r"\s+", " ", c)                         # collapse spaces
    c = re.sub(r"\s*-\s*", "-", c)                     # "A1- 11" -> "A1-11"
    return c

def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [_clean_colname(c) for c in df.columns]

    # Find time columns robustly
    start_candidates = [c for c in df.columns if re.search(r"interval\s*start", c, flags=re.I)]
    end_candidates   = [c for c in df.columns if re.search(r"interval\s*end",   c, flags=re.I)]
    if not start_candidates:
        start_candidates = [c for c in df.columns if re.search(r"\btime\b|\bstart\b", c, flags=re.I)]
    if not start_candidates:
        raise ValueError("Could not find a time-start column (e.g., 'Interval Start (S)').")

    start_col = start_candidates[0]
    df[start_col] = pd.to_numeric(df[start_col], errors="coerce")
    if end_candidates:
        end_col = end_candidates[0]
        df[end_col] = pd.to_numeric(df[end_col], errors="coerce")

    df = df[df[start_col].notna()].copy()
    df[start_col] = df[start_col].astype(int)
    if end_candidates:
        df[end_col] = df[end_col].astype(int)

    # Standardize names we use downstream
    df = df.rename(columns={start_col: "Interval Start (S)"})
    if end_candidates:
        df = df.rename(columns={end_candidates[0]: "Interval End (S)"})
    return df

def detect_columns(df: pd.DataFrame):
    """Return (macro_wells, subsite_cols) where:
       - macro_wells are A1..D6 totals if present (e.g., 'A1', 'B3', ...)
       - subsite_cols are A1_11..D6_44 or A1-11..D6-44 if present
    """
    # macro wells A1..D6
    macro_wells = [c for c in df.columns if re.match(r"^[ABCD][1-6]$", c)]
    # subsites: allow underscore or hyphen
    subsite_cols = [c for c in df.columns if re.match(r"^[ABCD][1-6][_ -][1-4][1-4]$", c)]
    return sorted(macro_wells), sorted(subsite_cols)

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    if cols:
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[cols] = df[cols].fillna(0.0)

def sum_numeric(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    return frame[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).fillna(0.0)

def bin_and_sum(df: pd.DataFrame, device_cols: list[str], bin_sec: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["time_bin"] = (tmp["Interval Start (S)"] // bin_sec) * bin_sec
    grouped = tmp.groupby("time_bin", sort=True)[device_cols].sum().fillna(0.0)
    return grouped

def make_y_labels(time_bins: pd.Index, bin_sec: int) -> list[str]:
    # e.g., "0–3 min", "3–6 min", ...
    return [f"{t//60}–{(t+bin_sec)//60} min" for t in time_bins]

def per_column_minmax(mat: np.ndarray) -> np.ndarray:
    """Scale each column to [0,1] so within-device changes pop over time."""
    m = np.asarray(mat, dtype=float)
    col_min = np.nanmin(m, axis=0, keepdims=True)
    col_max = np.nanmax(m, axis=0, keepdims=True)
    rng = np.where((col_max - col_min) == 0, 1.0, (col_max - col_min))
    return (m - col_min) / rng

def build_norm(mat: np.ndarray):
    m = np.asarray(mat, dtype=float)
    if USE_LOG_SCALE:
        m = m + LOG_EPS
        vmin = np.percentile(m, PCTL_MIN) if USE_PERCENTILE_CLIP else np.nanmin(m)
        vmax = np.percentile(m, PCTL_MAX) if USE_PERCENTILE_CLIP else np.nanmax(m)
        vmin = max(vmin, LOG_EPS)
        if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-12
        return LogNorm(vmin=vmin, vmax=vmax), "Total spikes (log)"
    else:
        vmin = np.percentile(m, PCTL_MIN) if USE_PERCENTILE_CLIP else np.nanmin(m)
        vmax = np.percentile(m, PCTL_MAX) if USE_PERCENTILE_CLIP else np.nanmax(m)
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-12
        return Normalize(vmin=vmin, vmax=vmax), f"Total spikes (per {BIN_SEC//60} min)"

def thin_labels(labels: list[str], step: int) -> list[str]:
    if step and step > 1:
        return [lab if i % step == 0 else "" for i, lab in enumerate(labels)]
    return labels

# Plot core (time on Y)
def heatmap_cellwise(
    mat: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    outpath: Path,
    cmap_name: str = CMAP,
    norm=None,
    cbar_label: str = "",
    width_scale: float = WIDTH_SCALE,
):
    mat = np.asarray(mat, dtype=float)
    n_rows, n_cols = mat.shape

    fig_w = max(18, min(40, n_cols * width_scale))
    fig_h = max(7,  min(26, n_rows * 1.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if norm is None:
        norm, cbar_label = build_norm(mat)

    data = mat + (LOG_EPS if isinstance(norm, LogNorm) else 0.0)
    im = ax.imshow(data, aspect='auto', interpolation='nearest',
                   cmap=plt.get_cmap(cmap_name), norm=norm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    # x = devices, y = time bins
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(thin_labels(x_labels, THIN_X_LABELS_STEP), rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels)

    # subtle grid
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='w', linewidth=0.35)
    ax.tick_params(which='minor', length=0)

    ax.set_xlabel("Devices")
    ax.set_ylabel("Time (minutes)")
    ax.set_title(title)

    if INVERT_TIME_AXIS:
        ax.invert_yaxis()  # earliest bin at bottom

    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")


# Build datasets
def build_cross_well_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return binned dataframe with columns A1..D6 (collapse subsites if needed)."""
    macro, subs = detect_columns(df)
    # If macro totals exist, use them. Otherwise sum subsites per well.
    if macro:
        cols = macro
        coerce_numeric(df, cols)
        binned = bin_and_sum(df, cols, BIN_SEC)
        return binned[sorted(binned.columns)]
    else:
        # Derive A1..D6 by summing A1_11..A1_44 etc.
        wells = sorted({re.split(r"[_ -]", c)[0] for c in subs})
        out = {}
        for w in wells:
            wcols = [c for c in subs if c.startswith(w + "_") or c.startswith(w + "-")]
            out[w] = sum_numeric(df, wcols)
        merged = pd.concat([df[["Interval Start (S)"]], pd.DataFrame(out)], axis=1)
        binned = bin_and_sum(merged, sorted(out.keys()), BIN_SEC)
        return binned[sorted(binned.columns)]

def build_single_well_matrix(df: pd.DataFrame, well: str) -> pd.DataFrame:
    """Return binned dataframe with 16 electrodes for the chosen well, relabeled 1..16."""
    _, subs = detect_columns(df)
    # Accept underscore or hyphen naming
    wcols = sorted([c for c in subs if re.match(rf"^{re.escape(well)}[_ -][1-4][1-4]$", c)])
    if not wcols:
        # graceful fallback to first available well
        if not subs:
            raise ValueError("No electrode-level columns found (e.g., A1_11..A1_44).")
        fallback = sorted({re.split(r'[_ -]', c)[0] for c in subs})[0]
        print(f"[info] Well '{well}' not found; using '{fallback}' instead.")
        well = fallback
        wcols = sorted([c for c in subs if re.match(rf"^{re.escape(well)}[_ -][1-4][1-4]$", c)])

    coerce_numeric(df, wcols)
    binned = bin_and_sum(df, wcols, BIN_SEC)

    # Relabel to 1..N (should be 16)
    rename_map = {old: str(i+1) for i, old in enumerate(binned.columns)}
    binned = binned.rename(columns=rename_map)
    return binned


# Make both figure sets
def plot_abs_and_relative(df_binned: pd.DataFrame, title_stem: str, stem: str):
    y_labels = make_y_labels(df_binned.index, BIN_SEC)
    X = list(df_binned.columns)
    M_abs = df_binned.values.astype(float)

    # Absolute (global scale)
    heatmap_cellwise(M_abs, X, y_labels,
                     f"{title_stem} — absolute",
                     OUTDIR / f"{stem}_absolute.png")

    # Per-column relative (min–max within each device)
    M_rel = per_column_minmax(M_abs)
    heatmap_cellwise(M_rel, X, y_labels,
                     f"{title_stem} — per-device RELATIVE (0–1)",
                     OUTDIR / f"{stem}_relative.png",
                     norm=Normalize(vmin=0, vmax=1),
                     cbar_label="Relative level (0–1)")

# MAIN
if __name__ == "__main__":
    df = load_and_clean(CSV_PATH)

    # ---- Set 1: Cross-well (A1..D6) ----
    cross_df = build_cross_well_matrix(df)
    plot_abs_and_relative(cross_df, "Cross-well (A1–D6, subsites collapsed)", "set1_cross_well")

    # ---- Set 2: Single well (16 electrodes relabeled 1–16) ----
    single_df = build_single_well_matrix(df, WELL_CHOICE)
    plot_abs_and_relative(single_df, f"Single well {WELL_CHOICE} — electrodes 1–16", f"set2_{WELL_CHOICE}_electrodes")

    print("Done.")
