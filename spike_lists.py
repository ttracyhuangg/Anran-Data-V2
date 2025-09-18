from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# === SETTINGS ===
CSV_PATH = Path("/Users/tracy/Downloads/Bai Lab Downloads/spike_list.csv")
OUTDIR   = Path("raster_outputs"); OUTDIR.mkdir(exist_ok=True, parents=True)

WELL = "A1"   # change to any of A1..D6 later when you want
LINEWIDTH = 0.8
FIGSIZE = (12, 6)

# Optional stimulation windows to overlay as orange bars (start_s, end_s) in seconds.
# If you have them, fill this list, e.g. [(10.0, 11.0), (20.0, 21.0)]
STIM_WINDOWS = []  # e.g., [(0.0, 1.0), (5.0, 6.0)]

# --- Helpers ---
def parse_spike_list(csv_path: Path, usecols=("Time (s)", "Electrode")) -> pd.DataFrame:
    """Load spike list CSV and return clean DataFrame with numeric time and well_subsite strings."""
    # Load necessary cols only; low_memory=False keeps types consistent
    df = pd.read_csv(csv_path, usecols=list(usecols), low_memory=False)

    # Coerce numeric time and drop rows that are metadata (NaNs / non-numeric)
    df["Time (s)"] = pd.to_numeric(df["Time (s)"], errors="coerce")
    df = df[df["Time (s)"].notna()].copy()

    # Standardize electrode label formatting, accept A1_11 or A1-11
    df["Electrode"] = (
        df["Electrode"]
        .astype(str)
        .str.strip()
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.replace(r"\s*-\s*", "-", regex=True)
    )

    # Keep only well_subsite patterns like A1_11 or A1-11
    pat = re.compile(r"^[A-D][1-6][_ -][1-4][1-4]$")
    df = df[df["Electrode"].str.match(pat, na=False)].copy()

    # Normalize to underscore form A1_11 for easy parsing
    df["Electrode"] = df["Electrode"].str.replace("-", "_", regex=False)

    return df

def map_subsite_to_index(subsite: str) -> int:
    """
    Convert 'A1_11' style to an index 1..16 in row-major order:
      11→1, 12→2, 13→3, 14→4, 21→5, ..., 44→16
    """
    # subsite like "A1_34"
    rowcol = subsite.split("_")[1]  # "34"
    r, c = int(rowcol[0]), int(rowcol[1])
    return (r - 1) * 4 + c  # 1..16

def raster_for_well(df_events: pd.DataFrame, well: str) -> pd.DataFrame:
    """Filter one well (e.g., 'A1') and map its subsites to channel numbers 1..16."""
    wpat = re.compile(rf"^{re.escape(well)}_([1-4][1-4])$")
    sub = df_events[df_events["Electrode"].str.match(rf"^{re.escape(well)}_")].copy()
    if sub.empty:
        # helpful message: list available wells in the file
        wells_found = sorted(set(df_events["Electrode"].str.split("_").str[0]))
        raise ValueError(f"No spikes found for well '{well}'. Wells present: {wells_found[:24]} ...")

    # Map to channel 1..16
    sub["Channel"] = sub["Electrode"].apply(map_subsite_to_index)
    # Sort by time for neat plotting
    sub = sub.sort_values("Time (s)").reset_index(drop=True)
    return sub

def plot_raster(sub: pd.DataFrame, well: str, outpath: Path,
                stim_windows=None, lw=LINEWIDTH, figsize=FIGSIZE):
    """Draw a raster: x=time, y=channel (1..16), tick=spike."""
    fig, ax = plt.subplots(figsize=figsize)

    # Draw each spike as a short vertical line centered at integer y
    # For speed: vectorized vlines by channel
    for ch in range(1, 17):
        times = sub.loc[sub["Channel"] == ch, "Time (s)"].to_numpy()
        if times.size:
            ax.vlines(times, ch - 0.45, ch + 0.45, linewidth=lw)

    # Optional: overlay stimulation windows as orange bars
    if stim_windows:
        ymin, ymax = 0.5, 16.5
        for (t0, t1) in stim_windows:
            ax.hlines(y=np.linspace(1, 16, 16), xmin=t0, xmax=t1, colors="orange", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Electrode / Channel (1–16)")
    ax.set_title(f"Raster plot of spikes — Well {well}")
    ax.set_ylim(0.5, 16.5)
    ax.set_yticks(range(1, 17))
    ax.set_xlim(left=max(0.0, sub["Time (s)"].min() - 0.05), right=sub["Time (s)"].max() + 0.05)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")

# --- Main ---
if __name__ == "__main__":
    df_events = parse_spike_list(CSV_PATH)
    sub = raster_for_well(df_events, WELL)
    out = OUTDIR / f"raster_{WELL}.png"
    plot_raster(sub, WELL, outpath=out, stim_windows=STIM_WINDOWS)
    print("Done.")
