import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_PATH = Path(
    "fifty_one/measurements/data/length_analysis_new_split_shai_exuviae_with_yolo.csv"
)
OUTPUT_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Ensure the expected columns exist
required_cols = {"pixel_rel_diff", "real_length_rel_diff", "lobster_size"}
missing = required_cols.difference(df.columns)
if missing:
    raise ValueError(f"Missing expected columns in CSV: {missing}")

# -----------------------------------------------------------------------------
# Basic statistics
# -----------------------------------------------------------------------------

corr = df["pixel_rel_diff"].corr(df["real_length_rel_diff"])
ratio = (df["real_length_rel_diff"] / df["pixel_rel_diff"]).rename("ratio")

stats_msg = (
    f"Pearson r = {corr:.3f}\n"
    f"Mean(real/pixel) = {ratio.mean():.3f}\n"
    f"Median(real/pixel) = {ratio.median():.3f}\n"
)
print(stats_msg)

# Save stats to a text file for quick reference
(OUTPUT_DIR / "pixel_vs_real_stats.txt").write_text(stats_msg)

# -----------------------------------------------------------------------------
# Identify and exclude outliers using the 1.5×IQR rule on the ratio
# -----------------------------------------------------------------------------
ratio_col = ratio  # pandas Series created above
q1, q3 = ratio_col.quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
is_outlier = (ratio_col < low) | (ratio_col > high)

n_outliers = is_outlier.sum()
print(f"Detected {n_outliers} outliers via IQR rule (ratio < {low:.2f} or > {high:.2f}).")

# Save outlier indices for record
(df.loc[is_outlier, ["image_name", "pixel_rel_diff", "real_length_rel_diff"]]
    .to_csv(OUTPUT_DIR / "outliers.csv", index=False))

# -----------------------------------------------------------------------------
# Scatter plot without outliers
# -----------------------------------------------------------------------------
clean_df = df.loc[~is_outlier].copy()
clean_corr = clean_df["pixel_rel_diff"].corr(clean_df["real_length_rel_diff"])
print(f"Correlation after removing outliers: r = {clean_corr:.3f}")

plt.figure(figsize=(8, 6))
ax = sns.scatterplot(
    data=clean_df,
    x="pixel_rel_diff",
    y="real_length_rel_diff",
    hue="lobster_size",
    palette="Set2",
    alpha=0.8,
)
# 1:1 reference line
lims = [0, max(clean_df["pixel_rel_diff"].max(), clean_df["real_length_rel_diff"].max()) * 1.05]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Relative pixel difference (fraction)")
ax.set_ylabel("Relative real-length difference (fraction)")
ax.set_title(
    f"Without outliers (n={len(clean_df)})  r = {clean_corr:.2f}"
)
ax.legend(title="Lobster size", loc="upper left")
plt.tight_layout()

clean_plot_path = OUTPUT_DIR / "pixel_vs_real_diff_scatter_no_outliers.png"
plt.savefig(clean_plot_path, dpi=300)
print(f"Saved cleaned scatter plot to {clean_plot_path.resolve()}")

# -----------------------------------------------------------------------------
# Density plot of the ratio
# -----------------------------------------------------------------------------

plt.figure(figsize=(6, 4))
ax = sns.kdeplot(ratio, fill=True, color="steelblue")
ax.axvline(1, ls="--", color="k", label="Ideal")
ax.set_xlabel("real_length_rel_diff / pixel_rel_diff")
ax.set_title("Distribution of ratio (ideal = 1)")
ax.legend()
plt.tight_layout()

ratio_plot_path = OUTPUT_DIR / "ratio_distribution.png"
plt.savefig(ratio_plot_path, dpi=300)
print(f"Saved ratio distribution plot to {ratio_plot_path.resolve()}")

# -----------------------------------------------------------------------------
# Per-pond summary (with and without outliers)
# -----------------------------------------------------------------------------

def extract_pond(name: str) -> str:
    match = re.search(r"GX\d{6}", name)
    return match.group(0) if match else "unknown"

df["pond_id"] = df["image_name"].apply(extract_pond)
clean_df["pond_id"] = clean_df["image_name"].apply(extract_pond)

summary_cols = [
    "pond_id",
    "n_samples",
    "mean_ratio",
    "median_ratio",
    "std_ratio",
    "corr_px_real",
]

rows = []
for pond, sub in df.groupby("pond_id"):
    r = (sub["real_length_rel_diff"] / sub["pixel_rel_diff"]).dropna()
    corr = sub["pixel_rel_diff"].corr(sub["real_length_rel_diff"])
    rows.append(
        {
            "pond_id": pond,
            "n_samples": len(sub),
            "mean_ratio": r.mean(),
            "median_ratio": r.median(),
            "std_ratio": r.std(),
            "corr_px_real": corr,
        }
    )
per_pond_all = pd.DataFrame(rows)

rows_clean = []
for pond, sub in clean_df.groupby("pond_id"):
    r = (sub["real_length_rel_diff"] / sub["pixel_rel_diff"]).dropna()
    corr = sub["pixel_rel_diff"].corr(sub["real_length_rel_diff"])
    rows_clean.append(
        {
            "pond_id": pond,
            "n_samples": len(sub),
            "mean_ratio": r.mean(),
            "median_ratio": r.median(),
            "std_ratio": r.std(),
            "corr_px_real": corr,
        }
    )
per_pond_clean = pd.DataFrame(rows_clean)

per_pond_all.to_csv(OUTPUT_DIR / "per_pond_stats_all.csv", index=False)
per_pond_clean.to_csv(OUTPUT_DIR / "per_pond_stats_no_outliers.csv", index=False)

print("Saved per-pond summaries (with and without outliers) to results/analysis/.")

# -----------------------------------------------------------------------------
# Add pond shape and compute per-pond & size summaries
# -----------------------------------------------------------------------------

def pond_shape(pond: str) -> str:
    return "circle" if pond == "GX010191" else "square"

df["pond_shape"] = df["pond_id"].apply(pond_shape)
clean_df["pond_shape"] = clean_df["pond_id"].apply(pond_shape)

# Helper to compute stats for any sub-DataFrame
def _stats(sub):
    r = (sub["real_length_rel_diff"] / sub["pixel_rel_diff"]).dropna()
    return {
        "n_samples": len(sub),
        "mean_ratio": r.mean(),
        "median_ratio": r.median(),
        "std_ratio": r.std(),
        "corr_px_real": sub["pixel_rel_diff"].corr(sub["real_length_rel_diff"]),
    }

# Per pond & size (all)
rows_pond_size_all = []
for (pond, size), sub in df.groupby(["pond_id", "lobster_size" ]):
    d = _stats(sub)
    d.update({"pond_id": pond, "lobster_size": size})
    rows_pond_size_all.append(d)
per_pond_size_all = pd.DataFrame(rows_pond_size_all)
per_pond_size_all.to_csv(OUTPUT_DIR / "per_pond_size_stats_all.csv", index=False)

# Per pond & size (clean)
rows_pond_size_clean = []
for (pond, size), sub in clean_df.groupby(["pond_id", "lobster_size" ]):
    d = _stats(sub)
    d.update({"pond_id": pond, "lobster_size": size})
    rows_pond_size_clean.append(d)
per_pond_size_clean = pd.DataFrame(rows_pond_size_clean)
per_pond_size_clean.to_csv(OUTPUT_DIR / "per_pond_size_stats_no_outliers.csv", index=False)

# Per pond shape overall
shape_rows = []
for shape, sub in df.groupby("pond_shape"):
    d = _stats(sub)
    d.update({"pond_shape": shape})
    shape_rows.append(d)
per_shape_all = pd.DataFrame(shape_rows)
per_shape_all.to_csv(OUTPUT_DIR / "per_shape_stats_all.csv", index=False)

print("Saved pond-size and shape summaries to results/analysis/.") 