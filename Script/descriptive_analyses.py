# Reproduce Panel E (fraction repeat vs. signed LLR)
# for **high- vs low-confidence subjects** (median split) in Exp C & D

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------- 1. Load & tidy raw data -------------
DATA_CSV = Path("./Data/rouault2022_data.csv")
OUTPUT_DIR = Path("./Output/figs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_CSV)

# Parse the stringified NumPy-style arrays into real ndarrays
if df["sample_log_likelihood_ratio"].dtype == object:
    df["sample_log_likelihood_ratio"] = df["sample_log_likelihood_ratio"].apply(
        lambda s: np.fromstring(s.strip("[]"), sep=" ")
    )

# Signed LLR (Σ LLR signed by previous response)
df["llr_total"] = df["sample_log_likelihood_ratio"].apply(np.sum)
df["signed_llr"] = df["llr_total"] * np.where(df["response_before"] == 1, 1, -1)

# Repeat indicator (1 = repeat, 0 = switch)
df["repeat"] = (df["response_before"] == df["response_after"]).astype(int)

# Unique participant ID (exp + subj)
df["pid"] = df["exp"].astype(str) + "_" + df["subj"].astype(str)

# ------------- 2. Median-split subjects by confidence -------------
# Mean trial-level confidence_after per participant
conf_mean = df.groupby("pid")["confidence_after"].mean()
median_conf = conf_mean.median()
conf_group_map = (conf_mean >= median_conf).map({True: "high", False: "low"})

df["conf_group"] = df["pid"].map(conf_group_map)

# ------------- 3. Bin signed LLR into 8 quantile bins -------------
# (equal-N bins give balanced error bars)
df["llr_bin"] = pd.qcut(df["signed_llr"], 8, labels=False)

# ------------- 4. Aggregate: fraction repeat & SEM -------------
grp = (
    df.groupby(["conf_group", "llr_bin"])
      .agg(
          mean_repeat=("repeat", "mean"),
          se_repeat=("repeat", lambda x: x.std(ddof=0) / np.sqrt(len(x))),
          bin_center=("signed_llr", "mean"),
      )
      .reset_index()
)

# ------------- 5. Plot: Repeat vs Signed LLR -------------
plt.figure(figsize=(6, 4))
for grp_name in grp["conf_group"].unique():
    sub = grp[grp["conf_group"] == grp_name]
    plt.errorbar(
        sub["bin_center"],
        sub["mean_repeat"],
        yerr=sub["se_repeat"],
        fmt="o-",
        capsize=3,
        label=f"{grp_name.capitalize()} confidence",
    )

plt.axhline(0.5, linestyle="--")
plt.axvline(0, color="black")
plt.xlabel("evidence direction (logLR)")
plt.ylabel("fraction repeat")
plt.title("Fraction repeat vs. signed LLR\nHigh vs Low confidence groups (Exp C & D)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "repeat_vs_signedLLR_confidence_groups.png", dpi=300)
plt.show()

# --- Plotting the Histogram of Signed LLR ---
plt.figure(figsize=(10, 6))
plt.hist(df['signed_llr'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of Signed Log-Likelihood Ratio', fontsize=16)
plt.xlabel('Signed LLR', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_signed_llr.png", dpi=300)
plt.show()

# --- Plotting the Distribution of Mean Confidence Across Participants ---
plt.figure(figsize=(10, 6))
plt.hist(conf_mean, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.axvline(median_conf, linestyle='--', color='black', label=f'Median = {median_conf:.2f}')
plt.title('Distribution of Mean Confidence Across Participants', fontsize=16)
plt.xlabel('Mean Confidence (confidence_after)', fontsize=12)
plt.ylabel('Number of Participants', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "hist_mean_confidence.png", dpi=300)
plt.show()