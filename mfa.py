import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem


# ======================================================
# Global config
# ======================================================

ABS_DIFF_BINS = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
ABS_DIFF_LABELS = [0, 1, 2, 3, 4]

SIGNED_DIFF_BINS = np.arange(-4.5, 5, 1)
SIGNED_DIFF_LABELS = np.arange(-4, 5)


# ======================================================
# Helpers
# ======================================================

def ensure_outdir(dirname: str = "mfa_figures") -> str:
    os.makedirs(dirname, exist_ok=True)
    return dirname


def basename_from_path(path: str) -> str:
    return path[-24:-4] if len(path) >= 24 else os.path.splitext(os.path.basename(path))[0]


def groupby_obs(df: pd.DataFrame, by):
    """Explicit observed=True everywhere to avoid pandas 2.x warnings."""
    return df.groupby(by, observed=True)


def add_abs_diff_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_diff"] = np.abs(df["avgWTP_left"] - df["avgWTP_right"])
    df["diff_bin"] = pd.cut(df["abs_diff"], bins=ABS_DIFF_BINS, labels=ABS_DIFF_LABELS)
    return df


def add_signed_diff(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signed_diff"] = df["avgWTP_left"] - df["avgWTP_right"]
    return df


# ======================================================
# Plotting / Saving
# ======================================================

def plot_fixation_properties(df: pd.DataFrame):
    fig, axs = plt.subplots(1, 5, figsize=(25, 4))

    # --------------------------------------------------
    # (1) First fixation toward higher value
    # --------------------------------------------------
    first_fix = df[df["fix_num"] == 1].copy()
    first_fix["abs_diff"] = np.abs(first_fix["avgWTP_left"] - first_fix["avgWTP_right"])
    first_fix["larger_side"] = np.where(
        first_fix["avgWTP_left"] > first_fix["avgWTP_right"], 1, 2
    )
    first_fix["toward_larger"] = first_fix["location"] == first_fix["larger_side"]
    first_fix["diff_bin"] = pd.cut(
        first_fix["abs_diff"], bins=[0.5, 1.5, 2.5, 3.5, 4.5], labels=[1, 2, 3, 4]
    )

    summary = (
        first_fix
        .groupby("diff_bin", observed=True)
        .agg(mean_toward=("toward_larger", "mean"),
             sem_toward=("toward_larger", sem))
        .reset_index()
    )

    axs[0].errorbar(
        summary["diff_bin"].astype(int),
        summary["mean_toward"],
        yerr=summary["sem_toward"],
        fmt="o-",
        capsize=5,
    )
    axs[0].axhline(0.5, linestyle="--", color="gray")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0.75, 4.25)
    axs[0].set_xlabel("Best − Worst")
    axs[0].set_ylabel("P(First fixation to best)")
    axs[0].grid(True)

    # --------------------------------------------------
    # (2) Fixation duration by position
    # --------------------------------------------------
    middle_fix = df[(df["fix_num"] != 1) & (df["fix_num_rev"] != 1)]
    last_fix = df[df["fix_num_rev"] == 1]

    means = [
        first_fix["fix_dur"].mean(),
        middle_fix["fix_dur"].mean(),
        last_fix["fix_dur"].mean(),
    ]
    errors = [
        sem(first_fix["fix_dur"]),
        sem(middle_fix["fix_dur"]),
        sem(last_fix["fix_dur"]),
    ]

    axs[1].bar(["First", "Middle", "Last"], means, yerr=errors, capsize=5)
    axs[1].set_ylabel("Fixation duration (ms)")
    axs[1].grid(axis="y")

    # --------------------------------------------------
    # (3) Middle fixation duration vs |ΔV|
    # --------------------------------------------------
    middle_fix = add_abs_diff_bins(middle_fix)
    summary = (
        middle_fix
        .groupby("diff_bin", observed=True)
        .agg(mean_fix_dur=("fix_dur", "mean"),
             sem_fix_dur=("fix_dur", sem))
        .reset_index()
    )

    axs[2].errorbar(
        summary["diff_bin"].astype(int),
        summary["mean_fix_dur"],
        yerr=summary["sem_fix_dur"],
        fmt="o-",
        capsize=5,
    )
    axs[2].set_xlabel("Best − Worst")
    axs[2].set_ylabel("Middle fixation duration (ms)")
    axs[2].grid(True)

    # --------------------------------------------------
    # (4) First fixation duration vs |ΔV|
    # --------------------------------------------------
    first_fix = add_abs_diff_bins(first_fix)
    summary = (
        first_fix
        .groupby("diff_bin", observed=True)
        .agg(mean_fix_dur=("fix_dur", "mean"),
             sem_fix_dur=("fix_dur", sem))
        .reset_index()
    )

    axs[3].errorbar(
        summary["diff_bin"].astype(int),
        summary["mean_fix_dur"],
        yerr=summary["sem_fix_dur"],
        fmt="o-",
        capsize=5,
    )
    axs[3].set_xlabel("Best − Worst")
    axs[3].set_ylabel("First fixation duration (ms)")
    axs[3].grid(True)

    # --------------------------------------------------
    # (5) Net fixation duration vs signed ΔV
    # --------------------------------------------------
    df = add_signed_diff(df)
    df["signed_fix_dur"] = np.where(df["location"] == 1, df["fix_dur"], -df["fix_dur"])

    trial_summary = (
        df
        .groupby(["sub_id", "trial"], observed=True)
        .agg(signed_fix_dur=("signed_fix_dur", "sum"),
             signed_diff=("signed_diff", "first"))
        .reset_index()
    )

    trial_summary["diff_bin"] = pd.cut(
        trial_summary["signed_diff"],
        bins=SIGNED_DIFF_BINS,
        labels=SIGNED_DIFF_LABELS,
    )

    net = (
        trial_summary
        .groupby("diff_bin", observed=True)
        .agg(mean_net=("signed_fix_dur", "mean"),
             sem_net=("signed_fix_dur", sem))
        .reset_index()
    )

    axs[4].errorbar(
        net["diff_bin"].astype(int),
        net["mean_net"],
        yerr=net["sem_net"],
        fmt="o-",
        capsize=5,
    )
    axs[4].axhline(0, linestyle="--", color="gray")
    axs[4].set_xlabel("Left − Right")
    axs[4].set_ylabel("Net fixation duration (ms)")
    axs[4].grid(True)

    fig.tight_layout()
    plt.show()

def plot_basic_psychometrics(df: pd.DataFrame):
    fig, axs = plt.subplots(1, 3, figsize=(20, 4))

    first_fix = df[df["fix_num"] == 1].copy()
    first_fix = add_signed_diff(first_fix)

    # --------------------------------------------------
    # (6) Choice psychometric
    # --------------------------------------------------
    first_fix["choice_left"] = first_fix["choice"] == 0
    first_fix["diff_bin"] = pd.cut(
        first_fix["signed_diff"], bins=np.arange(-4, 4.25, 0.25)
    )

    prob = first_fix.groupby("diff_bin", observed=True)["choice_left"].mean()
    centers = first_fix.groupby("diff_bin", observed=True)["signed_diff"].mean()

    axs[0].plot(centers, prob, marker="o")
    axs[0].axhline(0.5, linestyle="--", color="gray")
    axs[0].set_xlabel("Left − Right")
    axs[0].set_ylabel("P(Choose left)")
    axs[0].grid(True)

    # --------------------------------------------------
    # (7) RT vs |ΔV|
    # --------------------------------------------------
    first_fix = add_abs_diff_bins(first_fix)
    rt = (
        first_fix
        .groupby("diff_bin", observed=True)
        .agg(mean_rt=("RT", "mean"), sem_rt=("RT", sem))
        .reset_index()
    )
    rt[["mean_rt", "sem_rt"]] *= 1000

    axs[1].errorbar(
        rt["diff_bin"].astype(int),
        rt["mean_rt"],
        yerr=rt["sem_rt"],
        fmt="o-",
        capsize=5,
    )
    axs[1].set_xlabel("Best − Worst")
    axs[1].set_ylabel("Response time (ms)")
    axs[1].grid(True)

    # --------------------------------------------------
    # (8) Number of fixations vs |ΔV|
    # --------------------------------------------------
    nfix = (
        first_fix
        .groupby("diff_bin", observed=True)
        .agg(mean_nfix=("fix_num_rev", "mean"),
             sem_nfix=("fix_num_rev", sem))
        .reset_index()
    )

    axs[2].errorbar(
        nfix["diff_bin"].astype(int),
        nfix["mean_nfix"],
        yerr=nfix["sem_nfix"],
        fmt="o-",
        capsize=5,
        color="tab:red",
    )
    axs[2].set_xlabel("Best − Worst")
    axs[2].set_ylabel("Number of fixations")
    axs[2].grid(True)

    fig.tight_layout()
    plt.show()

def save_fixation_properties(parent_dir: str, path: str):
    df = pd.read_csv(os.path.join(parent_dir, path))
    outdir = ensure_outdir()

    fig, axs = plt.subplots(1, 5, figsize=(25, 4))

    # --------------------------------------------------
    # (1) First fixation toward higher value
    # --------------------------------------------------
    first_fix = df[df["fix_num"] == 1].copy()
    first_fix["abs_diff"] = np.abs(first_fix["avgWTP_left"] - first_fix["avgWTP_right"])
    first_fix["larger_side"] = np.where(
        first_fix["avgWTP_left"] > first_fix["avgWTP_right"], 1, 2
    )
    first_fix["toward_larger"] = first_fix["location"] == first_fix["larger_side"]
    first_fix["diff_bin"] = pd.cut(
        first_fix["abs_diff"], bins=[0.5, 1.5, 2.5, 3.5, 4.5], labels=[1, 2, 3, 4]
    )

    summary = (
        groupby_obs(first_fix, "diff_bin")
        .agg(
            mean_toward=("toward_larger", "mean"),
            sem_toward=("toward_larger", sem),
        )
        .reset_index()
    )

    axs[0].errorbar(
        summary["diff_bin"].astype(int),
        summary["mean_toward"],
        yerr=summary["sem_toward"],
        fmt="o-",
        capsize=5,
    )
    axs[0].axhline(0.5, linestyle="--", color="gray")
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0.75, 4.25)
    axs[0].set_xlabel("Best − Worst")
    axs[0].set_ylabel("P(First fixation to best)")
    axs[0].grid(True)

    # --------------------------------------------------
    # (2) Fixation duration by type
    # --------------------------------------------------
    middle_fix = df[(df["fix_num"] != 1) & (df["fix_num_rev"] != 1)]
    last_fix = df[df["fix_num_rev"] == 1]

    means = [
        first_fix["fix_dur"].mean(),
        middle_fix["fix_dur"].mean(),
        last_fix["fix_dur"].mean(),
    ]
    errors = [
        sem(first_fix["fix_dur"]),
        sem(middle_fix["fix_dur"]),
        sem(last_fix["fix_dur"]),
    ]

    axs[1].bar(["First", "Middle", "Last"], means, yerr=errors, capsize=5)
    axs[1].set_ylabel("Fixation duration (ms)")
    axs[1].grid(axis="y")

    # --------------------------------------------------
    # (3) Middle fixation duration vs |ΔV|
    # --------------------------------------------------
    middle_fix = add_abs_diff_bins(middle_fix)

    summary = (
        groupby_obs(middle_fix, "diff_bin")
        .agg(
            mean_fix_dur=("fix_dur", "mean"),
            sem_fix_dur=("fix_dur", sem),
        )
        .reset_index()
    )

    axs[2].errorbar(
        summary["diff_bin"].astype(int),
        summary["mean_fix_dur"],
        yerr=summary["sem_fix_dur"],
        fmt="o-",
        capsize=5,
    )
    axs[2].set_xlabel("Best − Worst")
    axs[2].set_ylabel("Middle fixation duration (ms)")
    axs[2].grid(True)

    # --------------------------------------------------
    # (4) First fixation duration vs |ΔV|
    # --------------------------------------------------
    first_fix = add_abs_diff_bins(first_fix)

    summary = (
        groupby_obs(first_fix, "diff_bin")
        .agg(
            mean_fix_dur=("fix_dur", "mean"),
            sem_fix_dur=("fix_dur", sem),
        )
        .reset_index()
    )

    axs[3].errorbar(
        summary["diff_bin"].astype(int),
        summary["mean_fix_dur"],
        yerr=summary["sem_fix_dur"],
        fmt="o-",
        capsize=5,
    )
    axs[3].set_xlabel("Best − Worst")
    axs[3].set_ylabel("First fixation duration (ms)")
    axs[3].grid(True)

    # --------------------------------------------------
    # (5) Net fixation duration vs signed ΔV
    # --------------------------------------------------
    df = add_signed_diff(df)
    df["signed_fix_dur"] = np.where(
        df["location"] == 1, df["fix_dur"], -df["fix_dur"]
    )

    trial_summary = (
        groupby_obs(df, ["sub_id", "trial"])
        .agg(
            signed_fix_dur=("signed_fix_dur", "sum"),
            signed_diff=("signed_diff", "first"),
        )
        .reset_index()
    )

    trial_summary["diff_bin"] = pd.cut(
        trial_summary["signed_diff"],
        bins=SIGNED_DIFF_BINS,
        labels=SIGNED_DIFF_LABELS,
    )

    net = (
        groupby_obs(trial_summary, "diff_bin")
        .agg(
            mean_net=("signed_fix_dur", "mean"),
            sem_net=("signed_fix_dur", sem),
        )
        .reset_index()
    )

    axs[4].errorbar(
        net["diff_bin"].astype(int),
        net["mean_net"],
        yerr=net["sem_net"],
        fmt="o-",
        capsize=5,
    )
    axs[4].axhline(0, linestyle="--", color="gray")
    axs[4].set_xlabel("Left − Right")
    axs[4].set_ylabel("Net fixation duration (ms)")
    axs[4].grid(True)

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"fixation_properties_{basename_from_path(path)}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ======================================================
# Basic psychometrics
# ======================================================

def save_basic_psychometrics(parent_dir: str, path: str):
    df = pd.read_csv(os.path.join(parent_dir, path))
    outdir = ensure_outdir()

    fig, axs = plt.subplots(1, 3, figsize=(20, 4))

    first_fix = df[df["fix_num"] == 1].copy()
    first_fix = add_signed_diff(first_fix)

    # --------------------------------------------------
    # (6) Choice psychometric
    # --------------------------------------------------
    first_fix["choice_left"] = first_fix["choice"] == 0
    first_fix["diff_bin"] = pd.cut(
        first_fix["signed_diff"], bins=np.arange(-4, 4.25, 0.25)
    )

    prob = groupby_obs(first_fix, "diff_bin")["choice_left"].mean()
    centers = groupby_obs(first_fix, "diff_bin")["signed_diff"].mean()

    axs[0].plot(centers, prob, marker="o")
    axs[0].axhline(0.5, linestyle="--", color="gray")
    axs[0].set_xlabel("Left − Right")
    axs[0].set_ylabel("P(Choose left)")
    axs[0].grid(True)

    # --------------------------------------------------
    # (7) RT vs |ΔV|
    # --------------------------------------------------
    first_fix = add_abs_diff_bins(first_fix)

    rt = (
        groupby_obs(first_fix, "diff_bin")
        .agg(mean_rt=("RT", "mean"), sem_rt=("RT", sem))
        .reset_index()
    )
    rt[["mean_rt", "sem_rt"]] *= 1000

    axs[1].errorbar(
        rt["diff_bin"].astype(int),
        rt["mean_rt"],
        yerr=rt["sem_rt"],
        fmt="o-",
        capsize=5,
    )
    axs[1].set_xlabel("Best − Worst")
    axs[1].set_ylabel("Response time (ms)")
    axs[1].grid(True)

    # --------------------------------------------------
    # (8) Number of fixations vs |ΔV|
    # --------------------------------------------------
    nfix = (
        groupby_obs(first_fix, "diff_bin")
        .agg(mean_nfix=("fix_num_rev", "mean"), sem_nfix=("fix_num_rev", sem))
        .reset_index()
    )

    axs[2].errorbar(
        nfix["diff_bin"].astype(int),
        nfix["mean_nfix"],
        yerr=nfix["sem_nfix"],
        fmt="o-",
        capsize=5,
        color="tab:red",
    )
    axs[2].set_xlabel("Best − Worst")
    axs[2].set_ylabel("Number of fixations")
    axs[2].grid(True)

    fig.tight_layout()
    fig.savefig(
        os.path.join(outdir, f"basic_psychometrics_{basename_from_path(path)}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ======================================================
# CLI
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", type=str)
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    save_fixation_properties(args.parent_dir, args.path)
    save_basic_psychometrics(args.parent_dir, args.path)
