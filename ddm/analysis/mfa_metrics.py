import numpy as np
import pandas as pd
from scipy.stats import sem
from .mfa_core import MFASummary

def mfa_first_fix_to_best(df: pd.DataFrame) -> MFASummary:
    first = df[df["fix_num"] == 1].copy()

    first["abs_diff"] = np.abs(first["avgWTP_left"] - first["avgWTP_right"])
    first["larger_side"] = np.where(
        first["avgWTP_left"] > first["avgWTP_right"], 1, 2
    )
    first["toward_larger"] = first["fix_location"] == first["larger_side"]
    first["diff_bin"] = pd.cut(
        first["abs_diff"], bins=[0.5, 1.5, 2.5, 3.5, 4.5], labels=[1, 2, 3, 4]
    )

    s = (
        first
        .groupby("diff_bin", observed=True)["toward_larger"]
        .agg(["mean", sem])
        .reset_index()
    )

    return MFASummary(
        name="p_first_fix_to_best",
        x=s["diff_bin"].astype(int).to_numpy(),
        y=s["mean"].to_numpy(),
        yerr=s["sem"].to_numpy(),
    )

def mfa_fix_dur_by_position(df: pd.DataFrame) -> MFASummary:
    first = df[df["fix_num"] == 1]["fix_dur"]
    middle = df[(df["fix_num"] != 1) & (df["fix_num_rev"] != 1)]["fix_dur"]
    last = df[df["fix_num_rev"] == 1]["fix_dur"]

    means = np.array([first.mean(), middle.mean(), last.mean()])
    errors = np.array([sem(first), sem(middle), sem(last)])

    return MFASummary(
        name="fix_dur_by_position",
        x=np.array([0, 1, 2]),   # First, Middle, Last
        y=means,
        yerr=errors,
    )

def mfa_middle_fix_dur_vs_absdiff(df: pd.DataFrame) -> MFASummary:
    mid = df[(df["fix_num"] != 1) & (df["fix_num_rev"] != 1)].copy()

    mid["abs_diff"] = np.abs(mid["avgWTP_left"] - mid["avgWTP_right"])
    mid["diff_bin"] = pd.cut(
        mid["abs_diff"],
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        labels=[0, 1, 2, 3, 4],
    )

    s = (
        mid
        .groupby("diff_bin", observed=True)["fix_dur"]
        .agg(["mean", sem])
        .reset_index()
    )

    return MFASummary(
        name="middle_fix_dur_vs_absdiff",
        x=s["diff_bin"].astype(int).to_numpy(),
        y=s["mean"].to_numpy(),
        yerr=s["sem"].to_numpy(),
    )

def mfa_first_fix_dur_vs_absdiff(df: pd.DataFrame) -> MFASummary:
    first = df[df["fix_num"] == 1].copy()

    first["abs_diff"] = np.abs(first["avgWTP_left"] - first["avgWTP_right"])
    first["diff_bin"] = pd.cut(
        first["abs_diff"],
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        labels=[0, 1, 2, 3, 4],
    )

    s = (
        first
        .groupby("diff_bin", observed=True)["fix_dur"]
        .agg(["mean", sem])
        .reset_index()
    )

    return MFASummary(
        name="first_fix_dur_vs_absdiff",
        x=s["diff_bin"].astype(int).to_numpy(),
        y=s["mean"].to_numpy(),
        yerr=s["sem"].to_numpy(),
    )

def mfa_net_fix_dur_vs_signeddiff(df: pd.DataFrame) -> MFASummary:
    df = df.copy()
    df["signed_diff"] = df["avgWTP_left"] - df["avgWTP_right"]
    df["signed_fix_dur"] = np.where(
        df["fix_location"] == 1, df["fix_dur"], -df["fix_dur"]
    )

    trial = (
        df
        .groupby(["sub_id", "trial"], observed=True)
        .agg(
            signed_fix_dur=("signed_fix_dur", "sum"),
            signed_diff=("signed_diff", "first"),
        )
        .reset_index()
    )

    trial["diff_bin"] = pd.cut(
        trial["signed_diff"],
        bins=np.arange(-4.5, 5, 1),
        labels=np.arange(-4, 5),
    )

    s = (
        trial
        .groupby("diff_bin", observed=True)["signed_fix_dur"]
        .agg(["mean", sem])
        .reset_index()
    )

    return MFASummary(
        name="net_fix_dur_vs_signeddiff",
        x=s["diff_bin"].astype(int).to_numpy(),
        y=s["mean"].to_numpy(),
        yerr=s["sem"].to_numpy(),
    )

def mfa_choice_psychometric(df: pd.DataFrame) -> MFASummary:
    first = df[df["fix_num"] == 1].copy()

    first["signed_diff"] = first["avgWTP_left"] - first["avgWTP_right"]
    first["choice_left"] = first["choice"] == 0
    first["diff_bin"] = pd.cut(
        first["signed_diff"], bins=np.arange(-4, 4.25, 0.25)
    )

    prob = (
        first
        .groupby("diff_bin", observed=True)["choice_left"]
        .mean()
    )
    centers = (
        first
        .groupby("diff_bin", observed=True)["signed_diff"]
        .mean()
    )

    # Binomial SE (theoretically correct)
    n = first.groupby("diff_bin", observed=True)["choice_left"].count()
    se = np.sqrt(prob * (1 - prob) / n)

    return MFASummary(
        name="choice_psychometric",
        x=centers.to_numpy(),
        y=prob.to_numpy(),
        yerr=se.to_numpy(),
    )

def mfa_rt_vs_absdiff(df: pd.DataFrame) -> MFASummary:
    first = df[df["fix_num"] == 1].copy()

    first["abs_diff"] = np.abs(first["avgWTP_left"] - first["avgWTP_right"])
    first["diff_bin"] = pd.cut(
        first["abs_diff"],
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        labels=[0, 1, 2, 3, 4],
    )

    s = (
        first
        .groupby("diff_bin", observed=True)["RT"]
        .agg(["mean", sem])
        .reset_index()
    )

    return MFASummary(
        name="rt_vs_absdiff",
        x=s["diff_bin"].astype(int).to_numpy(),
        y=(1000 * s["mean"]).to_numpy(),
        yerr=(1000 * s["sem"]).to_numpy(),
    )

def mfa_nfix_vs_absdiff(df: pd.DataFrame) -> MFASummary:
    first = df[df["fix_num"] == 1].copy()

    first["abs_diff"] = np.abs(first["avgWTP_left"] - first["avgWTP_right"])
    first["diff_bin"] = pd.cut(
        first["abs_diff"],
        bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        labels=[0, 1, 2, 3, 4],
    )

    s = (
        first
        .groupby("diff_bin", observed=True)["fix_num_rev"]
        .agg(["mean", sem])
        .reset_index()
    )

    return MFASummary(
        name="nfix_vs_absdiff",
        x=s["diff_bin"].astype(int).to_numpy(),
        y=s["mean"].to_numpy(),
        yerr=s["sem"].to_numpy(),
    )