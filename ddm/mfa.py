from scipy.stats import sem
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

@dataclass
class MFASummary:
    name: str
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None

def compute_mfa(df: pd.DataFrame) -> dict[str, MFASummary]:
    return {
        "p_first_fix_to_best": mfa_first_fix_to_best(df),
        "fix_dur_by_position": mfa_fix_dur_by_position(df),
        "middle_fix_dur_vs_absdiff": mfa_middle_fix_dur_vs_absdiff(df),
        "first_fix_dur_vs_absdiff": mfa_first_fix_dur_vs_absdiff(df),
        "net_fix_dur_vs_signeddiff": mfa_net_fix_dur_vs_signeddiff(df),
        "choice_psychometric": mfa_choice_psychometric(df),
        "rt_vs_absdiff": mfa_rt_vs_absdiff(df),
        "nfix_vs_absdiff": mfa_nfix_vs_absdiff(df),
    }

def model_free_loss(
    emp_mfa: dict[str, MFASummary],
    sim_mfa: dict[str, MFASummary],
    weights: dict[str, float] | None = None,
    *,
    # coverage / robustness parameters (global defaults)
    lambda_missing: float = 50000.0,
    gamma_precision: float = 0.0,
    min_overlap: int = 1,
    expected_x: dict[str, np.ndarray] | None = None,
    # optional per-metric overrides
    lambda_by_metric: dict[str, float] | None = None,
    gamma_by_metric: dict[str, float] | None = None,
) -> float:
    """
    Compute total model-free loss using coverage-aware summary losses.

    Parameters
    ----------
    emp_mfa, sim_mfa : dict[str, MFASummary]
        Empirical and simulated model-free analyses.
    weights : dict[str, float], optional
        Relative importance of each summary.
    lambda_missing : float
        Global penalty strength for missing bins. 
        Scaled with relevant losses.
    gamma_precision : float
        Global precision-weighted missing-bin penalty. 
        0.1 for gentle, 0.5 for strong.
    min_overlap : int
        Minimum number of overlapping bins before overlap loss is computed.
    expected_x : dict[str, np.ndarray], optional
        Explicit expected bin support per metric.
        If None, empirical x is used. 
        For large sample sizes for subjects, this will not be an issue.
    lambda_by_metric, gamma_by_metric : dict[str, float], optional
        Per-metric overrides for coverage penalties.
    """

    total = 0.0

    for name, emp in emp_mfa.items():
        sim = sim_mfa[name]

        # weight for this metric
        w = 1.0 if weights is None else weights.get(name, 1.0)

        # coverage penalties (allow per-metric override)
        lam = (
            lambda_missing
            if lambda_by_metric is None
            else lambda_by_metric.get(name, lambda_missing)
        )
        gam = (
            gamma_precision
            if gamma_by_metric is None
            else gamma_by_metric.get(name, gamma_precision)
        )

        # expected support (optional)
        exp_x = None if expected_x is None else expected_x.get(name)

        loss_k = mfa_summary_loss_with_coverage(
            emp=emp,
            sim=sim,
            lambda_missing=lam,
            gamma_precision=gam,
            min_overlap=min_overlap,
            expected_x=exp_x,
        )

        total += w * loss_k

    return float(total)

def mfa_summary_loss_with_coverage(
    emp: MFASummary,
    sim: MFASummary,
    eps: float = 1e-6,
    lambda_missing: float = 1.0,     # strength of coverage penalty
    gamma_precision: float = 0.0,    # 0 to disable precision-weighted missing penalty
    expected_x: np.ndarray | None = None,  # override expected bins if you want
    min_overlap: int = 1,            # allow 1-bin overlap (useful early in search)
) -> float:
    assert emp.name == sim.name

    # What bins do we "expect"?
    # Default: use empirical x as the expected support.
    B = np.asarray(expected_x if expected_x is not None else emp.x)

    # Overlap bins
    C = np.intersect1d(B, sim.x)

    # Missing bins from sim (relative to expected)
    M = np.setdiff1d(B, sim.x)

    # If no overlap, you still want a finite but bad loss
    if len(C) < min_overlap:
        overlap_loss = 0.0
    else:
        emp_mask = np.isin(emp.x, C)
        sim_mask = np.isin(sim.x, C)

        emp_y = emp.y[emp_mask]
        sim_y = sim.y[sim_mask]

        if emp.yerr is None:
            denom = 1.0
        else:
            denom = emp.yerr[emp_mask] ** 2 + eps

        overlap_loss = float(np.mean((sim_y - emp_y) ** 2 / denom))

    # Coverage penalty: fraction of expected bins missing
    missing_frac = len(M) / max(len(B), 1)
    coverage_penalty = lambda_missing * missing_frac

    # Optional: penalize missing bins more if empirical uncertainty is small
    precision_penalty = 0.0
    if gamma_precision > 0.0 and emp.yerr is not None and len(M) > 0:
        # map M -> indices in emp.x
        miss_mask_emp = np.isin(emp.x, M)
        # precision weight = 1/var
        precision_penalty = gamma_precision * float(
            np.mean(1.0 / (emp.yerr[miss_mask_emp] ** 2 + eps))
        )

    return overlap_loss + coverage_penalty + precision_penalty

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


def plot_fixation_properties(df: pd.DataFrame):
    """Visualize fixation properties directly from a DataFrame."""
    fig1, axs = plt.subplots(1, 5, figsize=(25, 4))

    # First fixation toward higher value
    first_fix_df = df[df['fix_num'] == 1].copy()
    first_fix_df['abs_diff'] = np.abs(first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right'])
    first_fix_df['larger_side'] = np.where(first_fix_df['avgWTP_left'] > first_fix_df['avgWTP_right'], 1, 2)
    first_fix_df['toward_larger'] = first_fix_df['fix_location'] == first_fix_df['larger_side']
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['abs_diff'], bins=[0.5, 1.5, 2.5, 3.5, 4.5], labels=[1, 2, 3, 4])
    summary = first_fix_df.groupby('diff_bin', observed=True)['toward_larger'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_toward_larger', 'sem_toward_larger']

    axs[0].axhline(0.5, color='grey', linestyle='--')
    axs[0].errorbar(summary['diff_bin'].astype(int), summary['mean_toward_larger'],
                    yerr=summary['sem_toward_larger'], fmt='o-', capsize=5)
    axs[0].set_xlabel('Best - Worst')
    axs[0].set_ylabel('P(First Fixation to Best)')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0.75, 4.25)
    axs[0].grid(True)

    # Fixation duration by type
    middle_fix_df = df[(df['fix_num'] != 1) & (df['fix_num_rev'] != 1)].copy()
    last_fix_df = df[df['fix_num_rev'] == 1].copy()
    means = [first_fix_df['fix_dur'].mean(), middle_fix_df['fix_dur'].mean(), last_fix_df['fix_dur'].mean()]
    errors = [sem(first_fix_df['fix_dur']), sem(middle_fix_df['fix_dur']), sem(last_fix_df['fix_dur'])]
    axs[1].bar(['First', 'Middle', 'Last'], means, yerr=errors, capsize=5)
    axs[1].set_ylabel('Fixation Duration (ms)')
    axs[1].grid(axis='y')

    # Middle fixation duration by difficulty
    middle_fix_df['abs_diff'] = np.abs(middle_fix_df['avgWTP_left'] - middle_fix_df['avgWTP_right'])
    middle_fix_df['diff_bin'] = pd.cut(middle_fix_df['abs_diff'],
                                       bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                                       labels=[0, 1, 2, 3, 4])
    summary = middle_fix_df.groupby('diff_bin', observed=True)['fix_dur'].agg(['mean', sem]).reset_index()
    summary.columns = ['difficulty', 'mean_fix_dur', 'sem_fix_dur']
    axs[2].errorbar(summary['difficulty'], summary['mean_fix_dur'],
                    yerr=summary['sem_fix_dur'], fmt='o-', capsize=5)
    axs[2].set_xlabel('Best - Worst')
    axs[2].set_ylabel('Middle Fixation Duration (ms)')
    axs[2].set_xticks([0, 1, 2, 3, 4])
    axs[2].set_xlim(-0.25, 4.25)
    axs[2].grid(True)

    # First fixation duration by difficulty
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['abs_diff'],
                                      bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                                      labels=[0, 1, 2, 3, 4])
    summary = first_fix_df.groupby('diff_bin', observed=True)['fix_dur'].agg(['mean', sem]).reset_index()
    summary.columns = ['difficulty', 'mean_fix_dur', 'sem_fix_dur']
    axs[3].errorbar(summary['difficulty'], summary['mean_fix_dur'],
                    yerr=summary['sem_fix_dur'], fmt='o-', capsize=5)
    axs[3].set_xlabel('Best - Worst')
    axs[3].set_ylabel('First Fixation Duration (ms)')
    axs[3].set_xticks([0, 1, 2, 3, 4])
    axs[3].grid(True)

    # Net fixation duration by signed difficulty
    df['signed_diff'] = df['avgWTP_left'] - df['avgWTP_right']
    df['signed_fix_dur'] = np.where(df['fix_location'] == 1, df['fix_dur'], -df['fix_dur'])
    trial_durations = df.groupby(['sub_id', 'trial'], observed=True).agg({'signed_fix_dur': 'sum', 'signed_diff': 'first'}).reset_index()
    trial_durations['diff_bin'] = pd.cut(trial_durations['signed_diff'],
                                         bins=np.arange(-4.5, 5, 1),
                                         labels=np.arange(-4, 5))
    summary = trial_durations.groupby('diff_bin', observed=True)['signed_fix_dur'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_net_dur', 'sem_net_dur']
    summary['bin_x'] = summary['diff_bin'].astype(int)

    axs[4].errorbar(summary['bin_x'], summary['mean_net_dur'],
                    yerr=summary['sem_net_dur'], fmt='o-', capsize=5)
    axs[4].axhline(0, color='gray', linestyle='--')
    axs[4].set_xlabel('Left - Right')
    axs[4].set_ylabel('Net Fixation Duration (ms)')
    axs[4].set_xticks(np.arange(-4, 5))
    axs[4].grid(True)

    fig1.tight_layout()
    plt.show()

def plot_basic_psychometrics(df: pd.DataFrame):
    """Visualize basic psychometrics directly from a DataFrame."""
    fig2, axs = plt.subplots(1, 3, figsize=(20, 4))

    # Choice by signed difficulty
    first_fix_df = df[df['fix_num'] == 1].copy()
    first_fix_df['signed_diff'] = first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right']
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['signed_diff'], bins=np.arange(-4, 4.25, 0.25))
    prob_left_by_bin = first_fix_df.groupby('diff_bin', observed=True)['choice'].apply(lambda x: (x == 0).mean())
    bin_centers = first_fix_df.groupby('diff_bin', observed=True)['signed_diff'].mean()

    axs[0].plot(bin_centers, prob_left_by_bin, marker='o', linestyle='-')
    axs[0].axhline(0.5, color='gray', linestyle='--')
    axs[0].set_xlabel("Left - Right")
    axs[0].set_ylabel("P(Choose Left)")
    axs[0].grid(True)

    # Response time by difficulty
    first_fix_df['abs_diff'] = np.abs(first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right'])
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['abs_diff'],
                                      bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                                      labels=[0, 1, 2, 3, 4])
    summary = first_fix_df.groupby('diff_bin', observed=True)['RT'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_RT', 'sem_RT']
    # summary['mean_RT'] *= 1000
    # summary['sem_RT'] *= 1000

    axs[1].errorbar(summary['diff_bin'].astype(int), summary['mean_RT'],
                    yerr=summary['sem_RT'], fmt='o-', capsize=5)
    axs[1].set_xlabel("Best - Worst")
    axs[1].set_ylabel("Response Time (ms)")
    axs[1].set_xticks([0, 1, 2, 3, 4])
    axs[1].grid(True)

    # Number of fixations by difficulty
    summary = first_fix_df.groupby('diff_bin', observed=True)['fix_num_rev'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_n_fixations', 'sem_n_fixations']
    summary['bin_x'] = summary['diff_bin'].astype(float)

    axs[2].errorbar(summary['bin_x'], summary['mean_n_fixations'],
                    yerr=summary['sem_n_fixations'], fmt='o-', capsize=5, color='tab:red')
    axs[2].set_xlabel("Best - Worst")
    axs[2].set_ylabel("Number of Fixations")
    axs[2].grid(True)

    fig2.tight_layout()
    plt.show()

def save_fixation_properties(parent_dir, path):
    df = pd.read_csv(os.path.join(parent_dir, path))

    # --------------------------- #
    # Figure 1: First 5 plots     #
    # --------------------------- #
    fig1, axs = plt.subplots(1, 5, figsize=(25, 4))

    ## Plot 1: First fixation toward higher value
    first_fix_df = df[df['fix_num'] == 1].copy()
    first_fix_df['abs_diff'] = np.abs(first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right'])
    first_fix_df['larger_side'] = np.where(first_fix_df['avgWTP_left'] > first_fix_df['avgWTP_right'], 1, 2)
    first_fix_df['toward_larger'] = first_fix_df['location'] == first_fix_df['larger_side']
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['abs_diff'], bins=[0.5, 1.5, 2.5, 3.5, 4.5], labels=[1, 2, 3, 4])
    summary = first_fix_df.groupby('diff_bin', observed=True)['toward_larger'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_toward_larger', 'sem_toward_larger']
    axs[0].axhline(0.5, color='grey', linestyle='--')
    axs[0].errorbar(summary['diff_bin'].astype(int), summary['mean_toward_larger'], yerr=summary['sem_toward_larger'], fmt='o-', capsize=5)
    axs[0].set_xlabel('Best - Worst')
    axs[0].set_ylabel('P(First Fixation to Best)')
    axs[0].set_ylim(0, 1)
    axs[0].set_xlim(0.75, 4.25)
    axs[0].set_xticks([1, 2, 3, 4])
    axs[0].grid(True)

    ## Plot 2: Fixation duration by type
    middle_fix_df = df[(df['fix_num'] != 1) & (df['fix_num_rev'] != 1)].copy()
    last_fix_df = df[df['fix_num_rev'] == 1].copy()
    means = [first_fix_df['fix_dur'].mean(), middle_fix_df['fix_dur'].mean(), last_fix_df['fix_dur'].mean()]
    errors = [sem(first_fix_df['fix_dur']), sem(middle_fix_df['fix_dur']), sem(last_fix_df['fix_dur'])]
    axs[1].bar(['First', 'Middle', 'Last'], means, yerr=errors, capsize=5)
    axs[1].set_ylabel('Fixation Duration (ms)')
    axs[1].grid(axis='y')

    ## Plot 3: Middle fixation duration by difficulty
    middle_fix_df['abs_diff'] = np.abs(middle_fix_df['avgWTP_left'] - middle_fix_df['avgWTP_right'])
    middle_fix_df['diff_bin'] = pd.cut(middle_fix_df['abs_diff'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], labels=[0, 1, 2, 3, 4])
    summary = middle_fix_df.groupby('diff_bin', observed=True)['fix_dur'].agg(['mean', sem]).reset_index()
    summary.columns = ['difficulty', 'mean_fix_dur', 'sem_fix_dur']
    axs[2].errorbar(summary['difficulty'], summary['mean_fix_dur'], yerr=summary['sem_fix_dur'], fmt='o-', capsize=5)
    axs[2].set_xlabel('Best - Worst')
    axs[2].set_ylabel('Middle Fixation Duration (ms)')
    axs[2].set_xticks([0, 1, 2, 3, 4])
    axs[2].set_xlim(-0.25, 4.25)
    axs[2].grid(True)

    ## Plot 4: First fixation duration by difficulty
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['abs_diff'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], labels=[0, 1, 2, 3, 4])
    summary = first_fix_df.groupby('diff_bin', observed=True)['fix_dur'].agg(['mean', sem]).reset_index()
    summary.columns = ['difficulty', 'mean_fix_dur', 'sem_fix_dur']
    axs[3].errorbar(summary['difficulty'], summary['mean_fix_dur'], yerr=summary['sem_fix_dur'], fmt='o-', capsize=5)
    axs[3].set_xlabel('Best - Worse')
    axs[3].set_ylabel('First Fixation Duration (ms)')
    axs[3].set_xticks([0, 1, 2, 3, 4])
    axs[3].set_xlim(-0.25, 4.25)
    axs[3].grid(True)

    ## Plot 5: Net fixation duration by signed difficulty
    df['signed_diff'] = df['avgWTP_left'] - df['avgWTP_right']
    df['signed_fix_dur'] = df.apply(lambda row: row['fix_dur'] if row['location'] == 1 else -row['fix_dur'], axis=1)
    trial_durations = df.groupby(['sub_id', 'trial'], observed=True).agg({'signed_fix_dur': 'sum', 'signed_diff': 'first'}).reset_index()
    trial_durations['diff_bin'] = pd.cut(trial_durations['signed_diff'], bins=np.arange(-4.5, 5, 1), labels=np.arange(-4, 5))
    summary = trial_durations.groupby('diff_bin', observed=True)['signed_fix_dur'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_net_dur', 'sem_net_dur']
    summary['bin_x'] = summary['diff_bin'].astype(int)
    axs[4].errorbar(summary['bin_x'], summary['mean_net_dur'], yerr=summary['sem_net_dur'], fmt='o-', capsize=5, label='Simulated Data')
    axs[4].axhline(0, color='gray', linestyle='--')
    axs[4].set_xlabel('Left - Right')
    axs[4].set_ylabel('Net Fixation Duration (ms)')
    axs[4].set_xticks(np.arange(-4, 5))
    axs[4].set_xlim(-4.5, 4.5)
    axs[4].grid(True)

    fig1.tight_layout()
    if not os.path.exists('mfa_figures'):
        os.makedirs('mfa_figures')
    fig1.savefig(os.path.join('mfa_figures', f'fixation_properties_{path[-24:-4]}.png'), dpi=300, bbox_inches='tight')

def save_basic_psychometrics(parent_dir, path):
    df = pd.read_csv(os.path.join(parent_dir, path))

    # --------------------------- #
    # Figure 2: Last 3 plots      #
    # --------------------------- #
    fig2, axs = plt.subplots(1, 3, figsize=(20, 4))

    ## Plot 6: Choice by signed difficulty
    first_fix_df = df[df['fix_num'] == 1].copy()
    first_fix_df['signed_diff'] = first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right']
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['signed_diff'], bins=np.arange(-4, 4.25, 0.25))
    prob_left_by_bin = first_fix_df.groupby('diff_bin', observed=True)['choice'].apply(lambda x: (x == 0).mean())
    bin_centers = first_fix_df.groupby('diff_bin', observed=True)['signed_diff'].mean()
    axs[0].plot(bin_centers, prob_left_by_bin, marker='o', linestyle='-')
    axs[0].axhline(0.5, color='gray', linestyle='--')
    axs[0].set_xlabel("Left - Right")
    axs[0].set_ylabel("P(Choose Left)")
    axs[0].grid(True)

    ## Plot 7: Response time by difficulty
    first_fix_df['abs_diff'] = np.abs(first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right'])
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['abs_diff'], bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], labels=[0, 1, 2, 3, 4])
    summary = first_fix_df.groupby('diff_bin', observed=True)['RT'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_RT', 'sem_RT']
    summary['mean_RT'] *= 1000
    summary['sem_RT'] *= 1000
    axs[1].errorbar(summary['diff_bin'].astype(int), summary['mean_RT'], yerr=summary['sem_RT'], fmt='o-', capsize=5)
    axs[1].set_xlabel("Best - Worst")
    axs[1].set_ylabel("Response Time (ms)")
    axs[1].set_xticks([0, 1, 2, 3, 4])
    axs[1].set_xlim(-0.25, 4.25)
    axs[1].grid(True)

    ## Plot 8: Number of fixations by difficulty
    summary = first_fix_df.groupby('diff_bin', observed=True)['fix_num_rev'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_n_fixations', 'sem_n_fixations']
    summary['bin_x'] = summary['diff_bin'].astype(float)
    axs[2].errorbar(summary['bin_x'], summary['mean_n_fixations'], yerr=summary['sem_n_fixations'], fmt='o-', capsize=5, color='tab:red')
    axs[2].set_xlabel("Best - Worst")
    axs[2].set_ylabel("Number of Fixations")
    axs[2].set_xticks(summary['bin_x'])
    axs[2].grid(True)

    fig2.tight_layout()
    if not os.path.exists('mfa_figures'):
        os.makedirs('mfa_figures')
    fig2.savefig(os.path.join('mfa_figures', f'basic_psychometrics_{path[-24:-4]}.png'), dpi=300, bbox_inches='tight')

# Alternative figures for poster

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
import os


# --------------------------- #
# Helpers
# --------------------------- #

def _ensure_outdir(dirname: str = "mfa_figures"):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def _basename_from_path(path: str) -> str:
    # keep your existing slice (adjust if your filenames change)
    # example: ...YYYY-mm-dd_HH-MM-SS.csv -> take last 20 chars before ".csv"
    # You used [-24:-4]; keep that for compatibility
    return path[-24:-4] if len(path) >= 24 else os.path.splitext(os.path.basename(path))[0]

def _save_svg(fig: plt.Figure, fname: str):
    outdir = _ensure_outdir()
    fig.savefig(os.path.join(outdir, fname), format="svg", bbox_inches="tight")
    plt.close(fig)


# --------------------------- #
# (1) Choice patterns
#     - (1) P(choose left) vs signed ΔV
#     - (4) P(first fixation to higher value) vs |ΔV|
#     Stacked vertically (2 x 1)
# --------------------------- #
def save_choice_patterns(parent_dir: str, path: str):
    df = pd.read_csv(os.path.join(parent_dir, path))

    # Compute per-trial signed diff for choice panel
    first_fix_df = df[df['fix_num'] == 1].copy()
    first_fix_df['signed_diff'] = first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right']

    # Panel A: P(Choose Left) by signed ΔV (bin centers)
    # Your original used 0.25-wide bins; keep that
    first_fix_df['choice_is_left'] = (first_fix_df['choice'] == 0)
    bins_choice = np.arange(-4, 4.25, 0.25)
    first_fix_df['diff_bin_choice'] = pd.cut(first_fix_df['signed_diff'], bins=bins_choice)
    prob_left_by_bin = first_fix_df.groupby('diff_bin_choice', observed=True)['choice_is_left'].mean()
    bin_centers = first_fix_df.groupby('diff_bin_choice', observed=True)['signed_diff'].mean()

    # Panel B: P(First Fixation to Higher Value) by |ΔV|
    first_fix_df['abs_diff'] = np.abs(first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right'])
    first_fix_df['larger_side'] = np.where(first_fix_df['avgWTP_left'] > first_fix_df['avgWTP_right'], 1, 2)
    first_fix_df['toward_larger'] = (first_fix_df['location'] == first_fix_df['larger_side'])
    # Match your previous binning (labels 0..4 over edges -0.5..4.5)
    bins_abs = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    labels_abs = [0, 1, 2, 3, 4]
    first_fix_df['diff_bin_abs'] = pd.cut(first_fix_df['abs_diff'], bins=bins_abs, labels=labels_abs)
    p_ff_toward = first_fix_df.groupby('diff_bin_abs', observed=True)['toward_larger'].agg(['mean', sem]).reset_index()
    p_ff_toward.columns = ['diff_bin', 'mean_toward', 'sem_toward']
    p_ff_toward['x'] = p_ff_toward['diff_bin'].astype(int)

    # Figure
    fig, axs = plt.subplots(2, 1, figsize=(5.0, 7.0), constrained_layout=True)

    # Top: Choice psychometric
    axs[0].plot(bin_centers, prob_left_by_bin, marker='o', linestyle='-')
    axs[0].axhline(0.5, linestyle='--', color='gray')
    axs[0].set_xlabel("Signed value difference (Left − Right)")
    axs[0].set_ylabel("P(Choose Left)")
    axs[0].set_title("Choice patterns")
    axs[0].grid(True)

    # Bottom: First fixation toward higher value vs |ΔV|
    axs[1].errorbar(p_ff_toward['x'], p_ff_toward['mean_toward'],
                    yerr=p_ff_toward['sem_toward'], fmt='o-', capsize=5)
    axs[1].axhline(0.5, linestyle='--', color='gray')
    axs[1].set_xlabel("Best − Worst (|ΔV| bins)")
    axs[1].set_ylabel("P(First fixation to higher value)")
    axs[1].set_xticks([0, 1, 2, 3, 4])
    axs[1].set_xlim(-0.25, 4.25)
    axs[1].grid(True)

    _save_svg(fig, f"choice_patterns_{_basename_from_path(path)}.svg")


# --------------------------- #
# (2) Response dynamics
#     - (2) RT vs |ΔV| (error bars)
# --------------------------- #
def save_response_dynamics(parent_dir: str, path: str):
    df = pd.read_csv(os.path.join(parent_dir, path))

    first_fix_df = df[df['fix_num'] == 1].copy()
    first_fix_df['abs_diff'] = np.abs(first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right'])
    bins_abs = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    labels_abs = [0, 1, 2, 3, 4]
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['abs_diff'], bins=bins_abs, labels=labels_abs)

    summary = first_fix_df.groupby('diff_bin', observed=True)['RT'].agg(['mean', sem]).reset_index()
    summary.columns = ['diff_bin', 'mean_RT', 'sem_RT']
    # Convert to ms like your original
    summary['mean_RT'] *= 1000.0
    summary['sem_RT'] *= 1000.0
    summary['x'] = summary['diff_bin'].astype(int)

    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.5), constrained_layout=True)
    ax.errorbar(summary['x'], summary['mean_RT'], yerr=summary['sem_RT'], fmt='o-', capsize=5)
    ax.set_title("Response dynamics")
    ax.set_xlabel("Best − Worst (|ΔV| bins)")
    ax.set_ylabel("Response time (ms)")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xlim(-0.25, 4.25)
    ax.grid(True)

    _save_svg(fig, f"response_dynamics_{_basename_from_path(path)}.svg")


# --------------------------- #
# (3) Fixation structure
#     - (3) Number of fixations vs |ΔV|
#     - (5) Fixation duration (First / Middle / Last) bars
#     Side-by-side (1 x 2)
# --------------------------- #
def save_fixation_structure(parent_dir: str, path: str):
    df = pd.read_csv(os.path.join(parent_dir, path))

    # First-fix rows (to reuse your binning for |ΔV|)
    first_fix_df = df[df['fix_num'] == 1].copy()
    first_fix_df['abs_diff'] = np.abs(first_fix_df['avgWTP_left'] - first_fix_df['avgWTP_right'])
    bins_abs = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    labels_abs = [0, 1, 2, 3, 4]
    first_fix_df['diff_bin'] = pd.cut(first_fix_df['abs_diff'], bins=bins_abs, labels=labels_abs)

    # (3) Number of fixations vs |ΔV|
    # You used fix_num_rev as count-like measure in prev code
    nfix_summary = first_fix_df.groupby('diff_bin', observed=True)['fix_num_rev'].agg(['mean', sem]).reset_index()
    nfix_summary.columns = ['diff_bin', 'mean_nfix', 'sem_nfix']
    nfix_summary['x'] = nfix_summary['diff_bin'].astype(int)

    # (5) Fixation duration by First/Middle/Last
    middle_fix_df = df[(df['fix_num'] != 1) & (df['fix_num_rev'] != 1)].copy()
    last_fix_df = df[df['fix_num_rev'] == 1].copy()
    means = [
        df[df['fix_num'] == 1]['fix_dur'].mean(),
        middle_fix_df['fix_dur'].mean(),
        last_fix_df['fix_dur'].mean()
    ]
    errors = [
        sem(df[df['fix_num'] == 1]['fix_dur']) if len(df[df['fix_num'] == 1]['fix_dur']) > 1 else 0.0,
        sem(middle_fix_df['fix_dur']) if len(middle_fix_df['fix_dur']) > 1 else 0.0,
        sem(last_fix_df['fix_dur']) if len(last_fix_df['fix_dur']) > 1 else 0.0
    ]
    labels = ['First', 'Middle', 'Last']

    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.5), constrained_layout=True)

    # Left: Number of fixations vs |ΔV|
    axs[0].errorbar(nfix_summary['x'], nfix_summary['mean_nfix'],
                    yerr=nfix_summary['sem_nfix'], fmt='o-', capsize=5)
    axs[0].set_title("Fixation count vs |ΔV|")
    axs[0].set_xlabel("Best − Worst (|ΔV| bins)")
    axs[0].set_ylabel("Number of fixations")
    axs[0].set_xticks([0, 1, 2, 3, 4])
    axs[0].set_xlim(-0.25, 4.25)
    axs[0].grid(True)

    # Right: Fixation duration by position
    axs[1].bar(labels, means, yerr=errors, capsize=5)
    axs[1].set_title("Fixation duration by position")
    axs[1].set_ylabel("Fixation duration (ms)")
    axs[1].grid(axis='y')

    _save_svg(fig, f"fixation_structure_{_basename_from_path(path)}.svg")


# --------------------------- #
# (4) Value-modulated fixation
#     - (6) Middle fixation duration vs |ΔV|
#     - (7) First fixation duration vs |ΔV|
#     - (8) Net fixation duration (L−R) vs signed ΔV
#     Horizontal triptych (1 x 3)
# --------------------------- #
def save_value_modulated_fixation(parent_dir: str, path: str):
    df = pd.read_csv(os.path.join(parent_dir, path))

    # Common prep
    first_fix_df = df[df['fix_num'] == 1].copy()
    middle_fix_df = df[(df['fix_num'] != 1) & (df['fix_num_rev'] != 1)].copy()

    # |ΔV| bins 0..4 as before
    def add_abs_bins(frame):
        frame = frame.copy()
        frame['abs_diff'] = np.abs(frame['avgWTP_left'] - frame['avgWTP_right'])
        frame['diff_bin'] = pd.cut(frame['abs_diff'],
                                   bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                                   labels=[0, 1, 2, 3, 4])
        return frame

    first_fix_df = add_abs_bins(first_fix_df)
    middle_fix_df = add_abs_bins(middle_fix_df)

    # (6) Middle fixation duration vs |ΔV|
    mid_summary = middle_fix_df.groupby('diff_bin', observed=True)['fix_dur'].agg(['mean', sem]).reset_index()
    mid_summary.columns = ['diff_bin', 'mean_fix_dur', 'sem_fix_dur']
    mid_summary['x'] = mid_summary['diff_bin'].astype(int)

    # (7) First fixation duration vs |ΔV|
    first_summary = first_fix_df.groupby('diff_bin', observed=True)['fix_dur'].agg(['mean', sem]).reset_index()
    first_summary.columns = ['diff_bin', 'mean_fix_dur', 'sem_fix_dur']
    first_summary['x'] = first_summary['diff_bin'].cat.codes

    # (8) Net fixation duration (Left − Right) vs signed ΔV
    df = df.copy()
    df['signed_diff'] = df['avgWTP_left'] - df['avgWTP_right']
    # signed contribution: +fix_dur if fixation on left (1), −fix_dur if on right (2)
    df['signed_fix_dur'] = np.where(df['location'] == 1, df['fix_dur'], -df['fix_dur'])
    trial_durs = df.groupby('trial', observed=True).agg({'signed_fix_dur': 'sum', 'signed_diff': 'first'}).reset_index()
    # integer bins from −4..4
    trial_durs['diff_bin'] = pd.cut(trial_durs['signed_diff'],
                                bins=np.arange(-4.5, 5, 1),
                                labels=np.arange(-4, 5))
    net_summary = trial_durs.groupby('diff_bin', observed=True)['signed_fix_dur'].agg(['mean', sem]).reset_index()
    net_summary.columns = ['diff_bin', 'mean_net_dur', 'sem_net_dur']
    net_summary['x'] = net_summary['diff_bin'].astype(int)

    # Figure
    fig, axs = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)

    # Left: Middle fixation duration vs |ΔV|
    axs[0].errorbar(mid_summary['x'], mid_summary['mean_fix_dur'],
                yerr=mid_summary['sem_fix_dur'], fmt='o-', capsize=5)
    axs[0].set_title("Middle fixation dur. vs |ΔV|")
    axs[0].set_xlabel("Best − Worst (|ΔV| bins)")
    axs[0].set_ylabel("Duration (ms)")
    axs[0].set_xticks([0, 1, 2, 3, 4])
    axs[0].set_xlim(-0.25, 4.25)
    axs[0].grid(True)

    # Middle: First fixation duration vs |ΔV|
    axs[1].errorbar(first_summary['x'], first_summary['mean_fix_dur'],
                yerr=first_summary['sem_fix_dur'], fmt='o-', capsize=5)
    axs[1].set_title("First fixation dur. vs |ΔV|")
    axs[1].set_xlabel("Best − Worst (|ΔV| bins)")
    axs[1].set_ylabel("Duration (ms)")
    axs[1].set_xticks([0, 1, 2, 3, 4])
    axs[1].set_xlim(-0.25, 4.25)
    axs[1].grid(True)

    # Right: Net fixation duration vs signed ΔV
    axs[2].errorbar(net_summary['x'], net_summary['mean_net_dur'],
                yerr=net_summary['sem_net_dur'], fmt='o-', capsize=5, label='Simulated')
    axs[2].axhline(0, linestyle='--', color='gray')
    axs[2].set_title("Net fixation dur. vs signed ΔV")
    axs[2].set_xlabel("Signed value difference (Left − Right)")
    axs[2].set_ylabel("Net duration (ms)")
    axs[2].set_xticks(np.arange(-4, 5))
    axs[2].set_xlim(-4.5, 4.5)
    axs[2].grid(True)

    _save_svg(fig, f"response_dynamics_{_basename_from_path(path)}.svg")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", type=str)
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    save_fixation_properties(args.parent_dir, args.path)
    save_basic_psychometrics(args.parent_dir, args.path)