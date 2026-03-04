import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem


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