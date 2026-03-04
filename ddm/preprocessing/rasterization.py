import os, ast
import pandas as pd
import numpy as np
from math import floor, ceil

def expand_addm_fixations(sacc_data, flag_data, rt_data, dt):
    """
    Parameters
    ----------
    sacc_data : list of 1D np.ndarray
        saccade times per trial (seconds)
    flag_data : 1D np.ndarray
        initial fixation per trial (0/1 or similar)
    rt_data : 1D np.ndarray
        reaction time per trial (seconds)
    dt : float
        timestep size (seconds)

    Encoding
    --------
      0 -> transition
      1 -> fixation with fix_start = 0
      2 -> fixation with fix_start = 1

    Returns
    -------
    array
        Each element is a tuple of fixation locations for one trial
    """
    all_trials = []

    for saccs, start_fix, rt in zip(sacc_data, flag_data, rt_data):

        fix_len = int(floor(rt / dt)) + 1

        # start in transition
        fix = np.zeros(fix_len, dtype=int)

        if len(saccs) > 0:
            switch_idxs = [int(ceil(s / dt)) for s in saccs]

            # fixation identity determined by start_fix
            current_fix = 1 if start_fix == 0 else 2
            state = "fixation"  # first switch enters fixation

            for idx in switch_idxs[1:]:
                if idx <= 0 or idx >= fix_len:
                    continue

                if state == "fixation":
                    fix[idx:] = current_fix
                    state = "transition"
                else:
                    fix[idx:] = 0
                    current_fix = 1 if current_fix == 2 else 2
                    state = "fixation"

        all_trials.append(tuple(fix.tolist()))

    return all_trials

def rasterize_data(
    df: pd.DataFrame,
    subject_col: str,
    trial_col: str,
    seq_col: str = "fixation",
    fill_codes: set = {0, 4},
    start_col: str = "fix_start",
    end_col: str = "fix_end",
    loc_col: str = "fix_location",
    fixnum_col: str | None = None,
    keep_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Expand per-(subject, trial) fixation sequences into fixation-level rows.
    Zero-valued segments are treated as transitions and excluded.
    """

    df = df.copy()

    if keep_cols is None:
        keep_cols = [
            c for c in df.columns
            if c not in {subject_col, trial_col, seq_col}
        ]

    rows = []

    for _, row in df.iterrows():
        seq = np.asarray(row[seq_col])

        changes = np.diff(seq, prepend=seq[0])
        starts = np.where(changes != 0)[0]

        fix_num = 0

        for i, start_idx in enumerate(starts):
            loc = seq[start_idx]

            end_idx = (
                starts[i + 1]
                if i + 1 < len(starts)
                else len(seq)
            )

            # Skip transitions
            if loc in fill_codes:
                continue

            data = {
                subject_col: row[subject_col],
                trial_col: row[trial_col],
                start_col: start_idx,
                end_col: end_idx,
                loc_col: loc,
            }

            if fixnum_col is not None:
                data[fixnum_col] = fix_num
                fix_num += 1

            for col in keep_cols:
                data[col] = row[col]

            rows.append(data)

    return pd.DataFrame(rows)

def reformat_fixations(parent_dir, path):
    df = pd.read_csv(os.path.join(parent_dir, path))
    df['fixation'] = df['fixation'].apply(ast.literal_eval)
    df['trial'] = np.arange(1, len(df) + 1)

    new_rows = []

    for idx, row in df.iterrows():
        fixation_sequence = row['fixation']
        if not fixation_sequence:
            continue

        fix_num = 1
        i = 0
        while i < len(fixation_sequence):
            val = fixation_sequence[i]

            if val in (1, 2):
                start_idx = i
                current_loc = val
                while i + 1 < len(fixation_sequence) and fixation_sequence[i + 1] == current_loc:
                    i += 1
                end_idx = i + 1  # exclusive
                fix_dur = end_idx - start_idx

                new_rows.append({
                    'trial': row['trial'],
                    'choice': row['choice'],
                    'RT': row['RT'],
                    'avgWTP_left': row['avgWTP_left'],
                    'avgWTP_right': row['avgWTP_right'],
                    'fix_num': fix_num,
                    'location': current_loc,
                    'fix_start': start_idx,
                    'fix_end': end_idx,
                    'fix_dur': fix_dur
                })
                fix_num += 1
            i += 1

    result_df = pd.DataFrame(new_rows)

    result_df['fix_num_rev'] = result_df.groupby('trial')['fix_num'].transform(
        lambda x: x.max() - x + 1
    )

    result_df.to_csv(os.path.join('formatted_data', path), index=False)