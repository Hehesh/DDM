import os, ast
import pandas as pd
import numpy as np
from math import floor, ceil

def compress_fixations(fixation_trials, max_d, dt):
    """
    Inverse of expand_fixations (up to dt resolution),
    with zero-padded saccade arrays, using
    left-inclusive / right-exclusive convention.

    Parameters
    ----------
    fixation_trials : iterable of tuple[int]
        Output of expand_fixations
    dt : float
        timestep size (seconds)
    max_d : int
        maximum number of saccades per trial (including padding)

    Returns
    -------
    sacc_data : list of np.ndarray, shape (max_d,)
        Zero-padded saccade times (seconds)
    flag_data : np.ndarray
        Initial fixation per trial
    rt_data : np.ndarray
        Reaction times per trial (seconds)
    """
    sacc_data = []
    flag_data = []
    rt_data = []

    eps = 1e-12

    for fix in fixation_trials:
        fix = np.asarray(fix, dtype=int)

        # initial fixation
        flag_data.append(fix[0])

        # reaction time (inclusive)
        rt_data.append((len(fix) - 1) * dt)

        # detect fixation switches (indices)
        switch_idxs = np.where(fix[1:] != fix[:-1])[0] + 1

        # map indices -> times inside ((k-1)dt, kdt]
        sacc_times = switch_idxs * dt - eps

        # prepend true start time (left-inclusive)
        sacc_times = np.insert(sacc_times, 0, 0.0)

        # pad with zeros up to max_d
        padded = np.zeros(max_d, dtype=float)
        n = min(len(sacc_times), max_d)
        padded[:n] = sacc_times[:n]

        sacc_data.append(padded)

    return sacc_data, np.array(flag_data), np.array(rt_data)

def expand_fixations(sacc_data, flag_data, rt_data, dt):
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

    Returns
    -------
    array
        Each element is a tuple of fixation locations for one trial
    """
    all_trials = []

    for saccs, start_fix, rt in zip(sacc_data, flag_data, rt_data):

        # number of discrete timesteps (inclusive rt)
        fix_len = int(floor(rt / dt)) + 1

        # initialize fixation array
        fix = np.full(fix_len, start_fix)

        if len(saccs) > 0:
            # convert saccade times to indices
            switch_idxs = [int(ceil(s / dt)) for s in saccs]

            # apply alternating flips
            for idx in switch_idxs[1:]:
                if idx >= fix_len or idx <= 0:
                    continue
                fix[idx:] = 1 - fix[idx]

        all_trials.append(tuple(int(x) for x in fix))

    return all_trials

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

    if keep_cols is None:
        excluded = {subject_col, trial_col, seq_col, start_col, end_col, loc_col}
        if fixnum_col is not None:
            excluded.add(fixnum_col)

        keep_cols = [c for c in df.columns if c not in excluded]

    rows = []

    for row in df.itertuples(index=False):
        seq = np.asarray(getattr(row, seq_col), dtype=np.int64)

        changes = np.flatnonzero(seq[1:] != seq[:-1]) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(seq)]))
        locs = seq[starts]

        mask = ~np.isin(locs, list(fill_codes))

        starts = starts[mask]
        ends = ends[mask]
        locs = locs[mask]

        keep_vals = {c: getattr(row, c) for c in keep_cols}

        for i in range(len(starts)):
            data = {
                **keep_vals,
                subject_col: getattr(row, subject_col),
                trial_col: getattr(row, trial_col),
                start_col: starts[i],
                end_col: ends[i],
                loc_col: locs[i],
            }

            if fixnum_col is not None:
                data[fixnum_col] = i

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