import numpy as np
import pandas as pd

def get_empirical_distributions(
    df: pd.DataFrame,
    value_diffs: list,
    *,
    legend: dict = None,
    fixation_col: str = None,
    v_left_col: str = "avgWTP_left",
    v_right_col: str = "avgWTP_right",
    trial_col: str = None,
    start_col: str = None,
    end_col: str = None,
    location_col: str = None,
    assume_bins: bool = True,
    num_fix_dists: int = 2,
    min_fix_time: float | None = None,
    max_fix_time: float | None = None,
):
    """
    Compute empirical fixation duration, latency, and transition distributions
    from fixation sequences or fixation event tuples, with flexible legend mapping,
    customizable column names, and ordinal fixation buckets.

    Supports either:
        (A) rasterized fixation sequences per trial (via `fixation_col`), or
        (B) start–end–location tuples (via `start_col`, `end_col`, `location_col`, grouped by `trial_col`).

    Estimates:
      • Probability that the first fixation is on the left option
      • Distribution of latencies (time to first fixation)
      • Distribution of transition durations between fixations
      • Fixation duration distributions split by fixation ordinal (1..num_fix_dists,
        with the last bucket pooling Nth+), and binned by signed value difference
        (fixated − unfixated)

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing either rasterized fixation sequences or fixation events.

    bin_size : float
        Time represented by one element in a fixation sequence, in seconds (or any consistent unit).

    value_diffs : list of float
        The set of possible (or desired) signed value differences to bin by. Each fixation
        duration is assigned to the closest bin in this list.

    legend : dict, optional
        Mapping of raw fixation codes to categories. Example:
            legend = {
                "left": {1},
                "right": {2},
                "transition": {0, 3, 4},
                # "ignore": set()  # optional
            }
        Default: assumes 1 = left, 2 = right, and {0,3,4} = transitions.
        Values may be lists or sets; they’re normalized to sets internally.

    fixation_col : str, optional
        Column containing pre-rasterized fixation sequences (list/array of codes) per trial.
        If provided, `start_col`/`end_col`/`location_col` are ignored.

    v_left_col, v_right_col : str
        Column names for the left/right values per trial (default: "avgWTP_left", "avgWTP_right").

    trial_col : str, optional
        Trial identifier column. Required if reconstructing from events.

    start_col, end_col : str, optional
        Start and end indices (inclusive) for each event row if reconstructing from events.
        If `assume_bins=False`, they are treated as times and converted using `bin_size`.

    location_col : str, optional
        Side for each event row ('left'/'right' strings or codes resolvable via `legend`).

    assume_bins : bool, default True
        If True, start/end are integer bin indices; if False, they are times to be
        converted to indices via `bin_size`.

    num_fix_dists : int, default 2
        Number of ordinal fixation buckets. Buckets are 1..num_fix_dists; the last bucket
        pools all remaining fixations (e.g., with 3 → 1st, 2nd, and 3rd+).

    Returns
    -------
    dict
        {
            "probFixLeftFirst": float,
            "latencies": np.ndarray,        # pre-first-fixation durations
            "transitions": np.ndarray,      # post-first transition durations
            "fixations": {                  # ordinal -> value_diff_bin -> np.ndarray of durations
                1: {vd: np.ndarray, ...},
                2: {vd: np.ndarray, ...},
                ...
                num_fix_dists: {vd: np.ndarray, ...}
            }
        }
    """
    # ---------- Legend normalization ----------
    # Default now distinguishes 0 vs 3 explicitly
    if legend is None:
        legend = {"left": {1}, "right": {2}, "transition": {0}, "blank_fixation": {3}}
    Lset = set(legend.get("left", set()))
    Rset = set(legend.get("right", set()))
    Tset = set(legend.get("transition", {0}))   # codes that behave like 0 (transitions)
    Bset = set(legend.get("blank_fixation", {3}))   # codes that behave like 3 (do NOT count as transitions after first fix)
    ignore_set = set(legend.get("ignore", set()))

    def classify(code):
        if code in ignore_set:
            return None
        if code in Lset:
            return "L"
        if code in Rset:
            return "R"
        if code in Tset:
            return "T"
        if code in Bset:
            return "B"
        return "T"        # default any other encodings as a transition

    def closest_bin(x, bins):
        return min(bins, key=lambda b: abs(x - b))

    def rasterize_from_events(events_df):
        if events_df.empty:
            return np.array([], dtype=object)
        if assume_bins:
            start_idx = events_df[start_col].to_numpy(dtype=int)
            end_idx   = events_df[end_col].to_numpy(dtype=int)
        else:
            start_idx = np.floor(events_df[start_col].to_numpy()).astype(int)
            end_idx   = np.floor(events_df[end_col].to_numpy()).astype(int)
        L = int(np.max(end_idx)) + 1
        seq = np.empty(L, dtype=object)
        seq[:] = None
        locs = events_df[location_col].to_numpy()
        for s, e, loc in zip(start_idx, end_idx, locs):
            if isinstance(loc, str):
                key = loc.strip().lower()
                code = "left" if key == "left" else ("right" if key == "right" else loc)
            else:
                code = loc
            seq[s:e+1] = code
        mask_none = (seq == None)  # noqa: E711
        if mask_none.any():
            seq[mask_none] = 0
        return seq

    # ---------- Output containers ----------
    valueDiffs = list(value_diffs)
    fixations_dict = {k: {v: [] for v in valueDiffs} for k in range(1, num_fix_dists + 1)}
    latencies_list, transitions_list = [], []
    count_left_first = 0
    count_total_first = 0

    # ---------- Iterate trials ----------
    if fixation_col is not None:
        trial_iter = df.itertuples(index=False)
        get_seq = lambda row: getattr(row, fixation_col)
        get_vL = lambda row: getattr(row, v_left_col)
        get_vR = lambda row: getattr(row, v_right_col)
    else:
        if not all([trial_col, start_col, end_col, location_col]):
            raise ValueError("To reconstruct from events, provide trial_col, start_col, end_col, and location_col.")
        grouped = df.groupby(trial_col, sort=False)
        def event_trial_iter():
            for trial_id, g in grouped:
                if v_left_col not in g or v_right_col not in g:
                    raise ValueError(f"Missing {v_left_col}/{v_right_col} for trial {trial_id}.")
                vL = g.iloc[0][v_left_col]
                vR = g.iloc[0][v_right_col]
                seq = rasterize_from_events(g)
                yield trial_id, seq, vL, vR
        trial_iter = event_trial_iter()
        get_seq = lambda item: item[1]
        get_vL  = lambda item: item[2]
        get_vR  = lambda item: item[3]

    for row in trial_iter:
        raw_seq = np.asarray(get_seq(row), dtype=object)
        if raw_seq.size == 0:
            continue

        # Classify each bin
        cls = np.array([classify(c) for c in raw_seq], dtype=object)
        keep = cls != None  # noqa: E711
        if not np.any(keep):
            continue
        cls = cls[keep]

        v_left  = float(get_vL(row))
        v_right = float(get_vR(row))
        sv_lookup = {"L": v_left - v_right, "R": v_right - v_left}

        lr_idx = np.flatnonzero((cls == "L") | (cls == "R"))
        if lr_idx.size <= 1:
            continue

        # Exclude the final censored L/R run from the tail
        exclude = np.zeros(cls.size, dtype=bool)
        last_i = lr_idx[-1]
        last_label = cls[last_i]
        j = last_i
        while j >= 0 and cls[j] == last_label:
            exclude[j] = True
            j -= 1
        exclude[last_i + 1 :] = True

        # Iterate runs (NOTE: with N0 vs N3, these do NOT merge)
        N = cls.size
        i = 0
        first_fix_reached = False
        latency_time = 0.0
        fix_number = 1

        while i < N:
            if exclude[i]:
                i += 1
                continue

            lab = cls[i]

            # grow contiguous run of same 'lab' (skipping excluded positions)
            start = i
            i += 1
            while i < N and (exclude[i] or cls[i] == lab):
                i += 1

            # run length in included bins
            run_len = np.count_nonzero(~exclude[start:i])
            if run_len == 0:
                continue
            dur = run_len

            if lab in ("L", "R"):
                if not first_fix_reached:
                    first_fix_reached = True
                    count_total_first += 1
                    if lab == "L":
                        count_left_first += 1
                    latencies_list.append(latency_time)

                b = closest_bin(sv_lookup[lab], valueDiffs)
                bucket = min(fix_number, num_fix_dists)
                fixations_dict[bucket][b].append(dur)
                fix_number += 1

            elif lab in ("T", "B"):
                # BEFORE first fixation: both T and B contribute to latency
                if not first_fix_reached:
                    latency_time += dur
                else:
                    # AFTER first fixation: treat each T or B run as its own transition
                    ok_lo = (min_fix_time is None) or (dur >= min_fix_time)
                    ok_hi = (max_fix_time is None) or (dur <= max_fix_time)
                    if ok_lo and ok_hi:
                        transitions_list.append(dur)

        prob_fix_left_first = (count_left_first / count_total_first) if count_total_first > 0 else np.nan

    return {
        "probFixLeftFirst": prob_fix_left_first,
        "latencies": np.array(latencies_list, dtype=float),
        "transitions": np.array(transitions_list, dtype=float),
        "fixations": {
            k: {kk: np.array(vv, dtype=float) for kk, vv in v.items()}
            for k, v in fixations_dict.items()
        }
    }

def create_numbered_empirical_distributions(
    df: pd.DataFrame,
    *,
    legend: dict = None,
    fixation_col: str = None,
    trial_col: str = None,
    start_col: str = None,
    end_col: str = None,
    location_col: str = None,
    assume_bins: bool = True,
    num_fix_dists: int = 3,
    min_fix_time: float | None = None,
    max_fix_time: float | None = None,
):
    """
    Same as get_empirical_distributions, but fixation durations are pooled
    purely by fixation number (ordinal), not by value-difference bins.

    Returns:
    {
        "probFixLeftFirst": float,
        "latencies": np.ndarray,
        "transitions": np.ndarray,
        "fixations": {
            1: np.ndarray,
            2: np.ndarray,
            ...
        }
    }
    """

    # ---------- Legend normalization ----------
    if legend is None:
        legend = {"left": {1}, "right": {2}, "transition": {0}, "blank_fixation": {3}}

    Lset = set(legend.get("left", set()))
    Rset = set(legend.get("right", set()))
    Tset = set(legend.get("transition", {0}))
    Bset = set(legend.get("blank_fixation", {3}))
    ignore_set = set(legend.get("ignore", set()))

    def classify(code):
        if code in ignore_set:
            return None
        if code in Lset:
            return "L"
        if code in Rset:
            return "R"
        if code in Tset:
            return "T"
        if code in Bset:
            return "B"
        return "T"

    def rasterize_from_events(events_df):
        if events_df.empty:
            return np.array([], dtype=object)

        if assume_bins:
            start_idx = events_df[start_col].to_numpy(dtype=int)
            end_idx = events_df[end_col].to_numpy(dtype=int)
        else:
            start_idx = np.floor(events_df[start_col].to_numpy()).astype(int)
            end_idx = np.floor(events_df[end_col].to_numpy()).astype(int)

        L = int(np.max(end_idx)) + 1
        seq = np.empty(L, dtype=object)
        seq[:] = None

        locs = events_df[location_col].to_numpy()

        for s, e, loc in zip(start_idx, end_idx, locs):
            if isinstance(loc, str):
                key = loc.strip().lower()
                code = "left" if key == "left" else ("right" if key == "right" else loc)
            else:
                code = loc
            seq[s:e+1] = code

        seq[seq == None] = 0  # noqa: E711
        return seq

    # ---------- Containers ----------
    fixations_dict = {k: [] for k in range(1, num_fix_dists + 1)}
    latencies_list, transitions_list = [], []
    count_left_first = 0
    count_total_first = 0

    # ---------- Trial iterator ----------
    if fixation_col is not None:
        trial_iter = df.itertuples(index=False)
        get_seq = lambda row: getattr(row, fixation_col)
    else:
        if not all([trial_col, start_col, end_col, location_col]):
            raise ValueError("Missing event reconstruction columns.")

        grouped = df.groupby(trial_col, sort=False)

        def event_trial_iter():
            for _, g in grouped:
                yield rasterize_from_events(g)

        trial_iter = event_trial_iter()
        get_seq = lambda seq: seq

    # ---------- Main loop ----------
    for row in trial_iter:
        raw_seq = np.asarray(get_seq(row), dtype=object)
        if raw_seq.size == 0:
            continue

        cls = np.array([classify(c) for c in raw_seq], dtype=object)
        keep = cls != None  # noqa: E711
        if not np.any(keep):
            continue
        cls = cls[keep]

        lr_idx = np.flatnonzero((cls == "L") | (cls == "R"))
        if lr_idx.size <= 1:
            continue

        exclude = np.zeros(cls.size, dtype=bool)
        last_i = lr_idx[-1]
        last_label = cls[last_i]

        j = last_i
        while j >= 0 and cls[j] == last_label:
            exclude[j] = True
            j -= 1
        exclude[last_i + 1:] = True

        N = cls.size
        i = 0
        first_fix_reached = False
        latency_time = 0.0
        fix_number = 1

        while i < N:
            if exclude[i]:
                i += 1
                continue

            lab = cls[i]
            start = i
            i += 1

            while i < N and (exclude[i] or cls[i] == lab):
                i += 1

            run_len = np.count_nonzero(~exclude[start:i])
            if run_len == 0:
                continue

            dur = run_len

            if lab in ("L", "R"):
                if not first_fix_reached:
                    first_fix_reached = True
                    count_total_first += 1
                    if lab == "L":
                        count_left_first += 1
                    latencies_list.append(latency_time)

                bucket = min(fix_number, num_fix_dists)
                fixations_dict[bucket].append(dur)
                fix_number += 1

            elif lab in ("T", "B"):
                if not first_fix_reached:
                    latency_time += dur
                else:
                    ok_lo = (min_fix_time is None) or (dur >= min_fix_time)
                    ok_hi = (max_fix_time is None) or (dur <= max_fix_time)
                    if ok_lo and ok_hi:
                        transitions_list.append(dur)

    prob_fix_left_first = (
        count_left_first / count_total_first
        if count_total_first > 0
        else np.nan
    )

    return {
        "probFixLeftFirst": prob_fix_left_first,
        "latencies": np.array(latencies_list, dtype=float),
        "transitions": np.array(transitions_list, dtype=float),
        "fixations": {
            k: np.array(v, dtype=float) for k, v in fixations_dict.items()
        },
    }