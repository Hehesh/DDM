import numpy as np
import pandas as pd
from pyddm.models import OverlayChain
import pyddm
import random
import ast
import os

def create_model(drift_rate, theta, noise):
    # Wrapper function to hyperparameterize drift_rate
    def make_drift_function(drift_rate):
        def drift_function(avgWTP_left, avgWTP_right, fixation, t):
            fixation_index = min(int(t/0.001), len(fixation)-1)
            current_fixation = fixation[fixation_index]
            if current_fixation == 0: # saccade
                return 0
            elif current_fixation == 1: # left
                return drift_rate * (avgWTP_left -  avgWTP_right * theta)
            else: # right
                return drift_rate * (avgWTP_left * theta -  avgWTP_right)
        
        return drift_function
    
    # Define the drift function
    my_drift_function = make_drift_function(drift_rate)

    # Define the model
    model = pyddm.gddm(
        drift=my_drift_function,
        noise=noise,
        bound=1,
        nondecision=0,
        conditions=["avgWTP_left", "avgWTP_right", "fixation"],
        choice_names=("left", "right"),
        T_dur=10,
        dx=0.001,
        dt=0.001
    )

    model._overlay = OverlayChain(overlays=[])
    return model

def create_trials(num_trials, empirical_distributions, seed=42):
    ''' This function takes a number of trials and generates left and right
    value pairs, calling generate_fixations to generate an extra-long fixation tuple.
    A list of dictionaries is returned to be used in pyddm's model.simulate.'''

    random.seed(seed)

    trials = list()
    for _ in range(num_trials):
        # Sample weights according to empirical weighted average of frequencies of that relative value difference
        # keys = [k for k in empirical_distributions['fixations'][1].keys()]
        # weights = [len(empirical_distributions['fixations'][1][k]) for k in keys]
        # sampled_key = float(random.choices(keys, weights=weights, k=1)[0])

        # Change this line of code to represent either one subject or empirical value combos
        sampled_key = random.choice(np.linspace(-4, 4, 33))

        # Randomly generate a viable left value
        raw_max_left_value = ((5.0 - sampled_key) // 0.25) * 0.25
        max_left_value = min(5.0, raw_max_left_value)
        min_left_value = max(1.0, 1.0 - sampled_key)

        num_steps = int((max_left_value - min_left_value) / 0.25) + 1
        left_values = [round(min_left_value + i*0.25,2) for i in range(num_steps)]

        # Pick the left value
        left_value = random.choice(left_values)
        trials.append({'avgWTP_left': left_value, 'avgWTP_right': left_value + sampled_key, 'fixation': generate_fixations(0.001, sampled_key, empirical_distributions)})

    return trials

def get_empirical_distributions(
    df: pd.DataFrame,
    bin_size: float,
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
    time_step: float | None = None,
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
            start_idx = np.floor(events_df[start_col].to_numpy() / bin_size).astype(int)
            end_idx   = np.floor(events_df[end_col].to_numpy() / bin_size).astype(int)
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
            dur = run_len * bin_size

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
                    ok_lo = (time_step is None) or (dur >= time_step)
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

def generate_fixations(bin_size, relative_value_difference, empirical_distributions):
    """
    Generates a fixation sequence binned in time using empirical distributions.
    For subsequent fixations (after the first), pools from both signs of 
    relative value difference. The first fixation remains sign-dependent.
    """
    probLeftFirst = empirical_distributions['probFixLeftFirst']
    left_first = np.random.rand() < probLeftFirst

    # Flip sign for matching fixation distribution to fixated side
    rel_val_for_first = relative_value_difference if left_first else -relative_value_difference

    events = []
    global_time = 0.0

    # Initial latency
    latency = np.random.choice(empirical_distributions['latencies'])
    events.append((global_time, global_time + latency, 0))
    global_time += latency

    # First fixation — sign-dependent
    first_fixation_duration = np.random.choice(
        empirical_distributions['fixations'][1][rel_val_for_first]
    )
    code = 1 if left_first else 2
    events.append((global_time, global_time + first_fixation_duration, code))
    global_time += first_fixation_duration

    # Prepare for alternation
    current_side_code = 2 if code == 1 else 1
    abs_val = abs(relative_value_difference)

    def pooled_fixation_durations(fixation_dict, abs_val):
        durations = []
        if abs_val in fixation_dict:
            durations.extend(fixation_dict[abs_val])
        if -abs_val in fixation_dict:
            durations.extend(fixation_dict[-abs_val])
        return durations

    # Subsequent fixations — pooled
    while global_time < 30.0:
        # Transition
        transition_duration = np.random.choice(empirical_distributions['transitions'])
        events.append((global_time, global_time + transition_duration, 0))
        global_time += transition_duration

        # Fixation from pooled durations
        next_durations = pooled_fixation_durations(empirical_distributions['fixations'][2], abs_val)
        next_fixation_duration = np.random.choice(next_durations)
        events.append((global_time, global_time + next_fixation_duration, current_side_code))
        global_time += next_fixation_duration

        # Alternate side
        current_side_code = 2 if current_side_code == 1 else 1

    # Bin into time steps
    num_bins = int(30 / bin_size)
    fixation_sequence = []

    for i in range(num_bins):
        bin_start_time = i * bin_size
        for start, end, event_code in events:
            if start <= bin_start_time < end:
                fixation_sequence.append(event_code)
                break
        else:
            fixation_sequence.append(0)

    return tuple(fixation_sequence)

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

def simulate(bin_size, model_conditions, trials, seed=42, save_results=True):
    random.seed(seed)

    model = create_model(model_conditions['drift_rate'], model_conditions['theta'], model_conditions['noise'])

    results_df = list()

    for trial in trials:
        res = model.simulate_trial(trial)
        trial_fixation_tuple = trial['fixation'][:int(len(res)/(bin_size/model.dt))]
        trial_RT = round(len(trial_fixation_tuple)*bin_size, 3)
        if res[-1] >= 1:
            choice = 0
        elif res[-1] <= -1:
            choice = 1
        else:
            raise ValueError(f'Simulation ended before a decision was reached with last relative decision value of {res[-1]} but boundaries {model._bounddep.B} and {-1 * model._bounddep.B}. Please extend T_dur or modify parameters.')

        results_df.append({'trajectory': res, 'avgWTP_left': trial['avgWTP_left'], 'avgWTP_right': trial['avgWTP_right'], 'fixation': trial_fixation_tuple, 'RT': trial_RT, 'choice': choice})

    results_df = pd.DataFrame(results_df)
    if save_results:
        simulated_df = results_df[['choice', 'RT', 'avgWTP_left', 'avgWTP_right', 'fixation']]
        if not os.path.exists('simulated_data'):
            os.makedirs('simulated_data')
        simulated_df.to_csv(os.path.join('simulated_data', f'sim_trials_s{seed}_d{model_conditions['drift_rate']}_t{model_conditions['theta']}_n{model_conditions['noise']}.csv'), index=False)
    return results_df