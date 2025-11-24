import numpy as np

def generate_fixations(bin_size, relative_value_difference, empirical_distributions, total_duration_s=30.0, seed=42):
    """
    Return a tuple of event codes binned at `bin_size` seconds for `total_duration_s` seconds.
    Assumes empirical distributions may be in ms; converts to seconds.
    """
    try:
        rng = np.random.default_rng(seed)

        # --- Normalize units to seconds ---
        def to_seconds(arr):
            arr = np.asarray(arr)
            return arr / 1000.0 if np.median(arr) > 10 else arr

        latencies_s   = to_seconds(empirical_distributions['latencies'])
        transitions_s = to_seconds(empirical_distributions['transitions'])

        # fixations structure: {1: {rel_val: [durations]}, 2: {...}}
        # Convert all fixation durations to seconds, but preserve the dict shape.
        def fix_to_seconds(fix_dict):
            out = {}
            for rel_val, durations in fix_dict.items():
                out[rel_val] = to_seconds(durations)
            return out

        fix1 = fix_to_seconds(empirical_distributions['fixations'][1])
        fix2 = fix_to_seconds(empirical_distributions['fixations'][2])

        # Helper: get durations for a given rel_val with nearest-key fallback (floats as keys are brittle)
        def get_durations(dct, key):
            if key in dct:
                return dct[key]
            # nearest key fallback
            k = min(dct.keys(), key=lambda kk: abs(kk - key))
            return dct[k]

        # Pooled durations (abs value across signs)
        def pooled_fixation_durations(fixation_dict, abs_val):
            pool = []
            # exact match
            if  abs_val in fixation_dict:  pool.extend(fixation_dict[abs_val])
            if -abs_val in fixation_dict:  pool.extend(fixation_dict[-abs_val])
            # nearest-key fallback if pool empty
            if not len(pool):
                k1 = min(fixation_dict.keys(), key=lambda kk: abs(abs(kk) - abs_val))
                pool.extend(fixation_dict[k1])
            return np.asarray(pool)

        probLeftFirst = float(empirical_distributions['probFixLeftFirst'])
        left_first = rng.random() < probLeftFirst

        # First fixation sign selection
        rel_val_for_first = relative_value_difference if left_first else -relative_value_difference

        events = []  # (start_s, end_s, code)
        global_time = 0.0

        # Initial latency
        latency = rng.choice(latencies_s)
        events.append((global_time, global_time + latency, 0))
        global_time += latency

        # First fixation — sign dependent
        first_fixation_duration = rng.choice(get_durations(fix1, rel_val_for_first))
        code = 1 if left_first else 2
        events.append((global_time, global_time + first_fixation_duration, code))
        global_time += first_fixation_duration

        # Prepare for alternation
        current_side_code = 2 if code == 1 else 1
        abs_val = abs(relative_value_difference)

        # Subsequent fixations — pooled
        # Guard against empty transitions or pooled durations
        if len(transitions_s) == 0:
            raise ValueError("empirical_distributions['transitions'] is empty after unit conversion.")
        pooled2 = pooled_fixation_durations(fix2, abs_val)
        if len(pooled2) == 0:
            raise ValueError("No pooled fixation durations for subsequent fixations.")

        while global_time < total_duration_s:
            # Transition
            tdur = rng.choice(transitions_s)
            start = global_time
            end   = min(global_time + tdur, total_duration_s)
            events.append((start, end, 0))
            global_time = end
            if global_time >= total_duration_s: break

            # Fixation
            fdur = rng.choice(pooled2)
            start = global_time
            end   = min(global_time + fdur, total_duration_s)
            events.append((start, end, current_side_code))
            global_time = end

            # Alternate side
            current_side_code = 2 if current_side_code == 1 else 1

        # Bin to fixed time grid in seconds
        num_bins = int(np.floor(total_duration_s / bin_size))
        fixation_sequence = []
        # Sweep through events once (O(Nbins + Nevents))
        e_idx = 0
        for i in range(num_bins):
            t = i * bin_size
            # advance event pointer
            while e_idx < len(events) and not (events[e_idx][0] <= t < events[e_idx][1]):
                e_idx += 1
            if e_idx < len(events) and events[e_idx][0] <= t < events[e_idx][1]:
                fixation_sequence.append(events[e_idx][2])
            else:
                fixation_sequence.append(0)

        return tuple(fixation_sequence)
    except Exception:
        return None