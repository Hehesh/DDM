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

def get_empirical_distributions(path, bin_size):
    """ Computes empirical fixation distributions from fixation 
    tuples in a dataframe."""

    raw_df = pd.read_csv(path) # path for 1ms_trial_data.csv
    exclusions = pd.read_csv('/Users/braydenchien/Desktop/Enkavilab/DDM/dropped_trials.csv') # Replace with your path

    excluded_subjects = set(exclusions['parcode'].dropna().unique())
    excluded_trials = set(exclusions['trial'].dropna().unique()) if 'exclude_trial' in exclusions.columns else set()

    df = raw_df.loc[
                    (raw_df['hidden'] == False) &                             # only visible trials
                    (~raw_df['sub_id'].isin(excluded_subjects)) &         # exclude certain subjects
                    (~raw_df['trial'].isin(excluded_trials))               # exclude certain trials
    ].copy()
    df['fixation'] = df['fixation'].apply(ast.literal_eval)
    
    count_left_first = 0
    count_total_first = 0
    
    latencies_list = []
    transitions_list = []

    valueDiffs = tuple(np.arange(-4, 4.25, 0.25))
    # Nested dict: fixation_number -> signed value difference -> list of durations
    fixations_dict = {1: {v: [] for v in valueDiffs},
                      2: {v: [] for v in valueDiffs}}
    
    for _, row in df.iterrows():
        fixation = row['fixation']
        v_left = row['avgWTP_left']
        v_right = row['avgWTP_right']
        
        # Signed value diff: fixated minus unfixated
        signed_value_diff_lookup = {
            1: v_left - v_right,
            2: v_right - v_left
        }
        
        # 1. Find indices where fixations are 1 or 2
        fixation_indices = [i for i, f in enumerate(fixation) if f in (1,2)]
        if len(fixation_indices) <= 1:
            continue  # skip if no fixations or only 1 (can't compute transitions)
        
        # 2. Identify the last consecutive fixation group to exclude
        last_idx = fixation_indices[-1]
        last_fix_value = fixation[last_idx]
        exclude_indices = set()
        i = last_idx
        while i >= 0 and fixation[i]==last_fix_value:
            exclude_indices.add(i)
            i -= 1
        
        # 3. Process fixations
        first_fix_reached = False
        latency_time = 0.0
        fix_number = 1
        
        i = 0
        while i < len(fixation):
            f = fixation[i]
            
            # Skip excluded indices
            if i in exclude_indices:
                i += 1
                continue
            
            if f in (1,2):
                if not first_fix_reached:
                    first_fix_reached = True
                    count_total_first +=1
                    if f ==1:
                        count_left_first +=1
                    latencies_list.append(latency_time)
                
                # Find consecutive run length
                start = i
                while i+1 < len(fixation) and fixation[i+1]==f and (i+1) not in exclude_indices:
                    i+=1
                run_length = i - start +1
                duration = run_length * bin_size
                
                # Bin signed value difference
                signed_value = signed_value_diff_lookup[f]
                # Find closest bin in valueDiffs
                closest_bin = min(valueDiffs, key=lambda x: abs(x - signed_value))
                
                if fix_number ==1:
                    fixations_dict[1][closest_bin].append(duration)
                else:
                    fixations_dict[2][closest_bin].append(duration)
                
                fix_number +=1
                i +=1
            elif f in (0,4):
                if not first_fix_reached:
                    latency_time += bin_size
                else:
                    start = i
                    while i+1 < len(fixation) and fixation[i+1] in (0,4) and (i+1) not in exclude_indices:
                        i+=1
                    transition_length = i - start +1
                    duration = transition_length * bin_size
                    transitions_list.append(duration)
                i+=1
            else:
                i+=1
    
    prob_fix_left_first = count_left_first / count_total_first if count_total_first>0 else np.nan
    
    return {
        'probFixLeftFirst': prob_fix_left_first,
        'latencies': np.array(latencies_list),
        'transitions': np.array(transitions_list),
        'fixations': {k: {kk: np.array(vv) for kk,vv in v.items()} for k,v in fixations_dict.items()}
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