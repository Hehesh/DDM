import numpy as np
import pandas as pd
import ast
import argparse

def get_empirical_distributions(path, bin_size):
    """
    Computes empirical fixation distributions from fixation tuples in a dataframe.
    
    Args:
      df: pandas DataFrame with columns including:
          - 'fixation' (tuple of 0,1,2,4)
          - 'avgWTP_left' (float)
          - 'avgWTP_right' (float)
      bin_size: duration in seconds per element of fixation tuple.
    
    Returns:
      A dict:
        {
          'probFixLeftFirst': float,
          'latencies': np.ndarray,
          'transitions': np.ndarray,
          'fixations': dict
        }
    """

    raw_df = pd.read_csv(path)
    exclusions = pd.read_csv('/Users/braydenchien/Desktop/Enkavilab/DDM/dropped_trials.csv')

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


# if __name__ == '__main__':
    # Save as csv
    