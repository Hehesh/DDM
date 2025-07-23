import os
import pandas as pd
import argparse

# In this file, I need a way to get from the data in all subjects to trial data
# The return will be a dataframe of all trials. Additionally, it will be labeled
# So something like this:

# sub_id | trial | hidden/visible | avgWTP_left | avgWTP_right | choice | RT | fixation

# create_all_trials_CSV <-- create_trial_dataframe <-- calculate_trial_fixation

def calculate_trial_fixation(parent_dir, sub_id, bin_size, trial):
    # Load fixation data
    fixations_df = pd.read_csv(os.path.join(parent_dir, sub_id, f'fixations_{sub_id}.csv'))
    fixations_df = fixations_df[fixations_df['trial'] == trial]

    fixation = []
    global_time = 0  # in seconds
    current_time = 0  # in seconds

    for row in fixations_df.itertuples():
        duration_sec = row.duration / 1000  # convert ms → s
        end_time = current_time + duration_sec

        while global_time < end_time:
            fixation.append(row.location)
            global_time += bin_size

        current_time = end_time

    return tuple(fixation)

def create_all_trials_CSV(parent_dir, output_dir, bin_size):
    # parent_dir = /hopper/groups/enkavilab/users/bchien/eum2023_data_code/data/joint
    rows = []
    for sub_id in os.listdir(parent_dir):

        # Indexing for choices csv
        if sub_id.startswith('2'):
            num_trials = 360
        elif sub_id.startswith('3'):
            num_trials = 400
        else:
            raise RuntimeError(f"Check parent folder path. Is this right: {parent_dir}?")
        
        # Reading choices csv's at once
        choices_dfs = {}
        for i in range(1, 5):
            path = os.path.join(parent_dir, sub_id, f'choices_s{i}_{sub_id}.csv')
            choices_dfs[i] = pd.read_csv(path)
        
        for file_number, choices_df in choices_dfs.items():
            for trial in choices_df['trial']:
                row = choices_df.loc[choices_df['trial'] == trial].squeeze()
                global_trial_num = int(trial + (file_number - 1) * num_trials / 4)

                isHidden = row['type'] == 'Hidden'
                choice = 'left' if row['choice'] == 1 else 'right'
                fixation = calculate_trial_fixation(parent_dir, sub_id, bin_size, global_trial_num)
                rt = round(len(fixation) * bin_size, 3)

                rows.append({
                    'sub_id': sub_id,
                    'trial': global_trial_num,
                    'hidden': isHidden,
                    'avgWTP_left': row['avgWTP_left'],
                    'avgWTP_right': row['avgWTP_right'],
                    'choice': choice,
                    'RT': rt,
                    'fixation': fixation
                })
        
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, f"{int(bin_size*1000)}ms_trial_data.csv"), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("bin_size", type=float)
    args = parser.parse_args()

    create_all_trials_CSV(args.parent_dir, args.output_dir, args.bin_size)