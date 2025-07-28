import pandas as pd
import numpy as np
import os
import ast

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

    if not os.path.exists('formatted_data'):
        os.makedirs('formatted_data')
    result_df.to_csv(os.path.join('formatted_data', path), index=False)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('parent_dir', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()
    reformat_fixations(args.parent_dir, args.path)