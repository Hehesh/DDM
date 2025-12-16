import argparse
import pyddm
import os
import pandas as pd
import ast

def create_subject_samples_dataframe_from_csv(path, sub_id, stimulus_visibility):
    df = pd.read_csv(os.path.join(path, f"pyddm_sample{sub_id}_{'HIDDEN' if stimulus_visibility.upper() == 'HIDDEN' else 'VISIBLE'}.csv"))
    df['fixation'] = df['fixation'].apply(ast.literal_eval)    
    return pyddm.Sample.from_pandas_dataframe(df, choice_column_name="choice", rt_column_name="RT", choice_names=("left", "right"))

def subject_max_RT(path, sub_id, stimulus_visibility):
    df = pd.read_csv(os.path.join(path, f"pyddm_sample{sub_id}_{'HIDDEN' if stimulus_visibility.upper() == 'HIDDEN' else 'VISIBLE'}.csv")) 
    return max(df['RT'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("sub_id", type=int)
    parser.add_argument("stimulus_visibility", type=str)
    parser.add_argument("bin_size", type=float)
    args = parser.parse_args()

    sample = create_subject_samples_dataframe_from_csv(args.path, args.sub_id, args.stimulus_visibility)

    pyddm.set_N_cpus(48)

    def drift_func(avgWTP_left, avgWTP_right, fixation, d, theta, t):
        current_fixation = fixation[min(int(t//args.bin_size), len(fixation)-1)]
        if current_fixation == 0:
            return d * (avgWTP_left - avgWTP_right * theta)
        else:
            return -d * (avgWTP_right - avgWTP_left * theta)

    model = pyddm.gddm(
        drift=drift_func,
        noise=0.1,
        bound=1,
        nondecision=0,
        mixture_coef=0.02,
        parameters={"d": (0, 0.005), "theta": (0, 0.75)},
        conditions=["avgWTP_left", "avgWTP_right", "fixation"],
        choice_names=("left", "right"),
        T_dur=subject_max_RT(args.path, args.sub_id, args.stimulus_visibility) + 1.5,
        dx=0.001,
        dt=0.01
    )

    fit_result = pyddm.fit_adjust_model(sample=sample, model=model, verbose=True)

    with open(f"{args.sub_id}_fit_stats.txt", "w") as f:
        print(fit_result, file=f)

