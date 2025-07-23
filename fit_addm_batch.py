import pandas as pd
import pyddm
import os
import argparse

def create_batch_samples_dataframe_from_csv(path, batch_num):
    df = pd.DataFrame({'choice': [], 'RT': [], 'avgWTP_left': [], 'avgWTP_right': [],'fixation': []})
    for item in os.listdir(path):
        if int(item[12]) == batch_num+1:
            df = pd.concat([df, pd.read_csv(os.path.join(path, item))], ignore_index=True)
    
    return pyddm.Sample.from_pandas_dataframe(df, choice_column_name="choice", rt_column_name="RT", choice_names=("left", "right"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("batch_num", type=int)
    parser.add_argument("bin_size", type=float)
    args = parser.parse_args()

    sample = create_batch_samples_dataframe_from_csv(args.path, args.batch_num)

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
        parameters={"d": (0, 0.005), "theta": (0, 1)},
        conditions=["avgWTP_left", "avgWTP_right", "fixation"],
        choice_names=("left", "right"),
        T_dur=30,
        dx=0.001,
        dt=0.01
    )

    fit_result = pyddm.fit_adjust_model(sample=sample, model=model, verbose=True)

    with open(f"batch{args.batch_num}_fit_stats.txt", "w") as f:
        f.write(f"Fitting method: {fit_result.fitting_method}\n")
        f.write(f"Optimizer: {fit_result.method}\n")
        f.write(f"Loss function: {fit_result.loss}\n")
        f.write(f"Final loss value: {fit_result.val:.4f}\n\n")

        for key, value in fit_result.properties.items():
            f.write(f"{key}: {value:.4f}\n")

