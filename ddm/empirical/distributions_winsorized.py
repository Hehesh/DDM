from .distributions import *

def get_corrected_empirical_distributions(
        df, 
        value_diffs, 
        *,
        legend: dict = None, 
        fixation_col: str = None, 
        left_value_col: str = None, 
        right_value_col: str = None, 
        cutoff: float = 0.9   # central percentage kept
    ):

    empirical_distributions = get_empirical_distributions(
        df, 
        value_diffs, 
        legend=legend, 
        fixation_col=fixation_col, 
        v_left_col=left_value_col, 
        v_right_col=right_value_col,
        num_fix_dists=3,
        min_fix_time=10,
        max_fix_time=3000
    )

    def winsorize_list(lst, middle_frac=0.9):
        """
        Winsorize to the central `middle_frac` of the data.
        E.g. middle_frac=0.9 -> bottom 5% and top 5% clipped.
        """
        if len(lst) == 0:
            return lst

        arr = np.asarray(sorted(lst))
        alpha = (1.0 - middle_frac) / 2.0

        lo = np.quantile(arr, alpha)
        hi = np.quantile(arr, 1.0 - alpha)

        arr = np.clip(arr, lo, hi)
        return arr.tolist()

    # Latencies
    empirical_distributions['latencies'] = winsorize_list(
        empirical_distributions['latencies'], cutoff
    )

    # Transitions
    empirical_distributions['transitions'] = winsorize_list(
        empirical_distributions['transitions'], cutoff
    )

    # Fixation 1
    for key in empirical_distributions['fixations'][1]:
        empirical_distributions['fixations'][1][key] = winsorize_list(
            empirical_distributions['fixations'][1][key], cutoff
        )

    # Fixation 2
    for key in empirical_distributions['fixations'][2]:
        empirical_distributions['fixations'][2][key] = winsorize_list(
            empirical_distributions['fixations'][2][key], cutoff
        )

    return empirical_distributions

def get_corrected_numbered_empirical_distributions(
    df,
    *,
    legend=None,
    fixation_col=None,
    cutoff=0.9,
):
    empirical = create_numbered_empirical_distributions(
        df,
        legend=legend,
        fixation_col=fixation_col,
    )

    def winsorize(arr, middle_frac=0.9):
        if len(arr) == 0:
            return arr
        arr = np.sort(np.asarray(arr))
        alpha = (1.0 - middle_frac) / 2.0
        lo = np.quantile(arr, alpha)
        hi = np.quantile(arr, 1.0 - alpha)
        return np.clip(arr, lo, hi)

    empirical["latencies"] = winsorize(empirical["latencies"], cutoff)
    empirical["transitions"] = winsorize(empirical["transitions"], cutoff)

    for k in empirical["fixations"]:
        empirical["fixations"][k] = winsorize(
            empirical["fixations"][k], cutoff
        )

    return empirical