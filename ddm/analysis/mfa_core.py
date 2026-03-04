import numpy as np
import pandas as pd
from dataclasses import dataclass
from .mfa_metrics import *

@dataclass
class MFASummary:
    name: str
    x: np.ndarray
    y: np.ndarray
    yerr: np.ndarray | None

def compute_mfa(df: pd.DataFrame) -> dict[str, MFASummary]:
    return {
        "p_first_fix_to_best": mfa_first_fix_to_best(df),
        "fix_dur_by_position": mfa_fix_dur_by_position(df),
        "middle_fix_dur_vs_absdiff": mfa_middle_fix_dur_vs_absdiff(df),
        "first_fix_dur_vs_absdiff": mfa_first_fix_dur_vs_absdiff(df),
        "net_fix_dur_vs_signeddiff": mfa_net_fix_dur_vs_signeddiff(df),
        "choice_psychometric": mfa_choice_psychometric(df),
        "rt_vs_absdiff": mfa_rt_vs_absdiff(df),
        "nfix_vs_absdiff": mfa_nfix_vs_absdiff(df),
    }