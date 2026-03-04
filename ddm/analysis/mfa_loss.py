import numpy as np
from .mfa_core import MFASummary

def model_free_loss(
    emp_mfa: dict[str, MFASummary],
    sim_mfa: dict[str, MFASummary],
    weights: dict[str, float] | None = None,
    *,
    # coverage / robustness parameters (global defaults)
    lambda_missing: float = 50000.0,
    gamma_precision: float = 0.0,
    min_overlap: int = 1,
    expected_x: dict[str, np.ndarray] | None = None,
    # optional per-metric overrides
    lambda_by_metric: dict[str, float] | None = None,
    gamma_by_metric: dict[str, float] | None = None,
) -> float:
    """
    Compute total model-free loss using coverage-aware summary losses.

    Parameters
    ----------
    emp_mfa, sim_mfa : dict[str, MFASummary]
        Empirical and simulated model-free analyses.
    weights : dict[str, float], optional
        Relative importance of each summary.
    lambda_missing : float
        Global penalty strength for missing bins. 
        Scaled with relevant losses.
    gamma_precision : float
        Global precision-weighted missing-bin penalty. 
        0.1 for gentle, 0.5 for strong.
    min_overlap : int
        Minimum number of overlapping bins before overlap loss is computed.
    expected_x : dict[str, np.ndarray], optional
        Explicit expected bin support per metric.
        If None, empirical x is used. 
        For large sample sizes for subjects, this will not be an issue.
    lambda_by_metric, gamma_by_metric : dict[str, float], optional
        Per-metric overrides for coverage penalties.
    """

    total = 0.0

    for name, emp in emp_mfa.items():
        sim = sim_mfa[name]

        # weight for this metric
        w = 1.0 if weights is None else weights.get(name, 1.0)

        # coverage penalties (allow per-metric override)
        lam = (
            lambda_missing
            if lambda_by_metric is None
            else lambda_by_metric.get(name, lambda_missing)
        )
        gam = (
            gamma_precision
            if gamma_by_metric is None
            else gamma_by_metric.get(name, gamma_precision)
        )

        # expected support (optional)
        exp_x = None if expected_x is None else expected_x.get(name)

        loss_k = mfa_summary_loss_with_coverage(
            emp=emp,
            sim=sim,
            lambda_missing=lam,
            gamma_precision=gam,
            min_overlap=min_overlap,
            expected_x=exp_x,
        )

        total += w * loss_k

    return float(total)

def mfa_summary_loss_with_coverage(
    emp: MFASummary,
    sim: MFASummary,
    eps: float = 1e-6,
    lambda_missing: float = 1.0,     # strength of coverage penalty
    gamma_precision: float = 0.0,    # 0 to disable precision-weighted missing penalty
    expected_x: np.ndarray | None = None,  # override expected bins if you want
    min_overlap: int = 1,            # allow 1-bin overlap (useful early in search)
) -> float:
    assert emp.name == sim.name

    # What bins do we "expect"?
    # Default: use empirical x as the expected support.
    B = np.asarray(expected_x if expected_x is not None else emp.x)

    # Overlap bins
    C = np.intersect1d(B, sim.x)

    # Missing bins from sim (relative to expected)
    M = np.setdiff1d(B, sim.x)

    # If no overlap, you still want a finite but bad loss
    if len(C) < min_overlap:
        overlap_loss = 0.0
    else:
        emp_mask = np.isin(emp.x, C)
        sim_mask = np.isin(sim.x, C)

        emp_y = emp.y[emp_mask]
        sim_y = sim.y[sim_mask]

        if emp.yerr is None:
            denom = 1.0
        else:
            denom = emp.yerr[emp_mask] ** 2 + eps

        overlap_loss = float(np.mean((sim_y - emp_y) ** 2 / denom))

    # Coverage penalty: fraction of expected bins missing
    missing_frac = len(M) / max(len(B), 1)
    coverage_penalty = lambda_missing * missing_frac

    # Optional: penalize missing bins more if empirical uncertainty is small
    precision_penalty = 0.0
    if gamma_precision > 0.0 and emp.yerr is not None and len(M) > 0:
        # map M -> indices in emp.x
        miss_mask_emp = np.isin(emp.x, M)
        # precision weight = 1/var
        precision_penalty = gamma_precision * float(
            np.mean(1.0 / (emp.yerr[miss_mask_emp] ** 2 + eps))
        )

    return overlap_loss + coverage_penalty + precision_penalty
