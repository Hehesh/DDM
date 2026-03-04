import numpy as np
from ddm.models import aDDModel
from efficient_fpt.utils import get_alternating_addm_mu_array

def simulate_empirical_trial(
    n,
    r1_data,
    r2_data,
    empirical_distributions,
    left_bias,
    *,
    eta, kappa, sigma, a, b, T, x0, dt, seed
):
    rng = np.random.default_rng(seed)

    r1 = r1_data[n]
    r2 = r2_data[n]

    # --- Generate fixations ---
    if left_bias:
        flag = rng.binomial(1, empirical_distributions['probFixLeftFirst']) # 1 for left first, 0 for right first
    else:
        flag = rng.binomial(1, 0.5) 

    fixations = []
    total_dur = 0.0

    latency = rng.choice(empirical_distributions['latencies'])
    fixations.append(latency)
    total_dur += latency

    diff = r1 - r2 # left - right
    if not flag: # right first
        diff = -diff
    fix_1 = rng.choice(empirical_distributions['fixations'][1][diff])
    fixations.append(fix_1)
    total_dur += fix_1

    while total_dur <= T:
        transition = rng.choice(empirical_distributions['transitions'])
        fixations.append(transition)
        total_dur += transition

        diff = -diff
        fix_dur = rng.choice(empirical_distributions['fixations'][2][diff])
        fixations.append(fix_dur)
        total_dur += fix_dur

    sacc_array = np.insert(np.cumsum(fixations), 0, 0.0)
    sacc_array = sacc_array[sacc_array < T] * dt

    # --- First fixation side ---
    mu1 = mu2 = None
    if flag: # left first
        mu1 = kappa * (r1 - eta * r2)
        mu2 = kappa * (eta * r1 - r2)
    else: # right first
        mu1 = kappa * (eta * r1 - r2)
        mu2 = kappa * (r1 - eta * r2)

    addm = aDDModel(
        mu1=mu1, mu2=mu2,
        sacc_array=sacc_array,
        flag=flag,
        sigma=sigma, a=a, b=b, x0=x0
    )

    decision = addm.simulate_fpt_datum(dt=dt)

    sacc_array = sacc_array[sacc_array < decision[0]]
    d = len(sacc_array)
    mu_array = get_alternating_addm_mu_array(mu1, mu2, d, flag)

    return decision, mu_array, sacc_array, r1, r2, flag

def simulate_numbered_empirical_trial(
    n,
    r1_data,
    r2_data,
    empirical_distributions,
    *,
    eta, kappa, sigma, a, b, T, x0, dt, seed
):
    rng = np.random.default_rng(seed)

    r1 = r1_data[n]
    r2 = r2_data[n]

    # --- Generate fixations ---
    # flag = rng.binomial(1, empirical_distributions['probFixLeftFirst']) # 1 for left first, 0 for right first
    flag = rng.binomial(1, 0.5) 

    fixations = []
    total_dur = 0.0

    latency = rng.choice(empirical_distributions['latencies'])
    fixations.append(latency)
    total_dur += latency

    fix_1 = rng.choice(empirical_distributions['fixations'][1])
    fixations.append(fix_1)
    total_dur += fix_1

    while total_dur <= T:
        transition = rng.choice(empirical_distributions['transitions'])
        fixations.append(transition)
        total_dur += transition

        fix_dur = rng.choice(empirical_distributions['fixations'][2])
        fixations.append(fix_dur)
        total_dur += fix_dur

    sacc_array = np.insert(np.cumsum(fixations), 0, 0.0)
    sacc_array = sacc_array[sacc_array < T] * dt

    # --- First fixation side ---
    mu1 = mu2 = None
    if flag: # left first
        mu1 = kappa * (r1 - eta * r2)
        mu2 = kappa * (eta * r1 - r2)
    else: # right first
        mu1 = kappa * (eta * r1 - r2)
        mu2 = kappa * (r1 - eta * r2)

    addm = aDDModel(
        mu1=mu1, mu2=mu2,
        sacc_array=sacc_array,
        flag=flag,
        sigma=sigma, a=a, b=b, x0=x0
    )

    decision = addm.simulate_fpt_datum(dt=dt)

    sacc_array = sacc_array[sacc_array < decision[0]]
    d = len(sacc_array)
    mu_array = get_alternating_addm_mu_array(mu1, mu2, d, flag)

    return decision, mu_array, sacc_array, r1, r2, flag