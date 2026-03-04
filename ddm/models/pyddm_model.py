import numpy as np
from pyddm import gddm
from pyddm.models import OverlayChain

def create_model(drift, theta, noise, dt):
    def drift_function(x, t, avgWTP_left, avgWTP_right, fixation):
        fixation_index = min(int(t/dt), len(fixation)-1)
        current_fixation = fixation[fixation_index]
        if current_fixation == 0: # saccade
            drift_val = 0
        elif current_fixation == 1: # left
            drift_val = drift * (avgWTP_left - avgWTP_right * theta)
        else: # right
            drift_val = drift * (avgWTP_left * theta - avgWTP_right)
        
        return np.ones_like(x) * drift_val
    
    def noise_function(x, t):
        return np.ones_like(x) * noise

    # Define the model
    model = gddm(
        drift=drift_function,
        noise=noise_function,
        bound=1,
        nondecision=0,
        conditions=["avgWTP_left", "avgWTP_right", "fixation"],
        choice_names=("left", "right"),
        T_dur=30,
        dx=0.01,
        dt=dt
    )

    model._overlay = OverlayChain(overlays=[])
    return model