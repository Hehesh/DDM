import os
import matplotlib.pyplot as plt

def _ensure_outdir(dirname: str = "mfa_figures"):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname

def _basename_from_path(path: str) -> str:
    # keep your existing slice (adjust if your filenames change)
    # example: ...YYYY-mm-dd_HH-MM-SS.csv -> take last 20 chars before ".csv"
    # You used [-24:-4]; keep that for compatibility
    return path[-24:-4] if len(path) >= 24 else os.path.splitext(os.path.basename(path))[0]

def _save_svg(fig: plt.Figure, fname: str):
    outdir = _ensure_outdir()
    fig.savefig(os.path.join(outdir, fname), format="svg", bbox_inches="tight")
    plt.close(fig)