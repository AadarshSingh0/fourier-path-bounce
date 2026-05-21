# make_fourier_csv_for_findbounce.py

import re
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar


WL_FILE = "findbounce_csvK_scan_READY.wl"
NPTS = 31
OUTDIR = Path("csv_data")


def parse_coeff_data(wl_file):
    txt = open(wl_file, "r").read()
    pattern = r'(\d+)\s*->\s*<\|"delta"\s*->\s*([0-9.Ee+-]+),\s*"c"\s*->\s*\{([^}]*)\}'
    data = {}
    for n, delta, cstr in re.findall(pattern, txt):
        n = int(n)
        delta = float(delta)
        c = np.array([float(x.strip()) for x in cstr.split(",") if x.strip()])
        data[n] = (delta, c)
    return data


def V(phi, delta, c):
    A = np.sum(c * (phi - 1.0) ** 2)
    R = np.sum(phi ** 2)
    return (A - delta) * R


def find_true_vac(delta, c):
    n = len(c)
    res = minimize(lambda x: V(x, delta, c), np.ones(n), method="BFGS",
                   options={"maxiter": 5000, "gtol": 1e-10})
    return res.x


def make_orthogonal_direction(false_vac, true_vac):
    d = true_vac - false_vac
    seed = np.ones_like(d)
    v = seed - np.dot(seed, d) / np.dot(d, d) * d
    if np.linalg.norm(v) < 1e-12:
        seed = np.arange(1, len(d) + 1, dtype=float)
        v = seed - np.dot(seed, d) / np.dot(d, d) * d
    return v / np.linalg.norm(v)


def make_fourier_path(false_vac, true_vac, amp, v, npts=NPTS):
    t = np.linspace(0.0, 1.0, npts)
    straight = (1 - t[:, None]) * false_vac + t[:, None] * true_vac
    deform = amp * np.sin(np.pi * t)[:, None] * v[None, :]
    return straight + deform


def path_objective(path, delta, c):
    vals = np.array([V(p, delta, c) for p in path])
    false_val = V(path[0], delta, c)
    barrier = np.max(vals) - false_val
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    return barrier + 0.01 * length


def optimize_one_mode_path(false_vac, true_vac, delta, c):
    v = make_orthogonal_direction(false_vac, true_vac)
    scale = np.linalg.norm(true_vac - false_vac)

    def obj(a):
        path = make_fourier_path(false_vac, true_vac, a * scale, v)
        return path_objective(path, delta, c)

    res = minimize_scalar(obj, bounds=(-0.5, 0.5), method="bounded")
    amp = res.x * scale
    path = make_fourier_path(false_vac, true_vac, amp, v)
    return path, res.fun, amp


def main():
    coeff_data = parse_coeff_data(WL_FILE)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("Parsed coefficient data for N =", sorted(coeff_data.keys())[:5], "...", sorted(coeff_data.keys())[-5:])
    print("Saving Fourier path CSV files to:", OUTDIR.resolve())

    for n in sorted(coeff_data.keys()):
        if n == 1:
            continue

        delta, c = coeff_data[n]
        false_vac = np.zeros(n)
        true_vac = find_true_vac(delta, c)

        path, obj, amp = optimize_one_mode_path(false_vac, true_vac, delta, c)

        fname = OUTDIR / f"fourier_path_n{n}.csv"
        np.savetxt(fname, path, delimiter=",", fmt="%.16e")

        print(f"N={n:2d} saved {str(fname):35s} amp={amp:.4e} obj={obj:.4e}")


if __name__ == "__main__":
    main()