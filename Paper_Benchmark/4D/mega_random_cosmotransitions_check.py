#!/usr/bin/env python3
"""
CosmoTransitions check for the same mega-random potentials.

Put this file in the same folder as:
    mega_random_coefficients.csv

or edit COEFF_CSV below.

Purpose
-------
Use CosmoTransitions on the same random potential family already used in:
    mega_random_jax_fourier.py
    mega_random_findbounce_runner.wl

Potential:
    V(phi) = (sum_i c_i (phi_i - 1)^2 - delta) * sum_i phi_i^2

The potential is shifted so that V(falseVac)=0.

Run:
    source cosmo_env/bin/activate
    python3 mega_random_cosmotransitions_check.py

Output:
    mega_random_cosmotransitions_results.csv

Notes:
    - We pass tunneling_init_params={"alpha": 3} for O(4).
    - Initial path is true vacuum -> false vacuum.
    - Try nphi <= 10 for a clean comparison.
    - Try nphi = 11 or 12 only to document the known high-field limitation.
"""

import ast
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from cosmoTransitions import pathDeformation


COEFF_CSV = "mega_random_coefficients.csv"

# Use range(1, 11) for the clean referee comparison.
# Use range(1, 13) if you want to explicitly see/report the >10 behavior.
NPHI_VALUES = list(range(1, 13))

NPTS_VALUES = [80]
REPEAT = 1
MAXITER = 60

OUT_CSV = "mega_random_cosmotransitions_results.csv"


@dataclass
class CaseData:
    nphi: int
    c: np.ndarray
    delta: float
    false_vac: np.ndarray
    true_vac: np.ndarray
    Vfalse: float
    Vtrue: float
    DeltaV: float


def load_coeff_data(path=COEFF_CSV):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put this script in the same folder as "
            "mega_random_coefficients.csv or edit COEFF_CSV."
        )

    df = pd.read_csv(path)
    data = {}
    for _, row in df.iterrows():
        nphi = int(row["nphi"])
        delta = float(row["delta"])
        c = ast.literal_eval(row["c_list"])
        data[nphi] = {"delta": delta, "c": c}
    return data


def V_raw(phi, c, delta):
    phi = np.asarray(phi, dtype=float)
    A = np.sum(c * (phi - 1.0)**2, axis=-1) - delta
    B = np.sum(phi**2, axis=-1)
    return A * B


def dV_raw(phi, c, delta):
    phi = np.asarray(phi, dtype=float)
    A = np.sum(c * (phi - 1.0)**2, axis=-1) - delta
    B = np.sum(phi**2, axis=-1)

    return (
        2.0 * c * (phi - 1.0) * np.expand_dims(B, axis=-1)
        + 2.0 * phi * np.expand_dims(A, axis=-1)
    )


def make_case(nphi, coeff_data):
    d = coeff_data[nphi]
    c = np.array(d["c"], dtype=float)
    delta = float(d["delta"])
    false = np.zeros(nphi)

    def obj(x):
        return float(V_raw(x, c, delta))

    def jac(x):
        return dV_raw(np.asarray(x), c, delta)

    res = minimize(
        obj,
        np.ones(nphi),
        jac=jac,
        method="BFGS",
        options={"gtol": 1e-12, "maxiter": 5000},
    )

    true = res.x.astype(float)

    Vfalse = float(obj(false))
    Vtrue = float(obj(true))
    DeltaV = Vfalse - Vtrue

    if DeltaV <= 0:
        raise RuntimeError(f"Bad vacuum ordering for nphi={nphi}: DeltaV={DeltaV}")

    return CaseData(
        nphi=nphi,
        c=c,
        delta=delta,
        false_vac=false,
        true_vac=true,
        Vfalse=Vfalse,
        Vtrue=Vtrue,
        DeltaV=DeltaV,
    )


def shifted_functions(case):
    def V(phi):
        return V_raw(phi, case.c, case.delta) - case.Vfalse

    def dV(phi):
        return dV_raw(phi, case.c, case.delta)

    return V, dV


def run_ct(case, npts):
    V, dV = shifted_functions(case)

    # CosmoTransitions convention: initial path from true vacuum to false vacuum.
    path = np.linspace(case.true_vac, case.false_vac, npts)

    out = pathDeformation.fullTunneling(
        path,
        V,
        dV,
        maxiter=MAXITER,
        verbose=False,
        tunneling_init_params={"alpha": 3},  # O(4)
    )

    return out


def main():
    coeff_data = load_coeff_data(COEFF_CSV)
    rows = []

    print("\nCosmoTransitions check on mega-random potentials")
    print("=" * 90)
    print(f"COEFF_CSV = {COEFF_CSV}")
    print(f"NPHI_VALUES = {NPHI_VALUES}")
    print(f"NPTS_VALUES = {NPTS_VALUES}")
    print("O(4): tunneling_init_params={'alpha': 3}")
    print()

    for nphi in NPHI_VALUES:
        print("\n" + "#" * 90)
        print(f"nphi = {nphi}")

        try:
            t0 = time.perf_counter()
            case = make_case(nphi, coeff_data)
            preprocess_time = time.perf_counter() - t0

            print(f"preprocess time = {preprocess_time:.6f} s")
            print(f"Vfalse = {case.Vfalse:.12g}")
            print(f"Vtrue  = {case.Vtrue:.12g}")
            print(f"DeltaV = {case.DeltaV:.12g}")

        except Exception as err:
            msg = f"preprocess failed: {type(err).__name__}: {err}"
            print(msg)
            rows.append({
                "nphi": nphi,
                "npts": "",
                "rep": "",
                "status": "preprocess_failed",
                "S4_CT": "",
                "fRatio": "",
                "time_s": "",
                "preprocess_time_s": "",
                "Vfalse": "",
                "Vtrue": "",
                "DeltaV": "",
                "message": msg,
            })
            continue

        for npts in NPTS_VALUES:
            for rep in range(REPEAT):
                print("-" * 70)
                print(f"npts={npts}, rep={rep+1}")

                try:
                    t0 = time.perf_counter()
                    out = run_ct(case, npts=npts)
                    elapsed = time.perf_counter() - t0

                    S4 = float(out.action)
                    fRatio = float(out.fRatio)

                    print(f"S4_CT = {S4:.10g}, fRatio={fRatio:.3e}, time={elapsed:.6f} s")

                    rows.append({
                        "nphi": nphi,
                        "npts": npts,
                        "rep": rep + 1,
                        "status": "ok",
                        "S4_CT": S4,
                        "fRatio": fRatio,
                        "time_s": elapsed,
                        "preprocess_time_s": preprocess_time,
                        "Vfalse": case.Vfalse,
                        "Vtrue": case.Vtrue,
                        "DeltaV": case.DeltaV,
                        "message": "",
                    })

                except Exception as err:
                    msg = f"CosmoTransitions failed: {type(err).__name__}: {err}"
                    print(msg)

                    rows.append({
                        "nphi": nphi,
                        "npts": npts,
                        "rep": rep + 1,
                        "status": "failed",
                        "S4_CT": "",
                        "fRatio": "",
                        "time_s": "",
                        "preprocess_time_s": preprocess_time,
                        "Vfalse": case.Vfalse,
                        "Vtrue": case.Vtrue,
                        "DeltaV": case.DeltaV,
                        "message": msg,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n\nSUMMARY")
    print("=" * 90)
    cols = ["nphi", "npts", "status", "S4_CT", "fRatio", "time_s", "message"]
    print(df[cols].to_string(index=False))
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
