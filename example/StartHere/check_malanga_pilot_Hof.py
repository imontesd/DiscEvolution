"""
check_malanga_pilot_Hof.py
==========================

(Hof Co-op project, 2026-10-20) Pilot checklist for the Malanga et al. (2026) population runs
(run_popsynth_malanga_Hof.sh). Reads every hof_malanga_w0_*.h5 in a folder, together with the
population CSV in the same folder (malanga_draws_Hof_*.csv, copied there by the launcher), and
prints one row per disc:

  1. completion  -- attr complete, and last snapshot time_snap[-1] [Myr] vs the config end time
  2. runtime     -- run_end - run_start [h] (HDF5 attrs; includes set-up)
  3. storage     -- file size [MB]
  4. wind mapping: achieved f_M vs the target of the e_rad mapping
         f_M,achieved(t) = M_wind(t) / M_acc(t),   M_acc(t) = int_0^t Mdot_*(t') dt'  [Msun],
         M_wind(t) = M0 - M_disc(t) - M_acc(t)                                       [Msun]
     compared with the CSV column f_M_grid_expected = (R0 / 0.1 au)^xi - 1 (steady state; the grid
     starts at 0.1 au, so there is no wind loss inside it). Caveats: (i) Mdot_* is measured at
     R_c[0] ~ 0.1 au from the gas velocity, so dust drifting through the inner edge faster than
     the gas (d2g = 0.01) is counted as "wind" (<~1 % of M0); (ii) for discs larger than the grid,
     mass leaving through the outer boundary is also counted as "wind".
  5. accretion time: tau = M_disc / Mdot_* [Myr] at the disc's sampled age t_age (CSV), vs the
     analytic omega = 0 value 2 t_acc,0 (1 + f_M) [Myr] (Malanga Eq. 5 with omega = 0, i.e.
     M = M0 exp(-t / (2 t_acc,0)) and Mdot_* = M / (2 t_acc,0 (1 + f_M)) -- form to verify).
     A ratio < 1 means the code disc accretes faster than the analytic one; see the Step 3 flag:
     the code's initial Mdot is ~2.2x the analytic estimate because the inner disc sits at
     Tmax = 1500 K (H/r larger than Malanga Eq. 3).

Also printed: the runtime projection for the full population and the total storage estimate.
Writes the table to <folder>/pilot_check_Hof.csv.

Usage (on Nibi, with the project venv active):
    python check_malanga_pilot_Hof.py /home/imontesd/projects/def-mbalogh/imontesd/output/DiscEvolution/popsynth/malanga/pilot1

Units: HDF5 't' is in years, 'time_snap' in Myr, 'disk_Mass' in grams (/ Msun), 'disk_Mdot_star'
in Msun/yr (see CLAUDE.md). NumPy 2.x: np.trapezoid, not np.trapz.
"""

import os
import sys
import glob
import re
from datetime import datetime

import numpy as np
import pandas as pd
import h5py

from DiscEvolution.constants import Msun          # g

T_END_MYR = 3.0          # Myr, last snapshot of DiscConfig_Hof_malanga.json
T_END_RTOL = 1e-3        # relative tolerance for "reached the end"
N_FULL = 500             # discs in the full population
NPROC_FULL = 192         # discs at once in the full run (run_popsynth_malanga_Hof.sh)


def disc_row(path, draws):
    """All checklist quantities for one .h5 file (see module docstring)."""
    name = os.path.basename(path)
    disc_id = int(re.search(r"hof_malanga_w0_(\d+)_", name).group(1))
    d = draws.loc[disc_id]
    row = {"id": disc_id, "R0_au": d.R0_au, "M0_Msun": d.M0_Msun, "size_MB": os.path.getsize(path) / 1e6}

    with h5py.File(path, "r") as h:
        a = h.attrs
        row["complete"] = bool(a.get("complete", False))
        t_last = float(h["time_snap"][-1]) if ("time_snap" in h and len(h["time_snap"])) else np.nan   # Myr
        row["t_last_Myr"] = t_last
        row["reached_end"] = bool(t_last >= T_END_MYR * (1 - T_END_RTOL))
        row["runtime_h"] = np.nan
        if "run_start" in a and "run_end" in a:
            row["runtime_h"] = (datetime.fromisoformat(str(a["run_end"]))
                                - datetime.fromisoformat(str(a["run_start"]))).total_seconds() / 3600
        row["Mdot0_code"] = float(a.get("calib_Mdot", np.nan))                 # Msun / yr, initial

        if "t" not in h or len(h["t"]) < 2:
            return row
        t = h["t"][:]                                                          # yr
        M = h["disk_Mass"][:] / Msun                                           # Msun
        Mdot = h["disk_Mdot_star"][:]                                          # Msun / yr

    # cumulative accreted mass M_acc(t) [Msun], trapezoid from t = 0 (Mdot(0) ~ initial calib_Mdot)
    t0 = np.concatenate([[0.0], t]); Mdot0 = np.concatenate([[row["Mdot0_code"]], Mdot])
    M_acc = np.concatenate([[0.0], np.cumsum(0.5 * (Mdot0[1:] + Mdot0[:-1]) * np.diff(t0))])[1:]   # Msun
    M_wind = d.M0_Msun - M - M_acc                                             # Msun

    for label, t_eval_Myr in [("tage", d.t_age_Myr), ("3Myr", T_END_MYR)]:
        te = t_eval_Myr * 1e6                                                  # yr
        if te > t[-1]:
            continue
        Mt, Mdt = np.interp(te, t, M), np.interp(te, t, Mdot)
        Macc, Mw = np.interp(te, t, M_acc), np.interp(te, t, M_wind)
        row[f"fM_{label}"] = Mw / Macc if Macc > 0 else np.nan
        row[f"M_{label}_Msun"] = Mt
        row[f"Mdot_{label}"] = Mdt
        if label == "tage":
            tau = Mt / Mdt / 1e6 if Mdt > 0 else np.nan                        # Myr
            tau_an = 2 * d.t_acc0_Myr * (1 + d.f_M)                            # Myr, omega = 0
            row["tau_code_Myr"], row["tau_analytic_Myr"] = tau, tau_an
            row["tau_ratio"] = tau / tau_an
    row["fM_expected"] = d.f_M_grid_expected
    row["t_age_Myr"] = d.t_age_Myr
    return row


def main(folder):
    csvs = glob.glob(os.path.join(folder, "malanga_draws_Hof_*.csv"))
    if len(csvs) != 1:
        sys.exit(f"ERROR: expected exactly one malanga_draws_Hof_*.csv in {folder}, found {csvs}")
    draws = pd.read_csv(csvs[0]).set_index("id")
    files = sorted(glob.glob(os.path.join(folder, "hof_malanga_w0_*.h5")))
    if not files:
        sys.exit(f"ERROR: no hof_malanga_w0_*.h5 files in {folder}")

    df = pd.DataFrame([disc_row(f, draws) for f in files]).sort_values("id")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    cols = ["id", "R0_au", "M0_Msun", "complete", "t_last_Myr", "runtime_h", "size_MB",
            "fM_tage", "fM_3Myr", "fM_expected", "t_age_Myr", "tau_code_Myr", "tau_analytic_Myr", "tau_ratio"]
    print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    # ---- summary ----
    ok = df.complete & df.reached_end
    print(f"\n1. completion: {int(ok.sum())} / {len(df)} complete and reached {T_END_MYR} Myr"
          + ("" if ok.all() else f"  -> NOT OK: ids {df.loc[~ok, 'id'].tolist()}"))
    rt = df.runtime_h
    print(f"2. runtime [h]: median {rt.median():.2f}, max {rt.max():.2f} (id {int(df.loc[rt.idxmax(), 'id'])})")
    # full run: wall time >= longest disc, and >= total CPU-hours / slots. The pilot over-represents
    # the slow compact discs, so scale with the population's own share of compact discs.
    compact = draws.R0_au < 2.0                                               # au; slow, dust-limited discs
    pil_c = df.R0_au < 2.0
    est_cpu_h = (compact.sum() * rt[pil_c].median() if pil_c.any() else 0) + \
                ((~compact).sum() * rt[~pil_c].median() if (~pil_c).any() else 0)
    print(f"   full run ({N_FULL} discs, {NPROC_FULL} at once): population has {int(compact.sum())} compact "
          f"discs (R0 < 2 au); estimated wall time >= max(longest disc {rt.max():.1f} h, "
          f"{est_cpu_h:.0f} CPU-h / {NPROC_FULL} = {est_cpu_h / NPROC_FULL:.1f} h)")
    print(f"3. storage: median {df.size_MB.median():.1f} MB/disc -> ~{df.size_MB.median() * N_FULL / 1e3:.1f} GB "
          f"for {N_FULL} discs (max {df.size_MB.max():.1f} MB)")
    if "fM_tage" in df:
        r = df.fM_tage / df.fM_expected
        print(f"4. achieved f_M at t_age / expected: median {r.median():.2f}, range {r.min():.2f} - {r.max():.2f}")
    if "tau_ratio" in df:
        print(f"5. M/Mdot at t_age, code / analytic (omega = 0): median {df.tau_ratio.median():.2f}, "
              f"range {df.tau_ratio.min():.2f} - {df.tau_ratio.max():.2f}  (< 1: code accretes faster; "
              f"see the Step 3 flag, initial Mdot ~2.2x analytic)")

    out = os.path.join(folder, "pilot_check_Hof.csv")
    df.to_csv(out, index=False)
    print(f"\nTable written to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
