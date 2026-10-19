"""
sample_malanga_Hof.py
=====================

(Hof Co-op project, 2026-10-19) Draw a synthetic disc population from the MHD-wind
parameter distributions of Malanga et al. (2026) and map each disc onto the parameters
DiscEvolution actually takes (M, Rd, psi_DW, e_rad, with a fixed viscous alpha_SS).

The population is drawn ONCE, from a fixed seed, and written to a CSV that is the single
record of it:
    * run_popsynth_malanga_Hof.sh reads it row by row and launches run_model_Hof.py
      (--M, --Rd, --psi_DW, --e_rad, --alpha_SS, --run_name);
    * the analysis notebook reads it to recover each disc's sampled age and the analytic
      reference values (t_acc,0, Mdot_0, expected f_M).
No disc physics runs here: this is sampling plus the parameter mapping only.

Usage
-----
    python sample_malanga_Hof.py                      # N = 324, seed = 1, M* = 1 Msun
    python sample_malanga_Hof.py --N 10000 --seed 1 --out /tmp/test.csv   # large test sample
Writes <out>.csv and <out>.json (metadata: seed, N, Table 2 values as used, git commit, ...).

Units
-----
    M0 [Msun], R0 [au], t_age / t_acc0 [Myr], Mdot0 [Msun/yr];
    alpha_tot, alpha_SS, psi_DW, xi, f_M, e_rad, lambda_DW dimensionless.

The mapping, per disc (derivation: see the comment block "PARAMETER MAPPING" below)
------------------------------------------------------------------------------------
 1. Malanga Eq. 7:   log10 X = log10 Xbar(0.3 Msun) + lambda_X log10(M*/0.3) + sigma_X z,
                     z ~ N(0, 1), independently for X = M0, R0, alpha_DW
                     (Malanga's alpha_DW is taken as our TOTAL alpha, alpha_tot).
 2. psi_DW = alpha_tot / alpha_SS - 1,  alpha_SS = 1e-4 fixed (code: TOTAL alpha =
                     alpha_SS (1 + psi_DW), disc_setup_Hof._fixed_initial_disc, hold = 'alpha_SS').
 3. Malanga Eq. 6:   xi = ln(1 + f_M) / ln(R0 / R_in),  R_in = 0.03 au.
 4. e_rad = 1 - xi (psi + 1 + 2 xi) / (psi + 3)    (from the steady-state wind relation and the
                     code's lambda(e_rad, psi); derivation below).
 5. t_age ~ N(<t>, sigma_t) Myr  (Malanga Table 2) -- used ONLY by the analysis.
"""

import os
import sys
import csv
import json
import argparse
import subprocess
from datetime import datetime

import numpy as np


# ============================================================================
# Malanga et al. (2026), Table 2, "MHD winds" column
# ============================================================================

TABLE2 = {
    # name: log10 of the median at the normalisation mass M_NORM, M*-scaling slope lambda_X,
    #       log-normal scatter sigma_X [dex]                                    (Malanga Eq. 7)
    "M0":       {"log10_median": -2.3, "lambda": 2.0, "sigma_dex": 0.4, "units": "Msun"},
    "R0":       {"log10_median":  0.7, "lambda": 1.0, "sigma_dex": 0.6, "units": "au"},
    "alpha_DW": {"log10_median": -3.4, "lambda": 2.0, "sigma_dex": 0.1, "units": "dimensionless"},
    # wind mass-loss to stellar accretion ratio; taken as FIXED (no scatter) -- TO VERIFY
    # whether Table 2 gives it a scatter
    "f_M":      {"log10_value": -0.6, "units": "dimensionless"},
    # disc age distribution: N(mean, sigma), Myr
    "t_age":    {"mean_Myr": 1.6, "sigma_Myr": 0.3},
}
M_NORM = 0.3        # Msun, normalisation mass of Eq. 7

# ---- fixed constants of the mapping ----
R_IN = 0.03         # au, inner radius in Malanga Eq. 6 (defines xi from f_M)
R_GRID_MIN = 0.1    # au, inner edge of our grid (config grid.rmin); draws with R0 below it are redrawn
R0_EDGE_FLAG = 300  # au, R0 above this is FLAGGED (kept): the profile then reaches the 1000 au grid edge
T_FINAL_MYR = 3.0   # Myr, planned end time of every run; ages must lie in (0, T_FINAL_MYR]
H_R_1AU = 1 / 30    # Malanga Eq. 3: H/r = (1/30) (M*/Msun)^(-1/2) (r/au)^(1/4); used only for t_acc,0

# significant figures of the values passed to run_model_Hof.py (and written to the CSV)
SIG_FIGS = {"M0": 4, "R0": 4, "psi_DW": 4, "e_rad": 6}
# e_rad gets 6: d ln(lambda - 1) / d e_rad = 1 / (1 - e_rad) ~ 30 near e_rad ~ 0.97, so 3-4
# significant figures would shift lambda by up to ~1-15 %.


# ============================================================================
# PARAMETER MAPPING (derivation summary; full version in the Hof project notes)
# ============================================================================
# Code equations (DiscEvolution/viscous_evolution.py, HybridWindModel), with
# nu = alpha_SS c_s H (eos.py IrradiatedEOS._f_nu) and constant psi, lambda:
#   accretion rate  Mdot_acc = 6 pi sqrt(R) d(nu Sigma sqrt(R))/dR + 3 pi psi nu Sigma
#   wind sink       Sigma_dot_w = v_DW Sigma / (2 (lambda-1) R),  v_DW = (3/2) psi nu / R
# Steady state, dMdot_acc/dR = 2 pi R Sigma_dot_w, is an Euler-Cauchy ODE for G = nu Sigma:
#   2 R^2 G'' + (3 + psi) R G' - psi / (2 (lambda-1)) G = 0  ->  G = A R^xi (+ inner-boundary term)
#   with  xi (2 xi + 1 + psi) = psi / (2 (lambda - 1))                                  (A)
# Code lever arm (eos.py docstring; disc_setup_Hof.py lambda_DW line):
#   lambda = 1 + psi / (2 (1 - e_rad) (3 + psi))                                        (C)
# Malanga Eq. 6 (steady Mdot_acc ~ R^xi integrated from R_in to R0):
#   xi = ln(1 + f_M) / ln(R0 / R_in)                                                    (B)
# (C) into (A):   e_rad = 1 - xi (psi + 1 + 2 xi) / (psi + 3)                           (D)
# NOTE: our grid starts at R_GRID_MIN = 0.1 au > R_IN, so no wind loss happens between 0.03 and
# 0.1 au in the runs. With xi from (B) the wind loss the runs can produce out to R0 is
#   f_M,grid = (R0 / R_GRID_MIN)^xi - 1   (~0.20 at the median, not 0.25),
# which is what the achieved-f_M diagnostic should be compared with (column f_M_grid_expected).


def round_sig(x, n):
    """Round array/scalar x to n significant figures (returns float array)."""
    x = np.asarray(x, dtype=float)
    return np.array([float(f"{v:.{n - 1}e}") for v in np.ravel(x)]).reshape(x.shape)


def xi_from_fM(R0, f_M, R_in=R_IN):
    """Ejection index, Malanga Eq. 6 (relation B). R0, R_in in au; dimensionless."""
    return np.log(1.0 + f_M) / np.log(R0 / R_in)


def e_rad_from_xi_psi(xi, psi):
    """Relation (D): fraction of accretion power radiated that reproduces ejection index xi."""
    return 1.0 - xi * (psi + 1.0 + 2.0 * xi) / (psi + 3.0)


def lambda_code(e_rad, psi):
    """Lever arm exactly as the code computes it (relation C; disc_setup_Hof.py lambda_DW line)."""
    return 1.0 / (2.0 * (1.0 - e_rad) * (3.0 / psi + 1.0)) + 1.0


def lambda_from_xi(xi, psi):
    """Lever arm implied directly by (A): lambda - 1 = psi / (2 xi (psi + 1 + 2 xi))."""
    return 1.0 + psi / (2.0 * xi * (psi + 1.0 + 2.0 * xi))


def t_acc0_Myr(R0, alpha_tot, Mstar):
    """
    Analytic accretion time at R0 (reference/diagnostic only):
        t_acc,0 = 1 / (3 alpha eps(R0)^2 Omega_K(R0)),  eps = H/r from Malanga Eq. 3,
        Omega_K = 2 pi sqrt(M*/Msun) (R/au)^(-3/2)  [1/yr].
    R0 in au, Mstar in Msun -> Myr. (Tabone et al. 2022a definition -- eq. number to verify.)
    """
    eps = H_R_1AU * Mstar ** -0.5 * R0 ** 0.25                       # H/r, dimensionless
    Omega = 2 * np.pi * np.sqrt(Mstar) * R0 ** -1.5                  # 1/yr
    return 1.0 / (3.0 * alpha_tot * eps ** 2 * Omega) / 1e6          # Myr


def git_commit():
    """Current git commit of this repo (for the metadata), or 'unknown'."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def draw_population(N, seed, Mstar, alpha_SS):
    """
    Draw N discs. Returns (dict of columns, info dict with redraw counts).
    Draw order (fixed, so a seed always gives the same population): ONE ROW of 4 standard normals
    per disc, z[i] = (z_M0, z_R0, z_alpha, z_t) (Hof, 2026-10-19: changed from "all N z_M0, then
    all N z_R0, ..."). The generator fills the (N, 4) array row by row, so disc i gets the same
    parameters for ANY N with the same seed: increasing N only APPENDS new discs, and existing
    runs (same run_name, same filename) stay valid and are skipped by the launcher.
    Redraws (R0 below the grid, age outside (0, T_FINAL_MYR]) come after, from the same generator;
    they are rare (~1e-4 per disc) and are the only way a disc can depend on N.
    """
    rng = np.random.default_rng(seed)
    shift = np.log10(Mstar / M_NORM)                       # dex, log10(M*/0.3)

    def lognormal(name, z):
        """Malanga Eq. 7 for parameter `name`, given standard-normal deviate(s) z."""
        p = TABLE2[name]
        return 10 ** (p["log10_median"] + p["lambda"] * shift + p["sigma_dex"] * z)

    def age(z):
        """t_age ~ N(<t>, sigma_t) [Myr], given standard-normal deviate(s) z."""
        return TABLE2["t_age"]["mean_Myr"] + TABLE2["t_age"]["sigma_Myr"] * z

    z = rng.standard_normal((N, 4))                        # row i = disc i: (z_M0, z_R0, z_alpha, z_t)
    M0 = lognormal("M0", z[:, 0])                          # Msun
    R0 = lognormal("R0", z[:, 1])                          # au
    alpha_tot = lognormal("alpha_DW", z[:, 2])             # dimensionless (Malanga's alpha_DW = our TOTAL alpha)
    t_age = age(z[:, 3])                                   # Myr

    # ---- redraws: R0 inside the grid's inner edge (also guarantees R0 > R_IN, so xi > 0) ----
    n_R0_redrawn = 0
    while np.any(bad := R0 < R_GRID_MIN):
        n_R0_redrawn += int(bad.sum())
        R0[bad] = lognormal("R0", rng.standard_normal(int(bad.sum())))
    # ---- redraws: ages outside (0, T_FINAL_MYR] (>4.7 sigma; essentially never) ----
    n_t_redrawn = 0
    while np.any(bad := (t_age <= 0) | (t_age > T_FINAL_MYR)):
        n_t_redrawn += int(bad.sum())
        t_age[bad] = age(rng.standard_normal(int(bad.sum())))

    # ---- round to what run_model_Hof.py will receive; everything below uses ROUNDED values ----
    M0 = round_sig(M0, SIG_FIGS["M0"])
    R0 = round_sig(R0, SIG_FIGS["R0"])
    psi = round_sig(alpha_tot / alpha_SS - 1.0, SIG_FIGS["psi_DW"])
    alpha_tot = alpha_SS * (1.0 + psi)                     # TOTAL alpha exactly as the code rebuilds it
    if np.any(psi <= 0):
        raise ValueError(f"{int((psi <= 0).sum())} draw(s) give psi <= 0 (alpha_tot <= alpha_SS = {alpha_SS}); "
                         f"not possible with a positive wind -- check TABLE2 / --alpha_SS / --Mstar.")

    f_M = 10 ** TABLE2["f_M"]["log10_value"]               # dimensionless
    xi = xi_from_fM(R0, f_M)                               # (B)
    e_rad = round_sig(e_rad_from_xi_psi(xi, psi), SIG_FIGS["e_rad"])   # (D)
    if not np.all((e_rad > 0) & (e_rad < 1)):
        raise ValueError("e_rad outside (0, 1) for some draws -- check TABLE2.")

    # ---- consistency: the code's lambda(e_rad, psi) must reproduce the lambda implied by xi ----
    lam = lambda_code(e_rad, psi)
    lam_xi = lambda_from_xi(xi, psi)
    lam_relerr = np.abs((lam - 1) / (lam_xi - 1) - 1)      # relative error of (lambda - 1)
    if lam_relerr.max() > 1e-4:                            # e_rad rounding (6 sf) allows ~1e-5
        raise AssertionError(f"lambda round-trip error {lam_relerr.max():.2e} > 1e-4")

    t_acc = t_acc0_Myr(R0, alpha_tot, Mstar)               # Myr
    # Mdot_* at t = 0 for omega = 0 (Malanga Eq. 5 at t = 0; form to verify): M0 / (2 t_acc,0 (1 + f_M))
    Mdot0 = M0 / (2 * t_acc * 1e6 * (1 + f_M))             # Msun / yr, ESTIMATE only (not imposed)
    fM_grid = (R0 / R_GRID_MIN) ** xi - 1                  # wind loss possible inside our grid (see NOTE above)

    cols = {
        "id": np.arange(N),
        "run_name": np.array([f"hof_malanga_w0_{i:04d}" for i in range(N)]),   # "w0" = winds, omega = 0
        "Mstar_Msun": np.full(N, Mstar),
        "M0_Msun": M0, "R0_au": R0,
        "alpha_tot": alpha_tot, "alpha_SS": np.full(N, alpha_SS), "psi_DW": psi,
        "xi": xi, "f_M": np.full(N, f_M), "f_M_grid_expected": fM_grid,
        "e_rad": e_rad, "lambda_DW": lam,
        "t_age_Myr": t_age, "t_acc0_Myr": t_acc, "Mdot0_est_Msun_yr": Mdot0,
        "flag_R0_edge": (R0 > R0_EDGE_FLAG).astype(int),
    }
    info = {"n_R0_redrawn": n_R0_redrawn, "n_t_age_redrawn": n_t_redrawn,
            "lambda_roundtrip_max_relerr": float(lam_relerr.max())}
    return cols, info


# Columns passed to run_model_Hof.py are written with exactly their rounded significant figures,
# so the bash launcher passes the same strings and the filename tokens are reproducible.
_FORMATS = {"M0_Msun": "{:.4g}", "R0_au": "{:.4g}", "psi_DW": "{:.4g}", "e_rad": "{:.6g}",
            "alpha_SS": "{:.6g}", "Mstar_Msun": "{:.6g}"}


def write_csv(path, cols):
    names = list(cols)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(len(cols["id"])):
            row = []
            for k in names:
                v = cols[k][i]
                if k in _FORMATS:
                    row.append(_FORMATS[k].format(float(v)))
                elif isinstance(v, (np.floating, float)):
                    row.append(f"{float(v):.8g}")
                else:
                    row.append(str(v))
            w.writerow(row)


def print_summary(cols, info, Mstar):
    """Sample statistics vs the Eq. 7 targets, and ranges of the derived parameters."""
    shift = np.log10(Mstar / M_NORM)
    print(f"\nSample statistics (log10; targets from Eq. 7 at M* = {Mstar} Msun):")
    for name, col in [("M0", "M0_Msun"), ("R0", "R0_au"), ("alpha_DW", "alpha_tot")]:
        p = TABLE2[name]
        x = np.log10(cols[col])
        print(f"  {name:8s} median {np.median(x):+.3f} (target {p['log10_median'] + p['lambda'] * shift:+.3f})  "
              f"std {np.std(x):.3f} dex (target {p['sigma_dex']:.3f})")
    t = cols["t_age_Myr"]
    print(f"  t_age    mean {t.mean():.3f} Myr (target {TABLE2['t_age']['mean_Myr']})  "
          f"std {t.std():.3f} Myr (target {TABLE2['t_age']['sigma_Myr']})")
    print("Derived parameters [min / median / max]:")
    for k in ["psi_DW", "xi", "e_rad", "lambda_DW", "t_acc0_Myr", "Mdot0_est_Msun_yr", "f_M_grid_expected"]:
        v = cols[k]
        print(f"  {k:18s} {v.min():.4g} / {np.median(v):.4g} / {v.max():.4g}")
    print(f"R0 redrawn (< {R_GRID_MIN} au): {info['n_R0_redrawn']};  ages redrawn: {info['n_t_age_redrawn']};  "
          f"flagged R0 > {R0_EDGE_FLAG} au: {int(cols['flag_R0_edge'].sum())}")
    print(f"lambda round-trip max relative error: {info['lambda_roundtrip_max_relerr']:.2e}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Draw a Malanga et al. (2026) MHD-wind disc population "
                                             "for run_popsynth_malanga_Hof.sh (see module docstring).")
    ap.add_argument("--N", type=int, default=324, help="number of discs (default 324 = 2 waves of 162 cores)")
    ap.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    ap.add_argument("--Mstar", type=float, default=1.0, help="stellar mass the parameters are drawn at [Msun]")
    ap.add_argument("--alpha_SS", type=float, default=1e-4, help="fixed viscous alpha (dimensionless)")
    ap.add_argument("--out", type=str, default=None,
                    help="output CSV path (default config/popsynth/malanga_draws_Hof_N<N>_seed<seed>.csv); "
                         "a .json with the same stem is written next to it")
    args = ap.parse_args()

    out = args.out or os.path.join(here, "config", "popsynth", f"malanga_draws_Hof_N{args.N}_seed{args.seed}.csv")
    cols, info = draw_population(args.N, args.seed, args.Mstar, args.alpha_SS)
    write_csv(out, cols)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__), "git_commit": git_commit(),
        "N": args.N, "seed": args.seed, "Mstar_Msun": args.Mstar, "alpha_SS": args.alpha_SS,
        "TABLE2": TABLE2, "M_NORM_Msun": M_NORM,
        "R_IN_au": R_IN, "R_GRID_MIN_au": R_GRID_MIN, "R0_EDGE_FLAG_au": R0_EDGE_FLAG,
        "T_FINAL_Myr": T_FINAL_MYR, "SIG_FIGS": SIG_FIGS, **info,
    }
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print_summary(cols, info, args.Mstar)
    print(f"\nWrote {args.N} discs -> {out}\n         metadata -> {os.path.splitext(out)[0] + '.json'}")
