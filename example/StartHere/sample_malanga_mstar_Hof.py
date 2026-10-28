"""
sample_malanga_mstar_Hof.py
===========================

(Hof Co-op project, 2026-10-26) Draw a synthetic disc population from the MHD-wind parameter
distributions of Malanga et al. (2026) with a PER-DISC STELLAR MASS, and map each disc onto
the parameters DiscEvolution takes (M, Rd, psi_DW, e_rad, alpha_SS, M*, R*, Teff, r_min).

This is the per-M* counterpart of sample_malanga_Hof.py (every disc at one M*). That file is
left untouched, so the committed 1 Msun population stays reproducible; its helpers and its
Table 2 values are imported from it.

The population is drawn ONCE, from a fixed seed, and written to a CSV that is the single
record of it (run_popsynth_malanga_Hof.sh reads it row by row; the analysis notebook joins
each .h5 file back to its row by the id in run_name).

Usage
-----
    python sample_malanga_mstar_Hof.py                 # N = 500, seed = 1 -> config/malanga/
    python sample_malanga_mstar_Hof.py --N 10000 --out /tmp/test.csv     # large test sample
Writes <out>.csv and <out>.json (metadata: seed, N, sources, rules, counts, git commit, ...).

Units
-----
    M*, M0 [Msun]; R* [Rsun]; Teff [K]; L* [Lsun]; R0, r_min [au]; t_age, t_acc0 [Myr];
    Mdot0 [Msun/yr]; alpha, psi, xi, f_M, e_rad, lambda dimensionless.

The mapping, per disc i (decisions made with the user, 2026-10-26)
------------------------------------------------------------------
 1. M*_i: bootstrap of the 65 MEASURED stellar masses of Malanga Table A.1 (exact values, no
    jitter; stellar_props_Hof.load_lupus_Mstar_Hof). Survivor-biased: disc-bearing stars only.
 2. R*_i, Teff_i: BHAC15 0.5 Myr isochrone (Baraffe et al. 2015), linear in log-log
    (stellar_props_Hof.star_R_Teff_Hof). These go to SimpleStar(M=, R=, T_eff=) in the code.
 3. Malanga Eq. 7 with shift_i = log10(M*_i / 0.3 Msun):
        log10 X_i = log10 Xbar(0.3 Msun) + lambda_X shift_i + sigma_X z_i,  X = M0, R0, alpha_DW
    (Malanga Table 2 MHD column; Malanga's alpha_DW is our TOTAL alpha -- "direct" mapping).
 4. Grid edges (r_min REVISED 2026-10-27 and 2026-10-28; r_max unchanged from 2026-10-26), nr = 500:
        R_T   = Konigl (1991) Eq. 1 magnetospheric truncation radius (stellar_props_Hof.
                truncation_radius_Konigl_Hof) with B* = 0.5 kG (equatorial, SAME for every star) and
                beta = 0.5, from the disc's own M*, R* and analytic initial Mdot0 (Tabone Eq. 42, below)
        r_min = max(min(R_T, 0.1 au), 0.03 au)                          (rmin_from_RT)
        r_max = min(400 R0, 1000 au)                                    (rmax_from_R0)
    (2026-10-28, user's choice) the cap on R_T is now a FIXED 0.1 au (the old fixed grid.rmin) instead
    of 0.16 R0. For the N500 seed1 population (alpha_SS = 1e-5) this changes r_min for 2 discs only
    (ids 68 and 125, R_T = 0.164 and 0.112 au -> 0.1 au), whose disc then starts inside R_T. The cap
    no longer depends on R0, so r_min is no longer guaranteed to sit well inside compact discs:
    a disc with 0.03 au < R0 < 0.1 au and R_T > R0 would get r_min > R0 (none in N500 seed1; the
    sampler STOPS with an error if a draw produces one -- see point 5).
    Why R_T: the gas disc physically ends where the stellar magnetosphere takes over (Konigl 1991,
    Sec. 2.1); this is also the inner edge of the wind-launching region r_in in f_M =
    (r_c/r_in)^xi - 1 (Tabone et al. 2022, Eqs. 22/31; Sec. 4.4, pp. 16-17).
    Why B* = 0.5 kG constant (chosen with the user, 2026-10-27): mid-range of observed T Tauri dipoles
    (~1 kG at the pole = 0.5 kG at the equator) and of the field needed for spin equilibrium (Konigl
    Eq. 3) at these low Mdot; star-to-star scatter in B is ignored. A constant B understates the spread
    of r_min (~0.25 dex from M*, R*, Mdot vs a further ~0.17-0.29 dex for a 0.3-0.5 dex B scatter), but
    the wind loss barely depends on it: (R0/r_min)^xi with median xi ~ 0.04 changes by < 2 % for B x2.
    Why the 0.1 au cap: the old fixed grid inner edge (2026-10-28; replaced the 0.16 R0 cap of
    2026-10-27). Why the 0.03 au floor (= Malanga's R_in, Eq. 6): keeps the smallest cells (and hence
    the timestep) bounded. The floor and the cap cannot both act (0.03 au < 0.1 au).
    Why r_max: on a fixed 250 au grid the initial Sigma ~ exp(-R/R0) underflows to exactly 0
    beyond ~745 R0 and the IrradiatedEOS temperature solve fails (all discs with R0 < 0.336 au
    crashed at set-up in a test, 2026-10-26); exp(-400) ~ 1e-174 is still representable.
    Caveats: Mdot0 is the ANALYTIC (Malanga-equivalent) value; the code's initial Mdot is ~2.2x higher
    (inner disc at T_max), which would give R_T ~ 0.80x smaller (2.2^(-2/7)). The grid edge is fixed
    for the whole run, while the real R_T grows as Mdot falls (R_T ~ Mdot^(-2/7)).
 5. R0 < 0.03 au (the floor): R0 is CLIPPED to 0.03 au and flagged (flag_R0_clipped), not redrawn;
    xi, Mdot0, r_min and r_max then follow the clipped R0. (2026-10-28) With the fixed 0.1 au cap,
    r_min <= R0 is no longer guaranteed for 0.03 au < R0 < 0.1 au; it is CHECKED, and the sampler
    raises an error if any disc has r_min > R0 (none for N500 seed1). Handling such a disc (clip R0 up
    to r_min, or redraw) would need a decision: r_min depends on Mdot0, which depends on R0.
 6. psi_raw = alpha / alpha_SS - 1 (alpha_SS fixed for every disc; DEFAULT 1e-5 since 2026-10-28,
    was 1e-4). psi_raw < 0 where the drawn alpha < alpha_SS: the MEDIAN alpha of Malanga Eq. 7,
    log10 alpha = -3.4 + 2.0 log10(M*/0.3), equals alpha_SS at
        M* = 0.3 x 10^((log10 alpha_SS + 3.4) / 2.0) = 0.15 Msun (alpha_SS = 1e-4), 0.048 Msun (1e-5).
    FLOOR: psi_DW = max(psi_raw, 0), so the TOTAL alpha used is alpha_SS (1 + psi_DW) =
    max(alpha, alpha_SS). Floored discs are purely viscous (no wind): e_rad = 1, lambda = inf.
    Changing alpha_SS with the same seed keeps every disc's M*, M0, R0 and alpha (same random
    numbers); only psi, the wind parameters (xi aside), Mdot0 of floored discs and r_min follow.
 7. psi_DW > 0, wind exactly as in sample_malanga_Hof.py:
        xi    = ln(1 + f_M) / ln(R0 / R_in),  f_M = 10^-0.6, R_in = 0.03 au   (Malanga Eq. 6)
        e_rad = 1 - xi (psi + 1 + 2 xi) / (psi + 3)                              (relation D)
        lambda = 1 + psi / (2 (1 - e_rad) (3 + psi))                    (Balogh et al. 2026, Eq. 16;
                                                                          = the code's lambda_DW)
    With this e_rad the steady-state ejection index of the HYBRID disc equals xi (relation A in
    sample_malanga_Hof.py), so every disc with psi > 0 targets Malanga's f_M. For small psi this
    needs lambda -> 1, below the lambda > 3/2 of an accretion-powered wind (Tabone et al. 2022,
    Sec. 2.3, p. 3); such discs are FLAGGED (flag_lambda_lt_1p5), not changed.
 8. t_age ~ N(1.6, 0.3) Myr (Malanga Table 2) -- used ONLY by the analysis.
"""

import os
import csv
import json
import argparse
from datetime import datetime

import numpy as np

from DiscEvolution.constants import AU, Rsun   # cm; only to print R_T in stellar radii (2026-10-27)

# Helpers and Malanga Table 2 values are shared with the 1 Msun sampler (single source of truth).
from sample_malanga_Hof import (
    TABLE2, M_NORM, R_IN, T_FINAL_MYR, SIG_FIGS,
    round_sig, xi_from_fM, e_rad_from_xi_psi, lambda_code, lambda_from_xi, t_acc0_Myr,
    git_commit, _FORMATS,
)
from stellar_props_Hof import (
    load_lupus_Mstar_Hof, star_R_Teff_Hof, Lstar_from_R_Teff_Hof,
    truncation_radius_Konigl_Hof,                       # (2026-10-27) Konigl 1991 Eq. 1, for r_min
    BHAC15_FILE, LUPUS_MSTAR_FILE, BHAC15_AGE_MYR,
)


# ============================================================================
# Constants specific to this population
# ============================================================================

# ---- grid edges (chosen with the user; r_min revised 2026-10-27, 2026-10-28); see rmin_from_RT(), rmax_from_R0() ----
# (2026-10-27) The old r_min = max(min(0.16 R0, 0.1 au), 0.03 au) was REPLACED by the magnetospheric
# truncation radius R_T (Konigl 1991, Eq. 1), kept between the 0.03 au floor and a 0.16 R0 cap.
# (2026-10-28, user's choice) the 0.16 R0 cap ("rmin_frac_R0") is REPLACED by a fixed 0.1 au cap:
#     r_min = max(min(R_T, 0.1 au), 0.03 au)
GRID_RULE = {
    "rmin_B_star_G": 500.0,   # G, stellar dipole field at the EQUATOR (Konigl's B* = mu/R*^3), same for all stars
    "rmin_beta": 0.5,         # dimensionless Konigl/Ghosh & Lamb geometry factor (Konigl 1991, p. L40)
    "rmin_cap_au": 0.10,      # au; r_min = R_T, but at most 0.1 au (the old fixed grid.rmin) ...
    "rmin_floor_au": 0.03,    # ... and at least 0.03 au (Malanga's R_in, Eq. 6)
    "rmax_frac_R0": 400.0,    # r_max = 400 R0 (Sigma ~ exp(-R/R0) stays > 0 up to ~745 R0) ...
    "rmax_cap_au": 1000.0,    # ... but at most 1000 au
}
OFFGRID_FLAG = 0.05         # flag discs whose initial profile has > 5 % of its mass beyond r_max
LAMBDA_MIN_WIND = 1.5       # accretion-powered MHD wind needs lambda > 3/2 (Tabone et al. 2022, Sec. 2.3)

# Extra significant figures of the per-disc star / grid values passed to run_model_Hof.py.
# M* gets 6 (filename token uses 4); r_min and r_max get 4, the same as R0 (R0 can be clipped to r_min).
SIG_FIGS_STAR = {"Mstar": 6, "Rstar": 6, "Teff": 6, "rmin": 4, "rmax": 4}
_FORMATS_MSTAR = {**_FORMATS, "Mstar_Msun": "{:.6g}", "Rstar_Rsun": "{:.6g}", "Teff_K": "{:.6g}",
                  "rmin_au": "{:.4g}", "rmax_au": "{:.4g}", "psi_raw": "{:.4g}", "e_rad": "{:.6g}"}

# Stellar-mass bins of the printed summary [Msun]
MSTAR_BINS = [0.0, 0.1, 0.15, 0.3, 0.5, np.inf]


def rmin_from_RT(R_T):
    """
    (2026-10-28) Grid inner edge [au]: r_min = max(min(R_T, 0.1 au), 0.03 au), rounded to 4 sf.
    R_T: magnetospheric truncation radius [au] (Konigl 1991 Eq. 1).
    (2026-10-27 version: max(min(R_T, 0.16 R0), 0.03 au), with R0 as a second argument.)
    Also returns two int flags, which cannot both be 1 (0.03 au < 0.1 au):
        flag_floor = 1 where R_T < 0.03 au, so the floor set r_min;
        flag_cap   = 1 where R_T > 0.1 au, so the cap set r_min (the disc then starts INSIDE R_T).
    """
    g = GRID_RULE
    R_T = np.asarray(R_T, dtype=float)                                     # au
    rmin = np.maximum(np.minimum(R_T, g["rmin_cap_au"]), g["rmin_floor_au"])   # au
    flag_floor = (R_T < g["rmin_floor_au"]).astype(int)                    # 0.03 au floor applied
    flag_cap = (R_T > g["rmin_cap_au"]).astype(int)                        # 0.1 au cap applied
    return round_sig(rmin, SIG_FIGS_STAR["rmin"]), flag_floor, flag_cap


def rmax_from_R0(R0):
    """Grid outer edge [au]: r_max = min(400 R0, 1000 au), rounded to 4 sf. R0 in au."""
    g = GRID_RULE
    return round_sig(np.minimum(g["rmax_frac_R0"] * np.asarray(R0, dtype=float), g["rmax_cap_au"]),
                     SIG_FIGS_STAR["rmax"])


def offgrid_mass_fraction(R0, rmax):
    """
    Fraction of the analytic initial disc mass lying beyond rmax [au], for the gamma = 1 profile
    built in disc_setup_Hof, Sigma ~ (R/R0)^-1 exp(-R/R0) (no xi in the initial profile).
    Then 2 pi R Sigma ~ exp(-R/R0), so M(> rmax) / M = exp(-rmax / R0)  (for r_min << R0).
    The code normalises the mass ON the grid, so this mass is redistributed, not lost.
    """
    return np.exp(-rmax / np.asarray(R0, dtype=float))


def draw_population(N, seed, alpha_SS):
    """
    Draw N discs with per-disc M*. Returns (dict of columns, info dict).

    Random numbers (fixed order, so a seed always gives the same population):
      * z: ONE ROW of 4 standard normals per disc, (z_M0, z_R0, z_alpha, z_t), from
        default_rng(seed) -- the SAME rows as disc i of sample_malanga_Hof.py with the same seed,
        so disc i differs from its 1 Msun twin only through M* (and the floor / clip / grid).
      * M* index: from an independent child generator (SeedSequence(seed).spawn), so drawing M*
        does not shift the z rows.
    Both streams are filled element by element, so disc i is identical for ANY N (increasing N
    only appends discs). The only redraw left is for ages outside (0, T_FINAL_MYR] (> 4.7 sigma).
    """
    rng = np.random.default_rng(seed)
    rng_Mstar = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])

    # ---- 1-2: stellar mass, radius, Teff and luminosity ----
    lupus_M = load_lupus_Mstar_Hof()                                    # Msun, 65 values
    k = rng_Mstar.integers(0, lupus_M.size, size=N)                     # bootstrap index into Table A.1
    Mstar = round_sig(lupus_M[k], SIG_FIGS_STAR["Mstar"])               # Msun
    Rstar, Teff = star_R_Teff_Hof(Mstar)                                # Rsun, K
    Rstar = round_sig(Rstar, SIG_FIGS_STAR["Rstar"])
    Teff = round_sig(Teff, SIG_FIGS_STAR["Teff"])
    Lstar = Lstar_from_R_Teff_Hof(Rstar, Teff)                          # Lsun (CSV only)

    # ---- 3: Malanga Eq. 7 at each disc's own M* ----
    shift = np.log10(Mstar / M_NORM)                                    # dex, log10(M*/0.3)

    def lognormal(name, z):
        """Malanga Eq. 7 for parameter `name`, given standard-normal deviates z (one per disc)."""
        p = TABLE2[name]
        return 10 ** (p["log10_median"] + p["lambda"] * shift + p["sigma_dex"] * z)

    def age(z):
        """t_age ~ N(<t>, sigma_t) [Myr], given standard-normal deviate(s) z."""
        return TABLE2["t_age"]["mean_Myr"] + TABLE2["t_age"]["sigma_Myr"] * z

    z = rng.standard_normal((N, 4))                        # row i = disc i: (z_M0, z_R0, z_alpha, z_t)
    M0 = lognormal("M0", z[:, 0])                          # Msun
    R0_drawn = lognormal("R0", z[:, 1])                    # au, before clipping
    alpha_DW = lognormal("alpha_DW", z[:, 2])              # dimensionless, Malanga's (= our TOTAL) alpha
    t_age = age(z[:, 3])                                   # Myr
    n_t_redrawn = 0
    while np.any(bad := (t_age <= 0) | (t_age > T_FINAL_MYR)):
        n_t_redrawn += int(bad.sum())
        t_age[bad] = age(rng.standard_normal(int(bad.sum())))

    # ---- round to what run_model_Hof.py will receive; everything below uses ROUNDED values ----
    M0 = round_sig(M0, SIG_FIGS["M0"])
    R0_drawn = round_sig(R0_drawn, SIG_FIGS["R0"])

    # ---- 5: clip R0 to the 0.03 au floor of r_min (2026-10-27: done FIRST, because r_min now needs
    # Mdot0, and Mdot0 needs R0). (2026-10-28) With the fixed 0.1 au cap this no longer guarantees
    # r_min <= R0 (a disc with 0.03 au < R0 < 0.1 au can have R_T > R0); that is checked in step 4. ----
    floor_au = GRID_RULE["rmin_floor_au"]                  # au
    clipped = R0_drawn < floor_au
    R0 = np.where(clipped, floor_au, R0_drawn)             # au

    # ---- 6: psi and its floor at 0 ----
    psi_raw = round_sig(alpha_DW / alpha_SS - 1.0, SIG_FIGS["psi_DW"])   # may be negative
    psi = np.maximum(psi_raw, 0.0)                                       # psi_DW passed to the code
    floored = psi_raw < 0
    alpha_tot = alpha_SS * (1.0 + psi)                     # TOTAL alpha exactly as the code rebuilds it
    wind = psi > 0

    # ---- 7: wind mapping (psi > 0 only); floored discs: no wind ----
    f_M = 10 ** TABLE2["f_M"]["log10_value"]               # dimensionless, Malanga's target
    # xi = ln(1+f_M)/ln(R0/R_IN) (Malanga Eq. 6) needs R0 > R_IN; a disc clipped to R0 = 0.03 au
    # gets xi = NaN, which is harmless only if it has no wind
    if np.any(wind & (R0 <= R_IN)):
        raise ValueError(f"{int((wind & (R0 <= R_IN)).sum())} disc(s) with psi > 0 have R0 <= R_IN = {R_IN} au")
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where(R0 > R_IN, xi_from_fM(R0, f_M), np.nan)   # Malanga Eq. 6 (relation B)
    e_rad = np.ones(N)                                     # 1 -> no wind (code sets lambda_DW = inf)
    e_rad[wind] = round_sig(e_rad_from_xi_psi(xi[wind], psi[wind]), SIG_FIGS["e_rad"])   # relation D
    if not np.all((e_rad[wind] > 0) & (e_rad[wind] < 1)):
        raise ValueError("e_rad outside (0, 1) for some draws with psi > 0.")
    lam = np.full(N, np.inf)
    lam[wind] = lambda_code(e_rad[wind], psi[wind])        # Balogh et al. (2026) Eq. 16
    # consistency: Eq. 16 with the rounded e_rad must reproduce the lambda implied by xi (relation A)
    lam_relerr = np.abs((lam[wind] - 1) / (lambda_from_xi(xi[wind], psi[wind]) - 1) - 1)
    lam_relerr_max = float(lam_relerr.max()) if wind.any() else 0.0
    if lam_relerr_max > 1e-4:                              # e_rad rounding (6 sf) allows ~1e-5
        raise AssertionError(f"lambda round-trip error {lam_relerr_max:.2e} > 1e-4")
    flag_lam = wind & (lam < LAMBDA_MIN_WIND)

    # ---- analytic reference values (diagnostic only; NOT imposed on the runs) ----
    # t_acc,0 with Malanga's H/r (Eq. 3) and the TOTAL alpha used (floor included) -> Malanga-equivalent
    # value, NOT the code's (the code's T(R0) is 2-5x lower, so its t_acc,0 is 2-5x longer).
    t_acc = t_acc0_Myr(R0, alpha_tot, Mstar)               # Myr
    # Mdot_* at t = 0 (Tabone et al. 2022, Eq. 42): (psi + 1 + 2 xi)/(psi + 1) M0 / (2 t_acc,0 (1 + f_M));
    # floored discs: xi_eff = 0, f_M,eff = 0 (pure viscous).
    xi_eff = np.where(wind, xi, 0.0)
    fM_eff = np.where(wind, f_M, 0.0)
    Mdot0 = (psi + 1 + 2 * xi_eff) / (psi + 1) * M0 / (2 * t_acc * 1e6 * (1 + fM_eff))   # Msun / yr

    # ---- 4: grid edges (2026-10-27: r_min from the magnetospheric truncation radius) ----
    # R_T from Konigl (1991) Eq. 1 with each disc's own M* [Msun], R* [Rsun] and analytic Mdot0
    # [Msun/yr], and B* [G, equatorial] and beta the SAME for every disc (GRID_RULE)
    R_T = truncation_radius_Konigl_Hof(Mstar, Rstar, Mdot0,
                                       GRID_RULE["rmin_B_star_G"], GRID_RULE["rmin_beta"])   # au
    rmin, flag_rmin_floor, flag_rmin_cap = rmin_from_RT(R_T)     # au, 4 sf (same rounding as R0)
    # (2026-10-28) NOT guaranteed any more with the fixed 0.1 au cap (see module docstring, points 4-5):
    # stop rather than run a disc whose grid starts outside its own characteristic radius
    if np.any(bad_edge := rmin > R0):
        raise ValueError(f"{int(bad_edge.sum())} disc(s) with r_min > R0 (ids {np.flatnonzero(bad_edge).tolist()}): "
                         "decide how to treat them (clip R0 up to r_min, or redraw) before using this population")
    rmax = rmax_from_R0(R0)                                # au, from the (possibly clipped) R0

    # wind loss the run can produce inside its own grid (r_min to R0); 0 without wind
    fM_grid = np.where(wind, (R0 / rmin) ** xi_eff - 1, 0.0)
    f_offgrid = offgrid_mass_fraction(R0, rmax)

    cols = {
        "id": np.arange(N),
        "run_name": np.array([f"hof_malanga_w0_{i:04d}" for i in range(N)]),   # "w0" = winds, omega = 0
        "lupus_index_A1": k,                                # 0-based row of lupus_TableA1_Mstar_Hof.csv
        "Mstar_Msun": Mstar, "Rstar_Rsun": Rstar, "Teff_K": Teff, "Lstar_Lsun": Lstar,
        "rmin_au": rmin, "rmax_au": rmax,
        # (2026-10-27) Konigl truncation radius [au] before the floor/cap, and which limit set r_min
        # (2026-10-28) flag_rmin_cap = the fixed 0.1 au cap set r_min (column was flag_rmin_R0cap, 0.16 R0 cap)
        "R_T_au": R_T, "flag_rmin_floor": flag_rmin_floor, "flag_rmin_cap": flag_rmin_cap,
        "M0_Msun": M0, "R0_au": R0, "R0_drawn_au": R0_drawn,
        "alpha_DW_Malanga": alpha_DW, "alpha_tot": alpha_tot, "alpha_SS": np.full(N, alpha_SS),
        "psi_raw": psi_raw, "psi_DW": psi, "psi_floored": floored.astype(int),
        "xi": xi, "f_M": np.full(N, f_M), "f_M_grid_expected": fM_grid,
        "e_rad": e_rad, "lambda_DW": lam, "flag_lambda_lt_1p5": flag_lam.astype(int),
        "t_age_Myr": t_age, "t_acc0_Myr": t_acc, "Mdot0_est_Msun_yr": Mdot0,
        "flag_R0_clipped": clipped.astype(int),
        "f_mass_offgrid": f_offgrid, "flag_offgrid": (f_offgrid > OFFGRID_FLAG).astype(int),
    }
    info = {
        "n_psi_floored": int(floored.sum()), "n_R0_clipped": int(clipped.sum()),
        "n_lambda_lt_1p5": int(flag_lam.sum()), "n_offgrid_flagged": int((f_offgrid > OFFGRID_FLAG).sum()),
        "n_t_age_redrawn": n_t_redrawn, "lambda_roundtrip_max_relerr": lam_relerr_max,
        "n_rmin_floor": int(flag_rmin_floor.sum()), "n_rmin_cap": int(flag_rmin_cap.sum()),   # (2026-10-27; 0.1 au cap since 2026-10-28)
    }
    return cols, info


def write_csv_mstar(path, cols):
    """
    One row per disc. Columns passed to run_model_Hof.py are written with exactly their rounded
    significant figures (_FORMATS_MSTAR), so the launcher passes the same strings and filename
    tokens are reproducible; other floats get 8 significant figures; lambda_DW = inf is "inf".
    """
    names = list(cols)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(len(cols["id"])):
            row = []
            for key in names:
                v = cols[key][i]
                if key in _FORMATS_MSTAR:
                    row.append(_FORMATS_MSTAR[key].format(float(v)))
                elif isinstance(v, (np.floating, float)):
                    row.append(f"{float(v):.8g}")
                else:
                    row.append(str(v))
            w.writerow(row)


def print_summary(cols, info):
    """Eq. 7 check at each disc's own M*, psi per M* bin, and flag counts."""
    shift = np.log10(cols["Mstar_Msun"] / M_NORM)
    print(f"\nStellar masses: N = {cols['Mstar_Msun'].size}, median {np.median(cols['Mstar_Msun']):.3f} Msun, "
          f"range {cols['Mstar_Msun'].min():.3f}-{cols['Mstar_Msun'].max():.3f} Msun")
    print("Eq. 7 residuals, log10(X) - [median + lambda log10(M*/0.3)]  (target: mean 0, std sigma):")
    for name, col in [("M0", "M0_Msun"), ("R0", "R0_drawn_au"), ("alpha_DW", "alpha_DW_Malanga")]:
        p = TABLE2[name]
        r = np.log10(cols[col]) - (p["log10_median"] + p["lambda"] * shift)
        print(f"  {name:8s} mean {r.mean():+.3f}  std {r.std():.3f} dex (target {p['sigma_dex']:.3f})")
    t = cols["t_age_Myr"]
    print(f"  t_age    mean {t.mean():.3f} Myr (target {TABLE2['t_age']['mean_Myr']})  "
          f"std {t.std():.3f} Myr (target {TABLE2['t_age']['sigma_Myr']})")

    print("\npsi per stellar-mass bin [psi_raw: 16th / 50th / 84th percentile]:")
    print(f"  {'M* bin [Msun]':>15} {'N':>5} {'floored':>8} {'psi_raw p16/p50/p84':>24} {'lambda<1.5':>11} {'R0 clipped':>11}")
    M = cols["Mstar_Msun"]
    for lo, hi in zip(MSTAR_BINS[:-1], MSTAR_BINS[1:]):
        m = (M >= lo) & (M < hi)
        if not m.any():
            continue
        p16, p50, p84 = np.percentile(cols["psi_raw"][m], [16, 50, 84])
        print(f"  {lo:6.2f}-{hi:<8.2f} {m.sum():5d} {cols['psi_floored'][m].mean():8.0%} "
              f"{p16:8.3g}/{p50:7.3g}/{p84:7.3g} {cols['flag_lambda_lt_1p5'][m].sum():11d} "
              f"{cols['flag_R0_clipped'][m].sum():11d}")
    print("\nDerived parameters [min / median / max]:")
    for key in ["psi_DW", "xi", "e_rad", "t_acc0_Myr", "Mdot0_est_Msun_yr", "f_M_grid_expected",
                "R_T_au", "rmin_au", "rmax_au"]:
        v = cols[key]
        print(f"  {key:18s} {v.min():.4g} / {np.median(v):.4g} / {v.max():.4g}")
    lw = cols["lambda_DW"][np.isfinite(cols["lambda_DW"])]
    print(f"  {'lambda_DW (psi>0)':18s} {lw.min():.4g} / {np.median(lw):.4g} / {lw.max():.4g}")
    print(f"\nFlags: psi floored {info['n_psi_floored']}, R0 clipped {info['n_R0_clipped']}, "
          f"lambda < 1.5 {info['n_lambda_lt_1p5']}, > {OFFGRID_FLAG:.0%} mass beyond r_max "
          f"{info['n_offgrid_flagged']}, ages redrawn {info['n_t_age_redrawn']}")
    # (2026-10-27) which limit set r_min, and R_T in stellar radii (Konigl: "a few stellar radii")
    RT_Rstar = cols["R_T_au"] * AU / (cols["Rstar_Rsun"] * Rsun)          # dimensionless (au, Rsun -> cm)
    print(f"r_min: = R_T for {cols['id'].size - info['n_rmin_floor'] - info['n_rmin_cap']} discs, "
          f"0.03 au floor {info['n_rmin_floor']}, 0.1 au cap {info['n_rmin_cap']}; "
          f"R_T / R* p5/p50/p95 = {np.percentile(RT_Rstar, [5, 50, 95]).round(2)}")
    print(f"lambda round-trip max relative error: {info['lambda_roundtrip_max_relerr']:.2e}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Draw a Malanga et al. (2026) MHD-wind disc population with "
                                             "per-disc M* (see module docstring).")
    ap.add_argument("--N", type=int, default=500, help="number of discs (default 500)")
    ap.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    # (Hof, 2026-10-28) default alpha_SS lowered from 1e-4 to 1e-5 at the user's request: fewer discs hit
    # the psi floor (alpha < alpha_SS only for M* below ~0.05 Msun instead of ~0.15 Msun). The 22-disc Nibi
    # pilot (job 25134509) was run with the OLD value 1e-4; its CSV copy in the pilot output folder keeps it.
    ap.add_argument("--alpha_SS", type=float, default=1e-5, help="fixed viscous alpha (dimensionless; default 1e-5)")
    ap.add_argument("--out", type=str, default=None,
                    help="output CSV path (default config/malanga/malanga_mstar_draws_Hof_N<N>_seed<seed>.csv); "
                         "a .json with the same stem is written next to it")
    args = ap.parse_args()

    out = args.out or os.path.join(here, "config", "malanga",
                                   f"malanga_mstar_draws_Hof_N{args.N}_seed{args.seed}.csv")
    cols, info = draw_population(args.N, args.seed, args.alpha_SS)
    write_csv_mstar(out, cols)

    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": os.path.basename(__file__), "git_commit": git_commit(),
        "N": args.N, "seed": args.seed, "alpha_SS": args.alpha_SS,
        "Mstar_source": os.path.relpath(LUPUS_MSTAR_FILE, here) + " (Malanga et al. 2026 Table A.1, 65 measured M*, "
                        "bootstrap, exact values)",
        "star_source": os.path.relpath(BHAC15_FILE, here) + f" (BHAC15, {BHAC15_AGE_MYR} Myr isochrone, "
                       "linear interpolation in log-log)",
        "grid_rule": {**GRID_RULE, "formula": "r_min = max(min(R_T, 0.1 au), 0.03 au), R_T = Konigl (1991) Eq. 1 "
                                              "with B* (equatorial) and beta above, each disc's M*, R* and "
                                              "Mdot0_est (Tabone et al. 2022 Eq. 42); "
                                              "r_max = min(400 R0, 1000 au); nr from the config"},
        "psi_floor": "psi_DW = max(alpha_DW / alpha_SS - 1, 0); floored discs: e_rad = 1, lambda = inf (no wind)",
        "R0_rule": "R0 < 0.03 au (the r_min floor) is clipped to 0.03 au (flag_R0_clipped); then r_min <= R0",
        "lambda_rule": "Balogh et al. (2026) Eq. 16 with e_rad from relation D (xi from Malanga Eq. 6)",
        "TABLE2": TABLE2, "M_NORM_Msun": M_NORM, "R_IN_au": R_IN,
        "T_FINAL_Myr": T_FINAL_MYR, "SIG_FIGS": {**SIG_FIGS, **SIG_FIGS_STAR}, **info,
    }
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print_summary(cols, info)
    print(f"\nWrote {args.N} discs -> {out}\n         metadata -> {os.path.splitext(out)[0] + '.json'}")
