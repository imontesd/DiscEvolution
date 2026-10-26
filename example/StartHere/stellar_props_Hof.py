"""
stellar_props_Hof.py
====================

(Hof Co-op project, 2026-10-26) Stellar properties for the Malanga population with a
per-disc stellar mass (sample_malanga_mstar_Hof.py).

What it provides
----------------
    star_R_Teff_Hof(Mstar)      -> R* [Rsun], Teff [K]   BHAC15 0.5 Myr isochrone, linear in log-log
    Lstar_from_R_Teff_Hof(R, T) -> L* [Lsun]             Stefan-Boltzmann, for checks / the CSV
    rmin_from_Rstar_Hof(Rstar)  -> grid inner edge [au]  linear in R*, clamped to [0.03, 0.1] au
    load_lupus_Mstar_Hof()      -> M* [Msun] of the 65 Lupus discs with measured M*
                                   (Malanga et al. 2026, Table A.1)

The R* and Teff returned here are what run_model_Hof.py passes on to
DiscEvolution/star.py: SimpleStar(M=Mstar, R=Rstar, T_eff=Teff), with M in Msun,
R in Rsun and T_eff in K (star.py StarBase.__init__).

Data files (both in config/malanga/, with full source notes in their headers)
----------------------------------------------------------------------------
    BHAC15_iso_0.5Myr_Hof.csv     first isochrone ("t (Gyr) = 0.0005") of
                                  http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_iso.2mass
                                  (Baraffe et al. 2015, A&A 577, A42); 0.01-1.4 Msun, 30 rows
    lupus_TableA1_Mstar_Hof.csv   log M* of the 65 Table A.1 sources with a measured M*

Interpolation choice
--------------------
Piecewise-linear in log10 M of log10 R and log10 Teff (np.interp), chosen with the user:
    * exact at every tabulated mass;
    * no overshoot between nodes (unlike a global polynomial or a non-monotone spline);
    * keeps the real dip in R between 0.09 and 0.10 Msun (1.101 -> 1.033 Rsun; deuterium
      burning in the BHAC15 models), which a smooth fit would wash out.
Masses outside the table (0.01-1.4 Msun) raise a ValueError rather than extrapolating.
"""

import os
import csv
from functools import lru_cache

import numpy as np


# ============================================================================
# Files and constants
# ============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
BHAC15_FILE = os.path.join(_HERE, "config", "malanga", "BHAC15_iso_0.5Myr_Hof.csv")
LUPUS_MSTAR_FILE = os.path.join(_HERE, "config", "malanga", "lupus_TableA1_Mstar_Hof.csv")

BHAC15_AGE_MYR = 0.5   # Myr, age of the isochrone block used (file label "t (Gyr) = 0.0005")
TEFF_SUN = 5772.0      # K, nominal solar effective temperature (IAU 2015 Resolution B3);
                       # only used to turn (R*, Teff) into L* in Lsun

# ---- grid inner edge vs stellar radius (chosen with the user, 2026-10-26) ----
# Linear in R* through the two ENDS of the 0.5 Myr isochrone:
#     R* = 0.339 Rsun (0.01 Msun)  ->  r_min = 0.03 au   (= Malanga's R_in, Eq. 6)
#     R* = 3.601 Rsun (1.40 Msun)  ->  r_min = 0.10 au   (= the old fixed grid.rmin)
# and clamped to [0.03, 0.1] au as a guard (the clamp has no effect for 0.01-1.4 Msun).
RMIN_ANCHOR_LO = {"Rstar_Rsun": 0.339, "rmin_au": 0.03}
RMIN_ANCHOR_HI = {"Rstar_Rsun": 3.601, "rmin_au": 0.10}
RMIN_CLAMP_AU = (0.03, 0.10)   # au


# ============================================================================
# Loaders (cached: each file is read once per process)
# ============================================================================

def _read_commented_csv(path):
    """Read a CSV whose leading lines start with '#' (comments); return a list of dict rows."""
    with open(path, newline="") as f:
        lines = [ln for ln in f if not ln.lstrip().startswith("#") and ln.strip()]
    return list(csv.DictReader(lines))


@lru_cache(maxsize=None)
def load_bhac15_Hof(path=BHAC15_FILE):
    """
    BHAC15 0.5 Myr isochrone as a dict of numpy arrays:
        M_Msun [Msun], Teff_K [K], logL_Lsun [log10 Lsun], logg_cgs [log10 cm s^-2], R_Rsun [Rsun].
    Rows are in increasing mass (checked, because np.interp needs increasing x).
    """
    rows = _read_commented_csv(path)
    table = {key: np.array([float(r[key]) for r in rows])
             for key in ("M_Msun", "Teff_K", "logL_Lsun", "logg_cgs", "R_Rsun")}
    if not np.all(np.diff(table["M_Msun"]) > 0):
        raise ValueError(f"{path}: masses must be strictly increasing")
    return table


@lru_cache(maxsize=None)
def load_lupus_Mstar_Hof(path=LUPUS_MSTAR_FILE):
    """
    Measured stellar masses [Msun] of the Lupus discs in Malanga et al. (2026) Table A.1,
    in table order (65 values). Returned as a read-only numpy array.
    """
    rows = _read_commented_csv(path)
    Mstar = 10.0 ** np.array([float(r["log_Mstar"]) for r in rows])   # Msun
    Mstar.setflags(write=False)   # cached object: protect it from in-place edits
    return Mstar


# ============================================================================
# Stellar radius, effective temperature and luminosity vs mass
# ============================================================================

def star_R_Teff_Hof(Mstar):
    """
    Stellar radius and effective temperature from the BHAC15 0.5 Myr isochrone.

    Mstar : float or array, stellar mass [Msun], must lie within 0.01-1.4 Msun.
    Returns (Rstar [Rsun], Teff [K]) with the same shape as Mstar.

    Linear interpolation of log10 R and log10 Teff in log10 M (see module docstring).
    """
    tab = load_bhac15_Hof()
    M = np.asarray(Mstar, dtype=float)
    lo, hi = tab["M_Msun"][0], tab["M_Msun"][-1]   # Msun, table range
    if np.any(~np.isfinite(M)):
        raise ValueError("Mstar contains NaN or inf")
    if np.any(M < lo) or np.any(M > hi):
        raise ValueError(f"Mstar must lie within the BHAC15 table range [{lo}, {hi}] Msun; "
                         f"got min {np.nanmin(M):.4g}, max {np.nanmax(M):.4g}")
    logM = np.log10(M)
    logM_tab = np.log10(tab["M_Msun"])
    Rstar = 10.0 ** np.interp(logM, logM_tab, np.log10(tab["R_Rsun"]))   # Rsun
    Teff = 10.0 ** np.interp(logM, logM_tab, np.log10(tab["Teff_K"]))    # K
    return Rstar, Teff


def Lstar_from_R_Teff_Hof(Rstar, Teff):
    """
    Stellar luminosity [Lsun] from Stefan-Boltzmann, L/Lsun = (R/Rsun)^2 (Teff/Teff_sun)^4.
    Rstar in Rsun, Teff in K. This is the luminosity the disc's irradiation actually sees
    (IrradiatedEOS heats with sigma Teff^4 times geometric factors in R*/R; eos.py).
    """
    Rstar = np.asarray(Rstar, dtype=float)
    Teff = np.asarray(Teff, dtype=float)
    return Rstar ** 2 * (Teff / TEFF_SUN) ** 4


# ============================================================================
# Grid inner edge vs stellar radius
# ============================================================================

def rmin_from_Rstar_Hof(Rstar):
    """
    Inner edge of the radial grid [au] for a star of radius Rstar [Rsun].

    Linear in R* through (0.339 Rsun, 0.03 au) and (3.601 Rsun, 0.10 au), the two ends of the
    BHAC15 0.5 Myr isochrone, i.e.
        r_min = 0.03 au + 0.02146 au/Rsun * (R*/Rsun - 0.339),
    then clamped to [0.03, 0.10] au. Examples: 0.023 Msun -> ~0.034 au, 0.3 Msun -> ~0.070 au,
    1 Msun -> ~0.089 au.
    """
    R = np.asarray(Rstar, dtype=float)
    slope = ((RMIN_ANCHOR_HI["rmin_au"] - RMIN_ANCHOR_LO["rmin_au"])
             / (RMIN_ANCHOR_HI["Rstar_Rsun"] - RMIN_ANCHOR_LO["Rstar_Rsun"]))   # au per Rsun
    rmin = RMIN_ANCHOR_LO["rmin_au"] + slope * (R - RMIN_ANCHOR_LO["Rstar_Rsun"])   # au
    return np.clip(rmin, *RMIN_CLAMP_AU)
