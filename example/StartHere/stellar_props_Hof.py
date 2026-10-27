"""
stellar_props_Hof.py
====================

(Hof Co-op project, 2026-10-26) Stellar properties for the Malanga population with a
per-disc stellar mass (sample_malanga_mstar_Hof.py).

What it provides
----------------
    star_R_Teff_Hof(Mstar)      -> R* [Rsun], Teff [K]   BHAC15 0.5 Myr isochrone, linear in log-log
    Lstar_from_R_Teff_Hof(R, T) -> L* [Lsun]             Stefan-Boltzmann, for checks / the CSV
    load_lupus_Mstar_Hof()      -> M* [Msun] of the 65 Lupus discs with measured M*
                                   (Malanga et al. 2026, Table A.1)
    truncation_radius_Konigl_Hof(M*, R*, Mdot, B*, beta)
                                -> magnetospheric truncation radius R_T [au] (Konigl 1991, Eq. 1)

(2026-10-26, later the same day) rmin_from_Rstar_Hof() (grid inner edge linear in R*) was REMOVED
at the user's request: the grid edges are no longer tied to the star. They now follow each disc's
R0 (sample_malanga_mstar_Hof.grid_edges_from_R0), which also avoids the initial Sigma underflowing
to exactly 0 (and the IrradiatedEOS solve failing) for compact discs on a fixed 250 au grid.

(2026-10-27) truncation_radius_Konigl_Hof() ADDED at the user's request: the grid inner edge is
again tied to the star, now through the PHYSICAL truncation of the disc by the stellar magnetosphere
(the disc is disrupted where the stellar field's stress removes the gas's angular momentum;
Konigl 1991, ApJ 370, L39, Sec. 2.1). The grid rule itself (floor / cap around R_T) lives in
sample_malanga_mstar_Hof.rmin_from_RT, which is where B* and beta are chosen for the population.

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

# CGS unit conversions shared with the rest of the code (DiscEvolution/constants.py):
#   Msun [g], Rsun [cm] (the "# g" comment there is a typo), AU [cm]
from DiscEvolution.constants import Msun, Rsun, AU


# ============================================================================
# Files and constants
# ============================================================================

_HERE = os.path.dirname(os.path.abspath(__file__))
BHAC15_FILE = os.path.join(_HERE, "config", "malanga", "BHAC15_iso_0.5Myr_Hof.csv")
LUPUS_MSTAR_FILE = os.path.join(_HERE, "config", "malanga", "lupus_TableA1_Mstar_Hof.csv")

BHAC15_AGE_MYR = 0.5   # Myr, age of the isochrone block used (file label "t (Gyr) = 0.0005")
TEFF_SUN = 5772.0      # K, nominal solar effective temperature (IAU 2015 Resolution B3);
                       # only used to turn (R*, Teff) into L* in Lsun

# Only for truncation_radius_Konigl_Hof (DiscEvolution/constants.py works in G = 1 code units, so
# the CGS values are defined here):
G_CGS = 6.674e-8       # cm^3 g^-1 s^-2, gravitational constant
YR_S = 3.15576e7       # s, one Julian year (turns Mdot in Msun/yr into g/s)


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
# Magnetospheric truncation radius (Hof, 2026-10-27)
# ============================================================================

def truncation_radius_Konigl_Hof(Mstar, Rstar, Mdot, B_star, beta):
    """
    Radius where the stellar dipole truncates the accretion disc [au], Konigl (1991) Eq. 1 (p. L39):

        r_in = beta * mu^(4/7) * (2 G M*)^(-1/7) * Mdot^(-2/7),    mu = B* R*^3   (Sec. 2.2, p. L40)

    Mstar  : stellar mass [Msun]
    Rstar  : stellar radius [Rsun]
    Mdot   : accretion rate through the inner disc [Msun/yr]
    B_star : dipole field at the stellar EQUATOR [G] -- Konigl's convention B* = mu/R*^3.
             (Spectropolarimetric maps usually quote the POLAR dipole field B_p = 2 mu/R*^3;
             convert with B* = B_p / 2 before calling, otherwise R_T is 2^(4/7) = 1.49x too large.)
    beta   : dimensionless geometry factor <~ 1 (beta = 1: classical Alfven radius of spherical
             free-fall accretion; Konigl/Ghosh & Lamb fiducial 0.5, p. L40)
    All arguments may be numpy arrays (broadcast). Returns R_T [au].

    Where the factor 2 comes from: (2 G M*)^(-1/7) is the Ghosh & Lamb (1979) Alfven radius, from
    B^2/(8 pi) = rho v_ff^2 / 2 with v_ff = (2 G M*/r)^(1/2). Forms without the 2 absorb
    2^(-1/7) = 0.906 into the prefactor (xi_m = 0.906 beta).

    Scaling form (same formula; 0.01845 au is the CGS value at beta = 1 and unit inputs):
        R_T = 0.01845 au * beta * (B*/1 kG)^(4/7) (R*/Rsun)^(12/7) (M*/Msun)^(-1/7)
              * (Mdot / 1e-8 Msun/yr)^(-2/7)
    Check: Konigl's fiducial star (M* = 0.8 Msun, Mdot = 1e-7 Msun/yr, R* = 2.5 Rsun, B* = 1 kG,
    beta = 0.5) gives 3.55e11 cm = 2.04 R*, vs his Eq. 4: 3.6e11 cm = 2.0 R*.
    """
    Mstar, Rstar, Mdot, B_star, beta = (np.asarray(x, dtype=float) for x in (Mstar, Rstar, Mdot, B_star, beta))
    if np.any(Mdot <= 0) or np.any(~np.isfinite(Mdot)):
        raise ValueError("Mdot must be finite and > 0 [Msun/yr]")
    mu = B_star * (Rstar * Rsun) ** 3                     # G cm^3, dipole moment
    two_GM = 2.0 * G_CGS * Mstar * Msun                   # cm^3 s^-2
    Mdot_cgs = Mdot * Msun / YR_S                         # g/s
    r_in = beta * mu ** (4 / 7) * two_GM ** (-1 / 7) * Mdot_cgs ** (-2 / 7)   # cm
    return r_in / AU                                      # au
