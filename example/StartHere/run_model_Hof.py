"""
run_model_HOF.py
=====================

A compact, heavily-commented walkthrough of a single DiscEvolution run,
written for someone seeing this codebase for the first time.

Modifications have been made to run_model_student.py to specify for 
Isaac Montesdeoca Hof Co-op Project Fall 2026

LOG OF CHANGES MADE (relative to run_model_student.py)
-------------------------------------------------------

-Change file name of output file to include value of erad so that it is
recoverable and can be used in calculating wind mass loss rate (Mdot_wind).
See output_filename(): the "_erad{e_rad}" token is inserted right after
"_psi{psi_DW}". Config key is "e_rad" (underscore); filename token is
"erad" (no underscore) -- Hof_figures.ipynb's load_data() regex-extracts
both from the filename.

-Added a config_log/ directory under each run's output_dir: log_config()
(new function, called once near the top of run_model(), right after the
skip-if-complete check) writes a full copy of the CLI-override-resolved
config plus a run_log.csv index, once per run that actually launches. See
the "Config logging" section and log_config()'s own docstring below for
exactly what gets written and why.

-Removed the ETA-based auto-abort: a run no longer stops itself early when
its rough ETA exceeds a threshold. The ETA is still computed and printed
every 1000 steps for progress info, but nothing acts on it now.

-Nothing else differs from run_model_student.py -- physics, config schema
(besides the erad addition above), and every other function are unchanged.
(A sibling file, run_model_Hof_dense.py, is a separate copy of THIS file
with one further addition -- see that file's own docstring.)

-Removed abort if alpha_ss is "out of range"

-Added steady_state_alpha_guess(): if config['disc']['alpha'] == 'SS', the initial
 guess for the alpha-calibration loop is computed from steady-state accretion
 (alpha c_s^2 = Mdot Omega_K / 3 pi Sigma, with T from an alpha-free energy balance)
 instead of taken from the config. setup_disc receives a deep copy of the config with
 the numeric guess; the logged config keeps 'SS'. Guess saved as HDF5 attrs
 'alpha_guess' and 'alpha_guess_mode'.

-Added a generic root-finder for the initial disc (evaluate_initial_disc(),
 _SOLVABLE_PARAMS, solve_initial_disc(), _build_grid_star_kappa()). Opt-in via a new
 optional config section, e.g.
     "calibration": {"solve_for": "psi_DW", "bracket": [0.01, 100], "rtol": 1e-6}
 It holds every parameter fixed except calibration.solve_for (one of alpha, M, Rd,
 gamma, psi_DW, e_rad) and finds, with a coarse scan + scipy brentq, the value for
 which the inner accretion rate disc.Mdot(v_r)[0] equals config['disc']['Mdot']
 (Msun/yr). The config value of the solved parameter is only the initial guess
 ('SS' allowed for alpha). If no value reaches the target, it raises with the
 reachable Mdot range. The solve runs BEFORE the output filename/skip check, and
 the rest of the run uses a config copy with the solved value written in (so the
 filename, config log, build_transport psi_DW and HDF5 attrs all see it). New HDF5
 attrs: alpha_solver, calib_solve_for, calib_guess, calib_value, calib_Mdot_target,
 calib_Mdot, calib_Mdot_relerr, calib_n_evals, calib_n_roots; alpha_guess_mode can
 now also be 'fixed'. Without a "calibration" section, behaviour is unchanged
 (disc_setup.setup_disc damped loop; alpha_solver = 'damped_iteration').


WHAT THIS RUNS
--------------
This script reproduces exactly one physical setup out of the several that
DiscEvolution supports: a viscously- and magnetically-(disc-wind)-accreting
disc, with two-population dust growth and radial drift, simple C/O
chemistry, planetesimal formation, and Bitsch-model planet growth/migration.

UNIT CONVENTIONS (the part that trips everyone up at first)
-------------------------------------------------------------
DiscEvolution works in units where G = 1, with length in AU and mass in
Msun (see DiscEvolution/constants.py). Those three choices fix the time
unit too, via Kepler's third law -- and it is *not* seconds or years, it's
whatever comes out of G = AU = Msun = 1. The constant `yr` (=2*pi) is the
conversion factor: multiply a duration in real years by `yr` to get the
equivalent duration `t` in code-time units, and divide a code-time `t` by
`yr` to get real years back. That's why you'll see a lot of `* yr` and
`/ yr` scattered through this file, and why the disc-orbital angular
frequency Omega_k comes out looking like it's "in units of 2*pi/yr".

Other quantities you'll meet below and their units:
    R, Rd, grid.Rc      radius                      AU
    Sigma               surface density             g / cm^2
    M (disc/planet)     mass                         Msun, unless stated
                                                       otherwise (planet
                                                       core/env masses are
                                                       tracked in Mearth --
                                                       see planet_formation.py)
    Mdot                accretion rate               Msun / yr
    T                   temperature                  K
    alpha, alpha_SS      viscosity parameters         dimensionless
    t (this script)      simulation time              code-time units
                                                       (divide by `yr` for
                                                       years)

PIPELINE OVERVIEW
-----------------
    1. Load the JSON config (and any --flag overrides from the CLI).
    2. Build the grid and star.
    3. Solve for the initial disc structure (disc_setup.setup_disc).
    4. Attach gas/dust transport (viscous+wind evolution, radial drift,
       turbulent diffusion) and wrap the disc in a DustGrowthTwoPop object.
    5. Seed the disc chemistry in equilibrium with the dust.
    6. Turn on planetesimal formation (optional).
    7. Place planets and attach the planet-growth model (optional).
    8. Open an HDF5 file and stream results to it as the simulation runs.
    9. Integrate forward in time, writing a snapshot at each requested
       output time.

Run it with:
    python run_model_student.py --config config/DiscConfig_default.json \\
        --psi_DW 0.01 --Mdot 1e-8 --M 0.1 --Rd 50
"""

import os
import sys
import json
import time
import csv                      # for the per-run config_log/run_log.csv index (log_config(), added for Isaac's Hof project)
from datetime import datetime   # timestamps for log_config()
import copy                     # deep-copy config so the 'SS' string in the logged config is preserved (steady-state alpha guess, Hof project)

import numpy as np
import h5py

from DiscEvolution.constants import AU, Msun, yr, Omega0, GasConst, sig_SB   # Omega0, GasConst, sig_SB added for steady_state_alpha_guess()
from DiscEvolution.disc import AccretionDisc      # Mtot() used to normalise Sigma in steady_state_alpha_guess()
from DiscEvolution.brent import brentq            # same vectorised Brent solver IrradiatedEOS uses
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.opacity import Tazzari2016, Zhu2012
from DiscEvolution.viscous_evolution import ViscousEvolutionFV, HybridWindModel
from DiscEvolution.dust import DustGrowthTwoPop, SingleFluidDrift, PlanetesimalFormation
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.planet_formation import Planets, Bitsch2015Model
from DiscEvolution.chemistry import (
    SimpleCOChemOberg, EquilibriumCOChemOberg, TimeDepCOChemOberg, SimpleCOAtomAbund,
)

from disc_setup import setup_disc
# make_eos: reused (not copied) by evaluate_initial_disc() below so the generic
# root-finder builds EXACTLY the same EOS object as disc_setup.winds_alpha_disc() (Hof project).
from disc_setup import make_eos
# SciPy's scalar Brent root-finder, used by solve_initial_disc() (Hof project).
# Imported under a different name because `brentq` above is DiscEvolution's
# VECTORISED Brent solver (used by steady_state_alpha_guess), with a different call signature.
from scipy.optimize import brentq as scipy_brentq

GAS_SOLVER = ViscousEvolutionFV   # viscous-evolution scheme used when winds are off


# ============================================================================
# Ice lines (used only for diagnostics written to the output file)
# ============================================================================

def compute_ice_lines(chem, grid, threshold=0.5):
    """
    Compute ice line radius for each molecular species.

    Parameters
    ----------
    chem : MolecularIceAbund
        Chemistry object with gas and ice abundances for each species
    grid : Grid
        Grid object with Rc radial positions (in AU)
    threshold : float
        Condensation fraction threshold (0-1) to define ice line. Default=0.5

    Returns
    -------
    ice_lines : ndarray of shape (Nspec,)
        Ice line radius (AU) for each species. NaN if no clear transition.
    """
    Nspec = chem.ice.Nspec
    ice_lines = np.full(Nspec, np.nan)

    for i, species in enumerate(chem.ice.names):
        ice_abund = chem.ice.data[i, :]
        gas_abund = chem.gas.data[i, :]
        total_abund = ice_abund + gas_abund

        if np.max(total_abund) < 1e-300:
            continue

        ice_fraction = np.divide(ice_abund, total_abund,
                                  where=total_abund > 0,
                                  out=np.zeros_like(ice_abund))

        min_frac = np.min(ice_fraction[total_abund > 0]) if np.any(total_abund > 0) else 0
        max_frac = np.max(ice_fraction[total_abund > 0]) if np.any(total_abund > 0) else 0
        if (max_frac - min_frac) < 0.1:
            continue

        crossing = np.diff(np.sign(ice_fraction - threshold))
        cross_idx = np.where(crossing != 0)[0]
        if len(cross_idx) > 0:
            idx = cross_idx[0]
            f1, f2 = ice_fraction[idx], ice_fraction[idx + 1]
            r1, r2 = grid.Rc[idx], grid.Rc[idx + 1]
            if abs(f2 - f1) > 1e-10:
                ice_lines[i] = r1 + (threshold - f1) / (f2 - f1) * (r2 - r1)
            else:
                ice_lines[i] = r1

    return ice_lines


# ============================================================================
# Steady-state initial guess for alpha (added for Isaac's Hof project)
# ============================================================================

def steady_state_alpha_guess(grid, star, config, kappa):
    """
    Steady-state estimate of the TOTAL alpha (alpha_SS + alpha_DW), used as the
    starting guess for the alpha-calibration loop in disc_setup.winds_alpha_disc()
    when config['disc']['alpha'] == 'SS'.

    Idea: in a steady disc, Mdot = 3 pi nu_SS Sigma (1 + psi), with
    nu_SS = alpha_SS c_s^2 / Omega_K and alpha = alpha_SS (1 + psi), so

        alpha * c_s^2 = Mdot * Omega_K / (3 pi Sigma)                    (*)

    Substituting (*) into the viscous heating term of IrradiatedEOS removes
    alpha from the energy balance, so T(R) can be solved from Mdot alone
    (one Brent solve, no iteration). alpha then follows from (*) with c_s(T).

    Everything below mirrors disc_setup.winds_alpha_disc (Sigma profile) and
    DiscEvolution/eos.py IrradiatedEOS.update/balance (energy balance), with
    only the viscous heating term rewritten in terms of Mdot.

    Returns
    -------
    alpha_guess : float
        Total alpha (dimensionless) at the inner cell R_c[0], which is where
        winds_alpha_disc measures Mdot.
    """
    disc_params = config['disc']
    eos_params = config['eos']
    wind_params = config['winds']

    # ---- inputs (units in comments) ----
    Mdot_target = disc_params['Mdot']            # Msun / yr
    Mdisk = disc_params['M'] * Msun              # g
    Rd = disc_params['Rd']                       # AU
    gamma = disc_params['gamma']                 # Sigma power-law index (dimensionless)
    psi = wind_params['psi_DW']                  # alpha_DW / alpha_SS (dimensionless)
    e_rad = wind_params['e_rad']                 # fraction of heating radiated (dimensionless)
    Tmax = eos_params['Tmax']                    # K, temperature cap (same as IrradiatedEOS)

    # IrradiatedEOS defaults (eos.py:315), NOT read from the config.
    # If these defaults are ever changed in make_eos/IrradiatedEOS, change them here too.
    Tc = 10.0                                    # K, external/nebular temperature floor
    mu = 2.4                                     # g / mol, mean molecular weight
    amax = 1e-5                                  # cm, grain size passed to the opacity
    tauP_over_tauR = 2.4                         # Planck/Rosseland optical-depth ratio
    dlogHdlogRm1 = 2 / 7.                        # flaring index used in f_flare

    # Mdot in cgs. Omega0 = 2 pi / (1 yr in s), so 1 yr = 2 pi / Omega0 seconds.
    sec_per_yr = 2 * np.pi / Omega0              # s / yr
    Mdot_cgs = Mdot_target * Msun / sec_per_yr   # g / s

    # ---- Sigma(R): identical to disc_setup.winds_alpha_disc ----
    R = grid.Rc                                  # AU
    Sigma = (R / Rd) ** (-gamma) * np.exp(-(R / Rd) ** (2 - gamma))
    Sigma *= Mdisk / AccretionDisc(grid, star, eos=None, Sigma=Sigma).Mtot()   # g / cm^2

    # ---- stellar/geometric factors: identical to IrradiatedEOS.update ----
    Om_k = Omega0 * star.Omega_k(R)              # s^-1
    X = star.Rau / R                             # R_star / R (dimensionless)
    f_flat = (2 / (3 * np.pi)) * X ** 3
    f_flare = 0.5 * dlogHdlogRm1 * X ** 2
    star_heat = sig_SB * star.T_eff ** 4         # erg cm^-2 s^-1
    max_heat = sig_SB * Tmax ** 4                # erg cm^-2 s^-1
    ext_heat = sig_SB * Tc ** 4                  # erg cm^-2 s^-1

    # Alpha-free viscous(+wind) heating prefactor. In IrradiatedEOS:
    #   Q_visc = e_rad (9/8) alpha_SS c_s^2 Om_k (1 + psi/3) Sigma [3/8 tau_R + 1/tau_P]
    # With alpha_SS c_s^2 = Mdot Om_k / (3 pi Sigma (1 + psi)) from (*):
    #   Q_visc = e_rad (3 / 8pi) Mdot Om_k^2 (1 + psi/3)/(1 + psi) [3/8 tau_R + 1/tau_P]
    # (= e_rad * 3 G M Mdot / (8 pi R^3) * wind factors). Units: erg cm^-2 s^-1.
    visc_prefactor = e_rad * (3 / (8 * np.pi)) * Mdot_cgs * Om_k ** 2 \
        * (1 + psi / 3) / (1 + psi)

    sqrt2pi = np.sqrt(2 * np.pi)

    def balance(Tm):
        """Net heating (erg cm^-2 s^-1) at midplane temperature Tm (K); root = T."""
        cs = np.sqrt(GasConst * Tm / mu)         # cm / s, isothermal sound speed
        H = cs / Om_k                            # cm, scale height
        kap = kappa(Sigma / (sqrt2pi * H), Tm, amax)   # cm^2 / g, at midplane density
        tauR = 0.5 * Sigma * kap                 # Rosseland optical depth
        tauP = tauP_over_tauR * tauR             # Planck optical depth

        dEdt = ext_heat
        dEdt = dEdt + star_heat * (f_flat + f_flare * (H / AU) / R)   # H/R with both in AU
        dEdt = dEdt + visc_prefactor * (3. / 8. * tauR + 1. / tauP)
        dEdt = np.minimum(dEdt, max_heat)        # temperature cap, as in IrradiatedEOS
        return dEdt - sig_SB * Tm ** 4

    # Same bracket as the first IrradiatedEOS solve: [Tc, Tmax] at every radius.
    T = brentq(balance, Tc * np.ones_like(R), Tmax * np.ones_like(R))   # K

    # ---- closed-form alpha from (*) at the inner cell ----
    cs2 = GasConst * T[0] / mu                   # cm^2 / s^2
    alpha_guess = Mdot_cgs * Om_k[0] / (3 * np.pi * Sigma[0] * cs2)     # dimensionless, TOTAL alpha
    return float(alpha_guess)


# ============================================================================
# Generic root-finder for the initial disc
# ============================================================================
#
# disc_setup.winds_alpha_disc() can only solve for alpha, and does so with 100
# damped fixed-point updates. The functions below generalise that to "hold
# every parameter fixed except ONE, and root-find that one so the disc
# reproduces the target accretion rate":
#
#     g(x) = log10( Mdot_inner(p) / Mdot_target ) = 0
#
#   p            : the full parameter set (alpha, M, Rd, gamma, psi_DW, e_rad)
#   x            : the free parameter, as log10(value) for 'log'-scaled
#                  parameters or the value itself for 'linear' ones
#   Mdot_inner   : Msun/yr, measured exactly like winds_alpha_disc does it,
#                  i.e. disc.Mdot(v_r)[0] at the inner cell R_c[0]
#   Mdot_target  : Msun/yr, config['disc']['Mdot']
#
# Using log10 of the RATIO (not the difference) keeps g O(1) whatever the
# target Mdot is, so the same absolute tolerances work for every run.
#
# Turned on by adding a "calibration" section to the config, e.g.
#     "calibration": {"solve_for": "psi_DW", "bracket": [0.01, 100]}
# With no "calibration" section, run_model() behaves exactly as before
# (disc_setup.setup_disc damped loop).

# Registry of parameters the solver can solve for.
#   solve_for name -> (config section, config key, search scale, default search range)
# Search ranges are in the parameter's own units (see comments). A range can be
# overridden per run with config['calibration']['bracket'] = [lo, hi].
# To make another parameter solvable: add a row here AND make sure
# evaluate_initial_disc() actually uses p[<name>].
_SOLVABLE_PARAMS = {
    #  name       section   key        scale     default [lo, hi]
    "alpha":  ("disc",  "alpha",  "log",    (1e-6, 1.0)),     # TOTAL alpha = alpha_SS (1 + psi), dimensionless
    "M":      ("disc",  "M",      "log",    (1e-5, 1.0)),     # disc mass, Msun
    "Rd":     ("disc",  "Rd",     "log",    (1.0, None)),     # characteristic radius, AU (None -> outermost cell centre grid.Rc[-1])
    "gamma":  ("disc",  "gamma",  "linear", (0.0, 1.9)),      # Sigma power-law index, dimensionless (must stay < 2)
    "psi_DW": ("winds", "psi_DW", "log",    (1e-3, 1e3)),     # alpha_DW / alpha_SS, dimensionless (psi = 0 not reachable in log space)
    "e_rad":  ("winds", "e_rad",  "linear", (0.0, 0.999)),    # fraction of accretion heating radiated, dimensionless (must stay < 1)
}


def evaluate_initial_disc(grid, star, p, eos_params, kappa):
    """
    Forward model: build the initial gas disc for ONE fixed parameter set and
    measure its inner accretion rate. This is exactly one pass of the loop
    body in disc_setup.winds_alpha_disc() (same Sigma profile, same
    normalisation, same lambda_DW formula, same EOS via make_eos, same
    HybridWindModel velocity), with no alpha update afterwards.

    Parameters
    ----------
    grid, star, kappa : as in run_model()
    p : dict with keys
        'alpha'  -- TOTAL alpha (alpha_SS + alpha_DW), dimensionless
        'M'      -- disc mass, Msun
        'Rd'     -- characteristic radius, AU
        'gamma'  -- Sigma power-law index, dimensionless
        'psi_DW' -- alpha_DW / alpha_SS, dimensionless
        'e_rad'  -- fraction of accretion heating radiated, dimensionless
    eos_params : config['eos']

    Returns
    -------
    dict with
        'disc'      -- AccretionDisc built with this EOS and Sigma
        'eos'       -- the EOS object (T, c_s, H at this alpha_SS)
        'Sigma'     -- g / cm^2, normalised to p['M']
        'alpha_SS'  -- viscous alpha = alpha / (1 + psi), dimensionless
        'lambda_DW' -- wind lever-arm parameter, dimensionless (inf if no wind)
        'Mdot'      -- Msun / yr, accretion rate at the inner cell R_c[0]
    """
    alpha = p['alpha']            # total alpha, dimensionless
    Mdisk = p['M'] * Msun         # g
    Rd = p['Rd']                  # AU
    gamma = p['gamma']            # dimensionless
    psi = p['psi_DW']             # dimensionless
    e_rad = p['e_rad']            # dimensionless

    # Wind lever arm: identical to disc_setup.winds_alpha_disc (psi -> 0 gives inf;
    # it then multiplies a wind velocity that is already exactly zero).
    if psi > 0 and e_rad < 1.0:
        lambda_DW = 1 / (2 * (1 - e_rad) * (3 / psi + 1)) + 1
    else:
        lambda_DW = np.inf
    alpha_SS = alpha / (1 + psi)  # viscous-only alpha, what the EOS heating uses

    # Sigma(R), g / cm^2: same self-similar profile + mass normalisation as winds_alpha_disc
    R = grid.Rc                   # AU
    Sigma = (R / Rd) ** (-gamma) * np.exp(-(R / Rd) ** (2 - gamma))
    Sigma *= Mdisk / AccretionDisc(grid, star, eos=None, Sigma=Sigma).Mtot()

    # EOS (temperature structure) for this alpha_SS / psi / e_rad
    eos = make_eos(eos_params, star, alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad)
    eos.set_grid(grid)
    eos.update(0, Sigma)

    # Inner accretion rate, Msun / yr (AccretionDisc.Mdot already converts to Msun/yr)
    disc = AccretionDisc(grid, star, eos, Sigma)
    gas = HybridWindModel(psi, lambda_DW)
    v_r = gas.viscous_velocity(disc, Sigma)
    Mdot = disc.Mdot(v_r)[0]

    return {'disc': disc, 'eos': eos, 'Sigma': Sigma, 'alpha_SS': alpha_SS,
            'lambda_DW': lambda_DW, 'Mdot': float(Mdot)}


def solve_initial_disc(grid, star, config, kappa):
    """
    Root-find ONE disc parameter (config['calibration']['solve_for']) so that
    the initial disc's inner accretion rate equals config['disc']['Mdot'],
    holding every other parameter at its config value.

    config['calibration'] keys
    --------------------------
    solve_for : str, required   -- one of _SOLVABLE_PARAMS ('alpha', 'M', 'Rd',
                                   'gamma', 'psi_DW', 'e_rad')
    bracket   : [lo, hi], opt.  -- search range in the parameter's own units
                                   (default: the range in _SOLVABLE_PARAMS)
    rtol      : float, opt.     -- max allowed |Mdot/Mdot_target - 1| (default 1e-6)
    n_scan    : int, opt.       -- number of points in the coarse scan (default 31)

    The config value of the solved parameter is used ONLY as the initial
    guess (to pick a root if there are several). For 'alpha' it may be 'SS'
    (steady_state_alpha_guess). Every other parameter must be numeric.

    Method
    ------
    1. Coarse scan: evaluate g(x) at n_scan points evenly spaced in x across
       [lo, hi] (x = log10(value) for 'log' parameters). Mdot can depend only
       weakly on psi_DW / e_rad / Rd, so a target may be unreachable; if g
       never changes sign we raise with the reachable Mdot range instead of
       returning something meaningless.
    2. Pick the sign-change interval closest to the guess (warn if >1).
    3. scipy.optimize.brentq inside that interval.
    4. Rebuild the disc AT the root (so returned disc/eos/alpha are mutually
       consistent) and check |Mdot/Mdot_target - 1| < rtol.

    Returns
    -------
    dict with
        'disc', 'eos', 'Sigma', 'alpha', 'alpha_SS', 'lambda_DW'
                      -- same meaning/units as setup_disc()'s return tuple
                         (alpha is TOTAL alpha, dimensionless)
        'run_config'  -- deep copy of config with the solved value written in
                         (and 'SS' replaced by the numeric alpha) plus a
                         'calibration_result' block; use it for the rest of the run
        'info'        -- dict summarising the solve (also stored in run_config)
    """
    calib = config['calibration']
    name = calib['solve_for']
    if name not in _SOLVABLE_PARAMS:
        raise ValueError(f"calibration.solve_for must be one of {list(_SOLVABLE_PARAMS)}, got {name!r}")
    section, key, scale, (lo, hi) = _SOLVABLE_PARAMS[name]
    if name == "Rd" and hi is None:
        hi = float(grid.Rc[-1])                                  # AU, outermost cell centre
    if calib.get('bracket') is not None:
        lo, hi = (float(v) for v in calib['bracket'])            # parameter's own units
    rtol = float(calib.get('rtol', 1e-6))                        # dimensionless, on Mdot
    n_scan = int(calib.get('n_scan', 31))
    Mdot_target = config['disc']['Mdot']                         # Msun / yr

    # ---- full parameter set at the config values (units as in evaluate_initial_disc) ----
    p = {
        'alpha':  config['disc']['alpha'],    # total alpha (may be 'SS' only if solving for alpha)
        'M':      config['disc']['M'],        # Msun
        'Rd':     config['disc']['Rd'],       # AU
        'gamma':  config['disc']['gamma'],    # dimensionless
        'psi_DW': config['winds']['psi_DW'],  # dimensionless
        'e_rad':  config['winds']['e_rad'],   # dimensionless
    }

    # ---- initial guess for the free parameter (used only to choose among roots) ----
    guess = p[name]
    guess_mode = 'config'
    if name == 'alpha' and guess == 'SS':
        guess = steady_state_alpha_guess(grid, star, config, kappa)   # total alpha, dimensionless
        guess_mode = 'SS'
    for k, v in p.items():
        if k != name and isinstance(v, str):
            raise ValueError(f"Fixed parameter {k!r} must be numeric when solving for {name!r} "
                             f"(got {v!r}); 'SS' is only allowed for alpha when solve_for = 'alpha'.")
    guess = float(guess)

    # ---- map parameter value <-> search variable x ----
    if scale == 'log':
        if lo <= 0:
            raise ValueError(f"bracket for log-scaled {name!r} must be > 0, got {[lo, hi]}")
        to_x, from_x = np.log10, (lambda x: 10.0 ** x)
    else:
        to_x, from_x = (lambda v: v), (lambda x: x)

    n_evals = [0]   # mutable counter of forward-model evaluations (list so g() can modify it)

    def g(x):
        """log10(Mdot / Mdot_target) with the free parameter set to from_x(x); nan if Mdot <= 0."""
        q = dict(p)
        q[name] = float(from_x(x))
        n_evals[0] += 1
        Mdot = evaluate_initial_disc(grid, star, q, config['eos'], kappa)['Mdot']   # Msun / yr
        if not np.isfinite(Mdot) or Mdot <= 0:
            return np.nan
        return np.log10(Mdot / Mdot_target)

    def g_safe(x):
        """g(x), but nan instead of an exception. At extreme parameter values (e.g. very
        small Rd -> huge Sigma at R_in) the EOS's own temperature solve can fail to
        converge; during the scan such points are just treated as unusable."""
        try:
            return g(x)
        except (RuntimeError, ValueError, FloatingPointError, ZeroDivisionError):
            return np.nan

    # ---- 1. coarse scan ----
    # np.errstate silences the numpy overflow/divide warnings the EOS emits at extreme
    # parameter values during the scan (those points just come back nan / far from 0).
    xs = np.linspace(to_x(lo), to_x(hi), n_scan)
    with np.errstate(all='ignore'):
        gs = np.array([g_safe(x) for x in xs])
    ok = np.isfinite(gs)
    if not ok.any():
        raise RuntimeError(f"calibration: Mdot was non-positive/non-finite for every {name} in [{lo:.4g}, {hi:.4g}]")
    # Mdot insensitive to the parameter (g flat to rounding): no meaningful root exists.
    # This happens e.g. for e_rad when the inner cells sit at the Tmax cap (T = Tmax there
    # whatever the heating), since Mdot is measured at the inner cell R_c[0].
    if np.nanmax(gs) - np.nanmin(gs) < 1e-10:
        raise RuntimeError(
            f"calibration: Mdot at R_c[0] does not depend on {name} over [{lo:.4g}, {hi:.4g}] "
            f"(Mdot = {Mdot_target * 10 ** np.nanmean(gs):.4e} Msun/yr everywhere; e.g. inner "
            f"cells at the Tmax = {config['eos'].get('Tmax')} K cap make Mdot independent of e_rad). "
            f"Solve for a different parameter."
        )
    # Candidate roots:
    #   - scan points where g is exactly 0 (the scan hit the root itself), and
    #   - intervals [xs[i], xs[i+1]] where g strictly changes sign.
    # (Kept separate so an exact zero on a scan point is not double-counted as two
    #  adjacent sign changes.)
    # "exactly 0" is |g| < 1e-12 in log10(Mdot ratio), i.e. Mdot equal to ~2e-12 relative.
    exact = [i for i in range(n_scan) if ok[i] and abs(gs[i]) < 1e-12]
    # Several scan points ALL reproducing the target means Mdot is flat (plateau) in this
    # parameter around the target: every value there works, so the solve is not meaningful.
    # (e.g. e_rad with the inner cells at the Tmax cap: g = 0 for every e_rad > 0.)
    if len(exact) > 1:
        raise RuntimeError(
            f"calibration: {len(exact)} scan values of {name} between {from_x(xs[exact[0]]):.4g} and "
            f"{from_x(xs[exact[-1]]):.4g} ALL give Mdot = {Mdot_target:.3e} Msun/yr: Mdot at R_c[0] is "
            f"insensitive to {name} here (e.g. inner cells at the Tmax = {config['eos'].get('Tmax')} K cap), "
            f"so {name} is not constrained by Mdot. Solve for a different parameter."
        )
    idx = [i for i in range(n_scan - 1) if ok[i] and ok[i + 1] and gs[i] * gs[i + 1] < 0]
    if not idx and not exact:
        Mdot_min = Mdot_target * 10 ** np.nanmin(gs)            # Msun / yr
        Mdot_max = Mdot_target * 10 ** np.nanmax(gs)            # Msun / yr
        raise RuntimeError(
            f"calibration: no {name} in [{lo:.4g}, {hi:.4g}] gives Mdot = {Mdot_target:.3e} Msun/yr "
            f"with the other parameters fixed. Reachable range: [{Mdot_min:.3e}, {Mdot_max:.3e}] Msun/yr. "
            f"Widen calibration.bracket, or change another parameter."
        )

    # ---- 2. choose the candidate root closest to the guess ----
    # Each candidate: (approximate x location, 'exact' scan point index or 'interval' index)
    x_guess = to_x(guess) if (scale == 'linear' or guess > 0) else 0.5 * (xs[0] + xs[-1])
    cands = [(xs[j], 'exact', j) for j in exact] + \
            [(0.5 * (xs[j] + xs[j + 1]), 'interval', j) for j in idx]
    cands.sort(key=lambda c: c[0])
    x_c, kind, j = min(cands, key=lambda c: abs(c[0] - x_guess))
    n_roots = len(cands)
    if n_roots > 1:
        approx = ", ".join(f"{from_x(c[0]):.4g}" for c in cands)
        print(f"WARNING calibration: Mdot is not monotonic in {name}; {n_roots} roots near "
              f"[{approx}]. Using the one closest to the guess {guess:.4g}.")

    # ---- 3. Brent's method inside that interval (x tolerance is in log10 or linear units) ----
    if kind == 'exact':
        x_root = xs[j]                                           # scan landed exactly on the root
    else:
        with np.errstate(all='ignore'):
            x_root = scipy_brentq(g, xs[j], xs[j + 1], xtol=1e-12, maxiter=200)
    value = float(from_x(x_root))                                # parameter's own units

    # ---- 4. rebuild the disc at the root and check the residual ----
    q = dict(p)
    q[name] = value
    res = evaluate_initial_disc(grid, star, q, config['eos'], kappa)
    relerr = abs(res['Mdot'] / Mdot_target - 1)                  # dimensionless
    if relerr > rtol:
        raise RuntimeError(f"calibration: Brent converged to {name} = {value:.6g} but "
                           f"|Mdot/Mdot_target - 1| = {relerr:.2e} > rtol = {rtol:.1e} "
                           f"(Mdot may be discontinuous in {name} here).")

    info = {
        'solve_for': name,
        'guess': guess,                         # parameter's own units
        'guess_mode': guess_mode,               # 'SS' or 'config'
        'value': value,                         # solved value, parameter's own units
        'Mdot_target': float(Mdot_target),      # Msun / yr
        'Mdot': res['Mdot'],                    # Msun / yr, achieved
        'Mdot_relerr': float(relerr),           # dimensionless
        'n_evals': int(n_evals[0] + 1),         # forward-model evaluations (scan + Brent + final rebuild)
        'n_roots': n_roots,                     # number of candidate roots found in the scan (>1: non-monotonic)
    }
    print(f"Calibration: {name} = {value:.6g} (guess {guess:.4g}) -> Mdot = {res['Mdot']:.4e} Msun/yr "
          f"(rel. err {relerr:.1e}, {info['n_evals']} evaluations)")

    # ---- config copy for the rest of the run, with the solved value written in ----
    run_config = copy.deepcopy(config)
    run_config[section][key] = value
    run_config['disc']['alpha'] = float(q['alpha'])   # numeric total alpha (replaces 'SS' if it was used)
    run_config['calibration_result'] = info           # ends up in the logged config json

    return {'disc': res['disc'], 'eos': res['eos'], 'Sigma': res['Sigma'],
            'alpha': float(q['alpha']), 'alpha_SS': res['alpha_SS'],
            'lambda_DW': res['lambda_DW'], 'run_config': run_config, 'info': info}


# ============================================================================
# Step 2: time grid
# ============================================================================

def make_time_grid(sim_params):
    """
    Build the array of simulation times (in code-time units) at which the
    disc state will be checkpointed to the output file.

    sim_params['t_interval'] can be given three ways:
        "power"   -- log-spaced snapshots from t_initial to t_final (years)
        a list    -- explicit snapshot times, in Myr
        a number  -- fixed linear spacing, in years, from t_initial to t_final
    """
    t_interval = sim_params['t_interval']

    if t_interval == "power":
        if sim_params['t_initial'] == 0:
            num_points = int(np.log10(sim_params['t_final'])) + 1
            years = np.logspace(0, np.log10(sim_params['t_final']), num=num_points)
        else:
            num_points = int(np.log10(sim_params['t_final'] / sim_params['t_initial'])) + 1
            years = np.logspace(np.log10(sim_params['t_initial']),
                                 np.log10(sim_params['t_final']), num=num_points)
        return years * yr

    elif isinstance(t_interval, list):
        Myr = np.array(t_interval)
        return Myr * 1e6 * yr

    else:
        years = np.arange(sim_params['t_initial'], sim_params['t_final'], t_interval)
        return years * yr


# ============================================================================
# Step 4: gas/dust transport + dust-growth disc wrapper
# ============================================================================

def build_transport(transport_params, wind_params, disc_params, dust_growth_params, lambda_DW):
    """
    Build the operators that move gas and dust around the disc each
    timestep. Any of the three can be turned off independently via
    `transport_params` (e.g. to hold the gas disc fixed while testing
    chemistry).
    """
    gas = None
    if transport_params['gas_transport']:
        if wind_params["on"]:
            gas = HybridWindModel(wind_params['psi_DW'], lambda_DW, boundary = "Mdot_inn")
        else:
            gas = GAS_SOLVER()

    diffuse = None
    if transport_params['diffusion']:
        diffuse = TracerDiffusion(Sc=disc_params["Sc"])

    dust = None
    if transport_params['radial_drift']:
        # SingleFluidDrift does its own internal diffusion call when given
        # a `diffusion` object, so hand ours off and stop calling it
        # separately further down.
        dust = SingleFluidDrift(diffusion=diffuse,
                                 settling=dust_growth_params['settling'],
                                 van_leer=transport_params['van_leer'])
        diffuse = None

    return gas, dust, diffuse


def build_dust_growth_disc(grid, star, eos, Sigma, disc_params, dust_growth_params, gas):
    """Wrap the bare (grid, star, eos, Sigma) disc in dust-growth physics."""
    return DustGrowthTwoPop(
        grid, star, eos, disc_params['d2g'], Sigma=Sigma,
        feedback=dust_growth_params["feedback"], Sc=disc_params["Sc"],
        f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'],
        uf_0=dust_growth_params["uf_0"], uf_ice=dust_growth_params["uf_ice"], gas=gas,
    )


# ============================================================================
# Step 5: chemistry
# ============================================================================

_CHEM_MODELS = {
    "Simple": lambda: SimpleCOChemOberg(),
    "Equilibrium": lambda: EquilibriumCOChemOberg(a=1e-5),
    "Equilibrium_Fixed": lambda: EquilibriumCOChemOberg(a=1e-5, fix_ratios=True),
    "TimeDep": lambda: TimeDepCOChemOberg(a=1e-5),
}


def build_chemistry(disc, chemistry_params, d2g_target, N_cell):
    """
    Seed the disc with ice/gas-phase chemical abundances in equilibrium
    with the initial dust-to-gas ratio, iterating a few times because the
    ice fraction and the dust-to-gas ratio depend on each other.

    Returns (chemistry_model, Natom, Nmol). Also sets disc.chem and
    initializes disc.dust_frac from the ice abundances.
    """
    if not chemistry_params["on"]:
        disc.chem = None
        return None, 1, 1  # dummy dimensions when chemistry is off

    try:
        chemistry = _CHEM_MODELS[chemistry_params["chem_model"]]()
    except KeyError:
        raise ValueError("Valid chemistry model not selected. "
                          "Choose Simple, Equilibrium, Equilibrium_Fixed, or TimeDep")

    # Solar reference abundances (number of atoms per H), same for every cell.
    X_solar = SimpleCOAtomAbund(N_cell)
    X_solar.set_solar_abundances()

    # The dust-to-gas ratio depends on how much ice has condensed, and the
    # equilibrium ice fraction depends on the dust-to-gas ratio (it sets the
    # dust surface area available for condensation) -- so iterate.
    chem = None
    for _ in range(100):
        if chemistry_params["assert_d2g"]:
            # Rescale the dust fraction so the *total* dust-to-gas ratio
            # matches disc_params['d2g'] exactly, rather than whatever the
            # ice chemistry alone would produce.
            M_dust = np.trapz(disc.Sigma_D.sum(0), np.pi * disc.grid.Rc ** 2)
            M_gas = np.trapz(disc.Sigma_G, np.pi * disc.grid.Rc ** 2)
            mod_frac = d2g_target / (M_dust / M_gas)
            disc.dust_frac[:] = disc.dust_frac * mod_frac

        dust_frac = disc.dust_frac.sum(0)
        chem = chemistry.equilibrium_chem(disc.T, disc.midplane_gas_density, dust_frac, X_solar)
        disc.initialize_dust_density(chem.ice.total_abund)

    disc.chem = chem
    disc.update_ices(disc.chem.ice)

    Natom = disc.chem.ice.atomic_abundance().data.shape[0]
    Nmol = disc.chem.gas.data.shape[0]
    return chemistry, Natom, Nmol


# ============================================================================
# Step 6: planetesimals
# ============================================================================

def build_planetesimals(disc, planetesimal_params):
    """Attach a PlanetesimalFormation object to the disc, if enabled."""
    disc._planetesimal = None
    if planetesimal_params['active']:
        disc._planetesimal = PlanetesimalFormation(
            disc,
            d_planetesimal=planetesimal_params['diameter'],
            St_min=planetesimal_params['St_min'],
            St_max=planetesimal_params['St_max'],
            pla_eff=planetesimal_params['pla_eff'],
        )


# ============================================================================
# Step 7: planets
# ============================================================================

def build_planets(disc, planet_params, chemistry_params, wind_params):
    """
    Create the Planets container and the Bitsch2015Model that grows/
    migrates them, and drop each planet in at its starting radius and mass.

    Planets start with a bare core and no envelope (X_env = 0); their core
    composition is read off the disc's ice abundance at the planet's
    starting radius.
    """
    if not planet_params['include_planets']:
        return None, None

    if chemistry_params["on"]:
        Nchem = disc.chem.ice.data.shape[0]
        planets = Planets(Nchem=Nchem)
    else:
        planets = Planets(Nchem=0)

    planet_model = Bitsch2015Model(
        disc, pb_gas_f=planet_params["pb_gas_f"],
        migrate=planet_params["migrate"],
        pebble_acc=planet_params["pebble_accretion"],
        gas_acc=planet_params["gas_accretion"],
        planetesimal_acc=planet_params["planetesimal_accretion"],
        winds=wind_params["on"],
    )
    planet_model.set_disc(disc)

    for Rp_i, Mp_i, t_implant in zip(planet_params['Rp'], planet_params['Mp'],
                                      planet_params['implant_time']):
        if chemistry_params["on"]:
            X_core = np.array([
                disc.interp(Rp_i, ice_spec) / disc.interp(Rp_i, disc.dust_frac[:2].sum(0))
                for ice_spec in disc.chem.ice.data
            ])
            X_env = np.zeros_like(X_core)
            planets.add_planet(t_implant, Rp_i, Mp_i, 0, X_core, X_env)
        else:
            planets.add_planet(t_implant, Rp_i, Mp_i, 0)

    return planets, planet_model


# ============================================================================
# HDF5 streaming output
# ============================================================================
#
# Every quantity below is stored as a "growable" dataset: created with an
# initial length of 0 along axis 0, then extended by one row per snapshot
# with `grow_and_set`. This keeps the file readable by
# run_model_stream.load_visc_data() (and the analysis notebooks that use
# it) whether the run is 10 steps or 10 million steps long, and without
# needing to know the final length in advance.
#
# Keep the dataset/group *names* exactly as they are here -- the loader
# and existing notebooks key off them by name.

def grow_and_set(dset, value):
    """Append one row to a growable HDF5 dataset."""
    n = dset.shape[0]
    dset.resize(n + 1, axis=0)
    dset[n] = value


def create_output_file(outfile, grid, config, Natom, Nmol, alpha_SS):
    """
    Create the HDF5 file and every dataset/group it will need, but do not
    write any data yet (see write_planet_row / write_disc_snapshot below).
    Returns the open h5py.File plus a dict of the per-planet group handles.
    """
    planet_params = config['planets']
    chemistry_params = config['chemistry']
    planetesimal_params = config['planetesimal']
    nR = len(grid.Rc)

    h5f = h5py.File(outfile, "w")
    h5f.attrs["alpha_SS"] = float(alpha_SS)

    # ---- scalar time series (one number per snapshot) ----
    for name in ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]:
        h5f.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")

    # ---- per-planet time series ----
    groups = {}
    if planet_params['include_planets']:
        grp_Mcs = h5f.create_group("Mcs")
        grp_Mes = h5f.create_group("Mes")
        grp_Rp = h5f.create_group("Rp")
        grp_Mdotp = h5f.create_group("disk_Mdot_p")
        grp_Xc = h5f.create_group("X_cores")
        grp_Xe = h5f.create_group("X_envs")
        grp_ice_lines = h5f.create_group("ice_lines")
        grp_M_transition = h5f.create_group("M_transition")
        grp_M_iso = h5f.create_group("M_iso")

        nplanets = len(planet_params["Mp"])
        for ip in range(nplanets):
            grp_Mcs.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_Mes.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_Rp.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_Mdotp.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_ice_lines.create_dataset(str(ip), shape=(0, Nmol), maxshape=(None, Nmol),
                                          dtype="f8", chunks=(1024, Nmol))
            grp_M_transition.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
            grp_M_iso.create_dataset(str(ip), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))

            pgrp_c = grp_Xc.create_group(str(ip))
            pgrp_e = grp_Xe.create_group(str(ip))
            if chemistry_params["on"]:
                nchem = Nmol  # one dataset per ice species tracked on the planet
                for js in range(nchem):
                    pgrp_c.create_dataset(str(js), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))
                    pgrp_e.create_dataset(str(js), shape=(0,), maxshape=(None,), dtype="f8", chunks=(1024,))

        groups = dict(Mcs=grp_Mcs, Mes=grp_Mes, Rp=grp_Rp, disk_Mdot_p=grp_Mdotp,
                      X_cores=grp_Xc, X_envs=grp_Xe, ice_lines=grp_ice_lines,
                      M_transition=grp_M_transition, M_iso=grp_M_iso)

    # ---- grid (written once) ----
    h5f.create_dataset("R", data=grid.Rc)

    # ---- disc profile snapshots (one row of length nR per snapshot time) ----
    h5f.create_dataset("time_snap", shape=(0,), maxshape=(None,), dtype="f8")
    h5f.create_dataset("Sigma_G", shape=(0, nR), maxshape=(None, nR), dtype="f8")
    h5f.create_dataset("Sigma_dust", shape=(0, nR), maxshape=(None, nR), dtype="f8")
    h5f.create_dataset("Sigma_pebbles", shape=(0, nR), maxshape=(None, nR), dtype="f8")
    h5f.create_dataset("Vdrift", shape=(0, 2, nR), maxshape=(None, 2, nR), dtype="f8")
    h5f.create_dataset("Sigma_pebble_size", shape=(0, nR), maxshape=(None, nR), dtype="f8")
    h5f.create_dataset("disk_atom_gas_abund", shape=(0, Natom, nR), maxshape=(None, Natom, nR), dtype="f8")
    h5f.create_dataset("disk_mol_gas_abund", shape=(0, Nmol, nR), maxshape=(None, Nmol, nR), dtype="f8")
    h5f.create_dataset("disk_atom_ice_abund", shape=(0, Natom, nR), maxshape=(None, Natom, nR), dtype="f8")
    h5f.create_dataset("disk_mol_ice_abund", shape=(0, Nmol, nR), maxshape=(None, Nmol, nR), dtype="f8")
    h5f.create_dataset("disk_ice_lines", shape=(0, Nmol), maxshape=(None, Nmol), dtype="f8")
    h5f.create_dataset("T", shape=(0, nR), maxshape=(None, nR), dtype="f8")

    if planetesimal_params["active"]:
        h5f.create_dataset("Sigma_planetesimals", shape=(0, nR), maxshape=(None, nR), dtype="f8")
        h5f.create_dataset("disk_planetesimal_atom_abund", shape=(0, Natom, nR), maxshape=(None, Natom, nR), dtype="f8")
        h5f.create_dataset("disk_planetesimal_mol_abund", shape=(0, Nmol, nR), maxshape=(None, Nmol, nR), dtype="f8")

    return h5f, groups


def write_planet_row(h5f, groups, planets, planet_model, disc, grid, disk_Mdot,
                      chemistry_params, dust_growth_params):
    """Append one row to every per-planet dataset (called every 5 steps)."""
    for ip, planet in enumerate(planets):
        grow_and_set(groups["Mcs"][str(ip)], planet.M_core.copy())
        grow_and_set(groups["Mes"][str(ip)], planet.M_env.copy())
        grow_and_set(groups["Rp"][str(ip)], planet.R.copy())
        grow_and_set(groups["disk_Mdot_p"][str(ip)], np.interp(planet.R, grid.Rc[0:-1], disk_Mdot))

        if planet_model._peb_acc:
            grow_and_set(groups["M_transition"][str(ip)], planet_model._peb_acc.M_transition(planet.R))
            grow_and_set(groups["M_iso"][str(ip)], planet_model._peb_acc.M_iso(planet.R))

        if chemistry_params["on"]:
            for js, chem in enumerate(planet.X_core):
                grow_and_set(groups["X_cores"][str(ip)][str(js)], chem)
            for js, env in enumerate(planet.X_env):
                grow_and_set(groups["X_envs"][str(ip)][str(js)], env)

            ice_lines = compute_ice_lines(disc.chem, disc.grid, threshold=dust_growth_params['thresh'])
            grow_and_set(groups["ice_lines"][str(ip)], ice_lines)


def write_disc_snapshot(h5f, disc, config, t, Natom, Nmol):
    """Append one row to every disc-profile dataset (called once per output time)."""
    chemistry_params = config["chemistry"]
    planetesimal_params = config["planetesimal"]

    grow_and_set(h5f["time_snap"], t / (1e6 * yr))  # Myr
    grow_and_set(h5f["Sigma_G"], disc.Sigma_G)
    grow_and_set(h5f["Sigma_dust"], disc.Sigma_D[0])
    grow_and_set(h5f["Sigma_pebbles"], disc.Sigma_D[1])
    grow_and_set(h5f["Sigma_pebble_size"], disc.grain_size[1])
    grow_and_set(h5f["T"], disc.T)
    grow_and_set(h5f["Vdrift"], disc.v_drift)

    if chemistry_params["on"]:
        grow_and_set(h5f["disk_atom_gas_abund"], disc.chem.gas.atomic_abundance().data)
        grow_and_set(h5f["disk_mol_gas_abund"], disc.chem.gas.data)
        grow_and_set(h5f["disk_atom_ice_abund"], disc.chem.ice.atomic_abundance().data)
        grow_and_set(h5f["disk_mol_ice_abund"], disc.chem.ice.data)
        grow_and_set(h5f["disk_ice_lines"],
                     compute_ice_lines(disc.chem, disc.grid, threshold=config["dust_growth"]["thresh"]))
    else:
        nR = len(disc.grid.Rc)
        grow_and_set(h5f["disk_atom_gas_abund"], np.zeros((Natom, nR)))
        grow_and_set(h5f["disk_mol_gas_abund"], np.zeros((Nmol, nR)))
        grow_and_set(h5f["disk_atom_ice_abund"], np.zeros((Natom, nR)))
        grow_and_set(h5f["disk_mol_ice_abund"], np.zeros((Nmol, nR)))

    if planetesimal_params["active"]:
        grow_and_set(h5f["Sigma_planetesimals"], disc.Sigma_D[2])
        if chemistry_params["on"]:
            grow_and_set(h5f["disk_planetesimal_atom_abund"], disc._planetesimal.ice_abund.atomic_abundance().data)
            grow_and_set(h5f["disk_planetesimal_mol_abund"], disc._planetesimal.ice_abund.data)


# ============================================================================
# Output filename
# ============================================================================
#
# This is the ONE place that decides the output filename, so it can never
# drift out of sync with a copy hardcoded somewhere else (that used to
# happen between this script and run_popsynth_parallel.sh). The prefix
# comes from config['simulation']['run_name'] -- change it there and every
# script that calls output_filename() picks it up automatically. A batch
# launcher does not need to know this format at all: it can just always
# invoke run_model_student.py and let `run_model()`'s own skip-if-complete
# check (below) decide whether there's anything to do.

def output_filename(config):
    """Build the deterministic output filename for one parameter combination."""
    sim_params = config['simulation']
    disc_params = config['disc']
    wind_params = config['winds']
    run_name = sim_params.get('run_name', 'run')
    return (f"{run_name}_psi{wind_params['psi_DW']}_erad{wind_params['e_rad']}_Mdot{disc_params['Mdot']:.1e}"
            f"_M{disc_params['M']:.1e}_Rd{disc_params['Rd']:.1e}.h5")


# ============================================================================
# Config logging (added for Isaac's Hof project -- see run log below)
# ============================================================================
#
# Every time run_model() actually launches a run (i.e. past the skip-if-
# complete check above), log_config() writes a full copy of the RESOLVED
# config -- config as passed to run_model(), i.e. AFTER any --flag CLI
# overrides from the __main__ block below have already been applied -- into
# CONFIG_LOG_SUBDIR under that run's own output_dir. This gives a permanent,
# per-run record of exactly which parameters produced which .h5 file, which
# matters once a parameter sweep has produced many output files and the JSON
# config on disk has since been edited/overwritten for the next sweep.
#
# Two things get written, both under <output_dir>/config_log/:
#   <output-file-stem>.json  -- the full config dict for THIS run, pretty-printed
#   run_log.csv              -- one row appended per run: timestamp, output
#                                filename, and the matching config json filename,
#                                so you can scan the whole run history at a glance
#                                without opening every individual json file.

CONFIG_LOG_SUBDIR = "config_log"   # subdirectory of the run's output_dir


def log_config(config, outfile, output_dir):
    """
    Save a copy of `config` (already CLI-override-resolved) next to this run's
    output, and append one row to a master CSV log. See module-level comment
    above for the exact files written and why.

    Parameters
    ----------
    config     : dict  -- the resolved config for this run (see output_filename())
    outfile    : str   -- full path to this run's .h5 output (used only for its
                           basename, to name/reference the logged config)
    output_dir : str   -- this run's output directory (same one `outfile` lives in)
    """
    log_dir = os.path.join(output_dir, CONFIG_LOG_SUBDIR)
    os.makedirs(log_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(outfile))[0]
    config_log_name = f"{stem}.json"
    config_log_path = os.path.join(log_dir, config_log_name)
    with open(config_log_path, "w") as f:
        json.dump(config, f, indent=2)

    timestamp = datetime.now().isoformat(timespec="seconds")
    run_log_path = os.path.join(log_dir, "run_log.csv")
    write_header = not os.path.exists(run_log_path)
    with open(run_log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "output_file", "config_log_file"])
        writer.writerow([timestamp, os.path.basename(outfile), config_log_name])


# ============================================================================
# Main driver
# ============================================================================

def _build_grid_star_kappa(config):
    """
    (Hof) Grid, star and opacity function from the config -- the exact code that used
    to be inline in step 2 of run_model(), moved here so the optional calibration
    block at the top of run_model() can build them before the output filename is known.

    Returns grid (radii in AU), star (M in Msun, R in Rsun, T_eff in K) and kappa
    (opacity function, cm^2/g; unknown names fall back to Zhu2012 as before).
    """
    grid_params = config['grid']
    star_params = config['star']
    grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'],
                spacing=grid_params['spacing'])
    star = SimpleStar(M=star_params["M"], R=star_params["R"], T_eff=star_params['T_eff'])
    opacity_tables = {"Tazzari": Tazzari2016, "Zhu2012": Zhu2012}
    kappa = opacity_tables.get(config['eos']["opacity"], Zhu2012)
    return grid, star, kappa


def run_model(config, cli_output_dir=None):
    """
    Run one disc-evolution simulation from start to finish and stream the
    result to an HDF5 file.

    Parameters
    ----------
    config : dict
        Parsed JSON configuration (see config/DiscConfig_default.json for
        an example of every field used below).
    cli_output_dir : str, optional
        Output directory, highest priority (beats $DISCEVOLUTION_OUTPUT
        and config['simulation']['output_dir']).
    """
    # ---- (Hof) optional generic calibration: solve for ONE parameter FIRST ----
    # If config has a "calibration" section, solve_initial_disc() root-finds the
    # parameter named in calibration.solve_for (alpha, M, Rd, gamma, psi_DW or e_rad)
    # so the initial disc gives config['disc']['Mdot'].
    # This has to happen BEFORE the output filename / skip check below, because the
    # solved psi_DW / e_rad / M / Rd appear in the filename. From here on `config` is
    # the returned run_config (solved value written in, 'SS' replaced by the numeric
    # alpha, plus a 'calibration_result' block), so every downstream consumer
    # (filename, config log, build_transport's psi_DW, HDF5 attrs, _integrate) sees
    # the solved value. Cost: ~n_scan + ~15 EOS solves, about a second.
    # Without a "calibration" section `calib` stays None and nothing below changes.
    calib = None
    if config.get('calibration') is not None:
        grid, star, kappa = _build_grid_star_kappa(config)
        calib = solve_initial_disc(grid, star, config, kappa)
        config = calib['run_config']

    grid_params = config['grid']
    sim_params = config['simulation']
    star_params = config['star']
    disc_params = config['disc']
    eos_params = config['eos']
    transport_params = config['transport']
    dust_growth_params = config['dust_growth']
    planet_params = config['planets']
    chemistry_params = config['chemistry']
    planetesimal_params = config['planetesimal']
    wind_params = config['winds']

    # ---- 0. skip immediately if this exact run already finished ----
    # (checked before any of the expensive setup below, so a batch sweep
    #  can be safely re-launched after an interruption without re-solving
    #  discs it already has answers for)
    output_dir = cli_output_dir or os.environ.get(
        'DISCEVOLUTION_OUTPUT', sim_params.get('output_dir', './output'))
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, output_filename(config))
    if os.path.exists(outfile):
        with h5py.File(outfile, "r") as existing:
            if existing.attrs.get("complete", False):
                print(f"Skipping -- output already complete: {outfile}")
                return
        print(f"Output file exists but is incomplete; re-running: {outfile}")

    # ---- 1. log the exact (CLI-override-resolved) config this run is using ----
    # Placed here (after the skip check, before the expensive disc solve) so
    # every run that actually launches gets logged -- including one that later
    # aborts on the alpha_SS sanity check below -- but a skipped/already-
    # complete run does not add a redundant log entry. See log_config() above
    # for exactly what gets written and where.
    log_config(config, outfile, output_dir)

    # ---- 2. grid + star ----
    # (Hof) grid/star/kappa construction moved into _build_grid_star_kappa() (same code)
    # so the calibration block at the top of run_model() can build them early. When
    # calibration ran, reuse ITS grid/star so the solved disc/EOS are tied to the same objects.
    if calib is None:
        grid, star, kappa = _build_grid_star_kappa(config)
    times = make_time_grid(sim_params)

    # ---- 3a. (Hof) generic calibration already solved the disc: unpack it ----
    if calib is not None:
        disc, eos, Sigma = calib['disc'], calib['eos'], calib['Sigma']
        alpha, alpha_SS, lambda_DW = calib['alpha'], calib['alpha_SS'], calib['lambda_DW']   # dimensionless
        info = calib['info']
        # alpha_guess attrs kept for backwards compatibility with the notebooks:
        # if alpha was the solved parameter they hold its guess, otherwise alpha was a FIXED input.
        alpha_guess = info['guess'] if info['solve_for'] == 'alpha' else alpha               # total alpha
        alpha_guess_mode = info['guess_mode'] if info['solve_for'] == 'alpha' else 'fixed'

    # ---- 3. solve for the initial disc structure ----
    # (Hof) only the original damped-loop path; skipped when the calibration above ran.
    else:
        # Initial guess for the alpha-calibration loop (Hof project):
        #   config['disc']['alpha'] = <number> -> use that number as the starting guess (original behaviour)
        #   config['disc']['alpha'] = 'SS'     -> compute a steady-state guess from Mdot, Sigma and
        #                                         the energy balance (steady_state_alpha_guess above)
        # The 'SS' string is kept in `config` (so the logged config shows 'SS'); only a deep
        # copy handed to setup_disc gets the numeric guess.
        alpha_guess_mode = 'SS' if isinstance(disc_params['alpha'], str) else 'config'   # -> HDF5 attr
        alpha_init = disc_params['alpha']
        if isinstance(alpha_init, str):
            if alpha_init != 'SS':
                raise ValueError(f"config['disc']['alpha'] must be a number or 'SS', got {alpha_init!r}")
            alpha_guess = steady_state_alpha_guess(grid, star, config, kappa)   # dimensionless, total alpha
            print(f"Steady-state initial alpha guess: {alpha_guess:.4e}")
            setup_config = copy.deepcopy(config)
            setup_config['disc']['alpha'] = alpha_guess
        else:
            alpha_guess = float(alpha_init)                                     # dimensionless, total alpha
            setup_config = config

        disc, eos, Sigma, alpha, alpha_SS, lambda_DW = setup_disc(grid, star, setup_config, kappa)

    # Sanity check: outside this range the alpha-solve above is not
    # meaningful (either essentially inviscid, or so viscous the disc
    # wouldn't survive), so there's no point integrating it forward.
    # # if (alpha_SS > 0.1) or (alpha_SS < 1e-5):
    #     print(f"Not running model - alpha_SS out of range. "
    #           f"alpha={eos.alpha}, Rd={disc_params['Rd']}, Mdisk={disc.Mtot()/Msun:.4g} Msun")
    #     return
    print(f"Running model. alpha={eos.alpha}, Rd={disc_params['Rd']}, "
          f"Mdisk={disc.Mtot()/Msun:.4g} Msun")

    # ---- 4. transport + dust growth ----
    gas, dust, diffuse = build_transport(transport_params, wind_params, disc_params,
                                          dust_growth_params, lambda_DW)
    disc = build_dust_growth_disc(grid, star, eos, Sigma, disc_params, dust_growth_params, gas)

    # ---- 5. chemistry ----
    chemistry, Natom, Nmol = build_chemistry(disc, chemistry_params, disc_params["d2g"],
                                              grid_params["nr"])

    # ---- 6. planetesimals ----
    build_planetesimals(disc, planetesimal_params)

    # ---- 7. planets ----
    planets, planet_model = build_planets(disc, planet_params, chemistry_params, wind_params)
    nplanets = len(planet_params["Mp"])

    # ---- 8. output file (path already computed in the skip-check above) ----
    h5f, groups = create_output_file(outfile, grid, config, Natom, Nmol, alpha_SS)
    h5f.attrs["alpha_guess"] = float(alpha_guess)   # starting guess (total alpha) for the calibration loop; numeric or SS-derived
    h5f.attrs["alpha_guess_mode"] = alpha_guess_mode   # 'SS' / 'config' (alpha solved), or 'fixed' (calibration solved another parameter)
    # (Hof) generic-calibration record: which parameter was solved, from what guess, to what value
    # (all in that parameter's own units, see _SOLVABLE_PARAMS), and how well Mdot matches.
    h5f.attrs["alpha_solver"] = 'brentq' if calib is not None else 'damped_iteration'
    if calib is not None:
        info = calib['info']
        h5f.attrs["calib_solve_for"] = info['solve_for']
        h5f.attrs["calib_guess"] = info['guess']
        h5f.attrs["calib_value"] = info['value']
        h5f.attrs["calib_Mdot_target"] = info['Mdot_target']     # Msun / yr
        h5f.attrs["calib_Mdot"] = info['Mdot']                   # Msun / yr, achieved at R_c[0]
        h5f.attrs["calib_Mdot_relerr"] = info['Mdot_relerr']     # dimensionless
        h5f.attrs["calib_n_evals"] = info['n_evals']
        h5f.attrs["calib_n_roots"] = info['n_roots']
    try:
        _integrate(h5f, groups, disc, grid, star, planets, planet_model, gas, dust, diffuse,
                   chemistry, times, config)
    finally:
        h5f.attrs["complete"] = True
        h5f.close()


def _disc_star_mdot(disc):
    """Accretion rate onto the star at the current disc state, in Msun/yr."""
    v = disc._gas.viscous_velocity(disc, disc.Sigma)
    Mdot = -2 * np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * v * (AU * AU) * (yr / Msun)
    return Mdot


def _integrate(h5f, groups, disc, grid, star, planets, planet_model, gas, dust, diffuse,
               chemistry, times, config):
    """The time-stepping loop, plus the periodic writes to `h5f`."""
    transport_params = config['transport']
    chemistry_params = config['chemistry']
    planet_params = config['planets']
    dust_growth_params = config['dust_growth']
    planetesimal_params = config['planetesimal']
    Natom, Nmol = h5f["disk_atom_gas_abund"].shape[1], h5f["disk_mol_gas_abund"].shape[1]

    # ---- t = 0 row, if the requested snapshot times don't already include it ----
    if 0.0 not in config['simulation']['t_interval']:
        disk_Mdot = _disc_star_mdot(disc)
        grow_and_set(h5f["t"], 0.0)
        grow_and_set(h5f["disk_Mdot_star"], disk_Mdot[0])
        grow_and_set(h5f["disk_Mass"], disc.Mtot())
        grow_and_set(h5f["Tc"], disc.T[0])
        grow_and_set(h5f["Sigc"], disc.Sigma[0])
        if planets is not None:
            write_planet_row(h5f, groups, planets, planet_model, disc, grid, disk_Mdot,
                              chemistry_params, dust_growth_params)
        write_disc_snapshot(h5f, disc, config, 0.0, Natom, Nmol)
        h5f.flush()

    # ---- bookkeeping for progress / ETA reporting ----
    t, n = 0.0, 0
    wall_spent = 0.0
    last_eta_hours, last_eta_minutes = 0, 0

    for ti in times:
        while t < ti:
            step_start = time.time()

            # Physics-limited timestep (each active operator proposes the
            # largest dt it can safely take; we use the smallest, further
            # capped so we land exactly on the next requested snapshot).
            dt_physics = float('inf')
            if transport_params['gas_transport']:
                dt_physics = min(dt_physics, disc._gas.max_timestep(disc))
            if transport_params['radial_drift']:
                dt_physics = min(dt_physics, dust.max_timestep(disc))
            dt = min(ti - t, dt_physics)

            # Rough ETA, recomputed every 1000 steps, for the progress print
            # below only (this file no longer aborts a long-running run --
            # removed per Isaac's request; see LOG OF CHANGES MADE above).
            if n >= 1000 and (n % 1000) == 0:
                avg_sec_per_step = wall_spent / n
                dt_for_eta = dt_physics if dt_physics < float('inf') else dt
                remaining_steps = int(np.ceil(max(times[-1] - t, 0.0) / max(dt_for_eta, 1e-300)))
                eta_seconds = avg_sec_per_step * remaining_steps
                last_eta_hours = int(eta_seconds // 3600)
                last_eta_minutes = int((eta_seconds % 3600) // 60)

            dust_frac = getattr(disc, "dust_frac", None)
            gas_chem = disc.chem.gas.data if chemistry_params["on"] else None
            ice_chem = disc.chem.ice.data if chemistry_params["on"] else None

            # --- gas viscous/wind evolution and dust radial drift ---
            # (tracer arrays are passed in/out so gas/dust transport also
            #  advects the chemical abundances along with the mass)
            if transport_params['gas_transport']:
                dust_frac_for_gas = dust_frac[:-1] if disc._planetesimal else dust_frac
                disc._gas(dt, disc, [dust_frac_for_gas, gas_chem, ice_chem])
            if transport_params['radial_drift']:
                dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

            # --- turbulent diffusion (only if not already folded into `dust`) ---
            if diffuse is not None:
                if gas_chem is not None:
                    gas_chem[:] += dt * diffuse(disc, gas_chem)
                if ice_chem is not None:
                    ice_chem[:] += dt * diffuse(disc, ice_chem)
                if dust_frac is not None:
                    dust_slice = dust_frac[:2] if disc._planetesimal else dust_frac[:]
                    dust_slice += dt * diffuse(disc, dust_slice)

            # --- enforce physical bounds (transport can produce small
            #     negative overshoots near sharp gradients) ---
            disc.Sigma[:] = np.maximum(disc.Sigma, 0)
            disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
            disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)
            if chemistry_params["on"]:
                disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
                disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)

            # --- planetesimal formation (converts drifting pebbles to a
            #     dynamically-decoupled planetesimal population) ---
            if disc._planetesimal:
                disc._planetesimal.update(dt, disc, dust)

            # --- chemistry: adsorption/desorption given the new T, rho,
            #     dust surface area ---
            if chemistry_params["on"]:
                dust_frac_for_chem = disc.dust_frac[:-1].sum(0) if disc._planetesimal else disc.dust_frac.sum(0)
                chemistry.update(dt, disc.T, disc.midplane_gas_density, dust_frac_for_chem, disc.chem)
                disc.update_ices(disc.chem.ice)

            # --- planet growth/migration (pebble + gas accretion, disc torques) ---
            if planets is not None:
                planet_model.integrate(dt, planets)

            # --- advance the disc's own bookkeeping (grain sizes, etc.) ---
            disc.update(dt)

            t += dt
            n += 1
            wall_spent += time.time() - step_start

            if (n % 1000) == 0:
                print(f"\rNstep: {n}", flush=True)
                print(f"\rTime: {t/(1e6*yr)} Myr", flush=True)
                print(f"\rdt: {dt/yr} yr", flush=True)
                print(f"\rETA: {last_eta_hours:02d}:{last_eta_minutes:02d} (h:m)", flush=True)

            # --- stream disc-level (and, if present, per-planet) series every 5 steps ---
            # These scalar series are useful on their own even with no planets
            # (e.g. Md(t), Mdot(t) for a disc-only run), so they're written
            # unconditionally; only the per-planet groups need `planets`.
            if (n % 5 == 0):
                disk_Mdot = _disc_star_mdot(disc)
                grow_and_set(h5f["t"], t / yr)  # years
                grow_and_set(h5f["disk_Mdot_star"], disk_Mdot[0])
                grow_and_set(h5f["disk_Mass"], disc.Mtot())
                grow_and_set(h5f["Tc"], disc.T[0])
                grow_and_set(h5f["Sigc"], disc.Sigma[0])
                if planets is not None:
                    write_planet_row(h5f, groups, planets, planet_model, disc, grid, disk_Mdot,
                                      chemistry_params, dust_growth_params)

        # --- once per requested snapshot time: full disc-profile row ---
        write_disc_snapshot(h5f, disc, config, t, Natom, Nmol)
        h5f.flush()


# ============================================================================
# Command-line entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run disc evolution model with HDF5 streaming output "
                    "Configuration can be loaded from a JSON file and overridden "
                    "via command-line arguments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_model_student.py

  # Run with custom config file
  python run_model_student.py --config config/DiscConfig_default.json

  # Override specific parameters 
  python run_model_student.py --psi_DW 0.01 --Mdot 1e-8 --M 0.1 --Rd 50

  # Use environment variable for output directory
  export DISCEVOLUTION_OUTPUT=/path/to/output
  python run_model_student.py
        """
    )
    parser.add_argument("--config", type=str,
                         default=os.path.join(os.path.dirname(__file__), "config", "DiscConfig_default.json"),
                         help="Path to configuration JSON file")
    parser.add_argument("--psi_DW", type=float, default=None, help="Override wind parameter psi_DW")
    parser.add_argument("--Mdot", type=float, default=None, help="Override accretion rate [Msun/yr]")
    parser.add_argument("--M", type=float, default=None, help="Override disc mass [Msun]")
    parser.add_argument("--Rd", type=float, default=None, help="Override characteristic disc radius [AU]")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Loaded configuration from: {args.config}")

    overrides = {
        ("winds", "psi_DW"): args.psi_DW,
        ("disc", "Mdot"): args.Mdot,
        ("disc", "M"): args.M,
        ("disc", "Rd"): args.Rd,
    }
    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
            print(f"Overriding {section}.{key}: {value}")

    run_model(config, cli_output_dir=args.output_dir)
