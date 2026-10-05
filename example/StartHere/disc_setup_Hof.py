"""
disc_setup_Hof.py
=================

Initial-disc setup for Isaac Montesdeoca Hof's Co-op project (Fall 2026).

Started as a copy of disc_setup.py (example/StartHere/). That file is left
untouched so run_model_student.py keeps working; every Hof script imports
from THIS file instead.

WHAT LIVES HERE
---------------
Everything needed to build the INITIAL gas disc, before any time evolution:

    make_eos()                 -- EOS object for a given viscous alpha_SS
                                  (unchanged from disc_setup.py)
    winds_alpha_disc(),        -- the original damped fixed-point alpha loop
    setup_disc()                  (unchanged from disc_setup.py). Used by
                                  run_model_Hof*.py only when the config has NO
                                  "calibration" section; also the reference the
                                  calibration notebook compares against.
    _build_grid_star_kappa()   -- grid (AU), star (Msun, Rsun, K) and opacity
                                  function (cm^2/g) from the config
    steady_state_alpha_guess() -- closed-form steady-state TOTAL alpha guess
                                  (config['disc']['alpha'] = 'SS')
    _SOLVABLE_PARAMS,          -- registry of solvable parameters and the unit
    _SOLVABLE_UNITS               labels used in the solver's messages
    evaluate_initial_disc()    -- forward model: parameters -> Mdot at R_c[0] (Msun/yr)
    _fixed_initial_disc()      -- calibration.solve_for = "none" (nothing solved)
    solve_initial_disc()       -- generic root-finder: solve ONE parameter so the
                                  initial Mdot at R_c[0] equals config['disc']['Mdot']

How to call it (run_model_Hof.py, run_model_Hof_dense.py and
notebooks/test_calibration_Hof.ipynb all do this):

    from disc_setup_Hof import _build_grid_star_kappa, solve_initial_disc
    grid, star, kappa = _build_grid_star_kappa(config)
    calib = solve_initial_disc(grid, star, config, kappa)   # needs a "calibration" section
    config = calib['run_config']                              # solved value written in

The import works because Python puts the running script's own directory
(example/StartHere/) on sys.path. From notebooks/ add that directory first:
    sys.path.insert(0, os.path.abspath('..'))

Dependency direction: run_model_Hof*.py -> disc_setup_Hof.py, never the other way
round, so there can be no circular import.

LOG OF CHANGES (relative to disc_setup.py)
------------------------------------------
-(2026-10-02) MOVED here from run_model_Hof.py, code unchanged:
 steady_state_alpha_guess(), the "Generic root-finder" comment block (equations for
 hold = 'alpha' vs 'alpha_SS'), _SOLVABLE_PARAMS, _SOLVABLE_UNITS,
 evaluate_initial_disc(), _fixed_initial_disc(), solve_initial_disc() and
 _build_grid_star_kappa(). The full change history of these functions (2026-09-25 ..
 2026-09-30: 'SS' guess, hold = 'alpha_SS', solver messages, solve_for = 'none')
 is still in the LOG OF CHANGES of run_model_Hof.py's module docstring.
 The only edits made while moving were in comments/docstrings: references to
 "disc_setup.winds_alpha_disc" / "disc_setup.setup_disc" now point to the functions
 of the same name in THIS file (they are identical to disc_setup.py's).
 run_model_Hof_dense.py used to carry its own, OLDER copy of the solver (without the
 2026-09-29 diagnostic messages; same physics, same returned 'info' keys); it now
 imports this one as well, so both scripts always use the same solver.
-Module docstring rewritten (this one). The original docstring described the
 Balogh et al. setup and the five other disc recipes in run_model_discchem_stream.py;
 see disc_setup.py for that text.
-make_eos(), winds_alpha_disc(), _SETUP_FUNCS and setup_disc() are unchanged
 from disc_setup.py (kept for the no-"calibration" fallback path).
-(2026-10-05) solve_initial_disc(): calibration.alpha_SS is now REQUIRED when
 hold = 'alpha_SS', which is also the default hold for solve_for = 'psi_DW'. The
 fallback alpha_SS = disc.alpha / (1 + winds.psi_DW) (dimensionless) was REMOVED and
 replaced by a ValueError that explains why and how to fix the config (add
 "alpha_SS": <value>, or "hold": "alpha"), quoting the removed default's value for
 reference. Why: winds.psi_DW is also the solver's initial GUESS for psi, so the old
 default made the fixed viscous alpha, and therefore the solved psi, depend on the
 guess. The solve still converged and met the target Mdot, but to the root of a
 different problem for each guess (notebooks/test_calibration_Hof.ipynb section 3c:
 true psi = 0.01; guess 1 -> psi 2.02, guess 100 -> psi 203). With alpha_SS explicit,
 all guesses across the bracket gave the true psi (same notebook, section 3a).
 Unchanged: hold = 'alpha' (TOTAL alpha fixed) needs no alpha_SS; all other
 solve_for values; the solved value whenever alpha_SS IS given; the returned 'info'
 keys ('alpha_SS_fixed' is always the user's value now). Configs checked:
 only config/DiscConfig_Hof_Ben.json solves for psi_DW, and it already sets
 "alpha_SS": 1e-5, so no config needed changing.

UNITS
-----
Code units G = AU = Msun = 1 (time unit = yr / 2pi; `yr` = 2pi is the conversion
factor), see DiscEvolution/constants.py. In this file:
    R, Rd, grid.Rc   AU            Sigma   g / cm^2       Mdot   Msun / yr
    M (disc)         Msun          T       K              alpha, alpha_SS, psi_DW,
    kappa            cm^2 / g      t_visc  yr                 e_rad, gamma: dimensionless
"""

import copy                     # deep-copy config in solve_initial_disc() / _fixed_initial_disc()

import numpy as np

from DiscEvolution.constants import AU, Msun, yr, Omega0, GasConst, sig_SB   # yr: t_visc in _fixed_initial_disc(); Omega0, GasConst, sig_SB: steady_state_alpha_guess()
from DiscEvolution.eos import IrradiatedEOS, LocallyIsothermalEOS, SimpleDiscEOS
from DiscEvolution.disc import AccretionDisc
from DiscEvolution.viscous_evolution import HybridWindModel
from DiscEvolution.brent import brentq            # VECTORISED Brent solver (same one IrradiatedEOS uses), for steady_state_alpha_guess()
from DiscEvolution.grid import Grid               # _build_grid_star_kappa()
from DiscEvolution.star import SimpleStar         # _build_grid_star_kappa()
from DiscEvolution.opacity import Tazzari2016, Zhu2012   # _build_grid_star_kappa()
# SciPy's scalar Brent root-finder, used by solve_initial_disc().
# Imported under a different name because `brentq` above is DiscEvolution's
# VECTORISED Brent solver (used by steady_state_alpha_guess), with a different call signature.
from scipy.optimize import brentq as scipy_brentq


def make_eos(eos_params, star, alpha_t, kappa=None, psi=None, e_rad=None):
    """
    Build the equation-of-state object named in eos_params['type'].

    alpha_t here must be the *viscous* alpha (what DiscEvolution calls
    alpha_SS): if a disc wind is present, this is smaller than the total
    alpha by a factor (1 + psi), since the wind carries away angular
    momentum without contributing to viscous heating.
    """
    eos_type = eos_params["type"]
    if eos_type == "SimpleDiscEOS":
        return SimpleDiscEOS(star, alpha_t=alpha_t)
    elif eos_type == "LocallyIsothermalEOS":
        return LocallyIsothermalEOS(star, eos_params["h0"], eos_params["q"], alpha_t)
    elif eos_type == "IrradiatedEOS":
        return IrradiatedEOS(star, alpha_t=alpha_t, kappa=kappa, psi=psi, e_rad=e_rad,
                              Tmax=eos_params["Tmax"])
    else:
        raise ValueError(f"Unknown eos type {eos_type!r}")


def winds_alpha_disc(grid, star, disc_params, eos_params, wind_params, kappa):
    """
    Solve for alpha given a fixed Rd, Mdot, Mdisk and disc-wind strength.

    Units in / out (this is the part that trips people up):
        grid.Rc                        -- radii, in AU
        disc_params['Mdot']            -- target accretion rate, in Msun/yr
        disc_params['M']               -- disc mass, in Msun (converted to
                                           grams internally, since AU/Msun
                                           are cgs constants and G = 1 in
                                           these units)
        disc_params['Rd']              -- characteristic radius, in AU
        returned Sigma                 -- surface density, in g/cm^2

    Physics:
        Sigma(R) starts as the standard self-similar power-law profile
        Sigma ~ (R/Rd)^-gamma * exp[-(R/Rd)^(2-gamma)], renormalized to
        the target disc mass. We then iterate:
            1. build the EOS (temperature structure) for the current alpha
            2. measure the Mdot that alpha actually produces
            3. rescale alpha by (target Mdot / actual Mdot)
        Steps are damped by averaging old/new alpha 50/50 each iteration,
        since the direct update can oscillate. 100 iterations converges
        comfortably for the (psi, Mdot, Mdisk, Rd) ranges used in this
        project.

        With a disc wind (psi_DW > 0) not all of alpha drives viscous
        accretion: `lambda_DW` is the wind lever-arm parameter (Tabone
        et al. 2022 notation) and `alpha_SS = alpha / (1 + psi)` is the
        viscous-only alpha that actually goes into the EOS.

    Returns
    -------
    disc, eos, Sigma, alpha, alpha_SS, lambda_DW
    """
    Mdot_target = disc_params["Mdot"]        # Msun/yr
    Mdisk = disc_params["M"] * Msun          # g
    Rd = disc_params["Rd"]                   # AU
    gamma = disc_params["gamma"]              # Sigma power-law index
    alpha = disc_params["alpha"]              # initial guess, total alpha

    psi = wind_params["psi_DW"]               # wind mass-loss parameter
    e_rad = wind_params["e_rad"]               # wind lever-arm efficiency
    if psi > 0 and e_rad < 1.0:
        lambda_DW = 1 / (2 * (1 - e_rad) * (3 / psi + 1)) + 1
    else:
        # Pure-viscous limit (psi -> 0): the wind velocity v_DW in
        # HybridWindModel is proportional to psi, so it's exactly zero here
        # regardless of lambda_DW -- there's no finite limit of the lambda_DW
        # formula itself (it diverges as psi -> 0), so just pick any finite
        # placeholder value; it multiplies a wind term that's already zero.
        lambda_DW = np.inf
    alpha_SS = alpha / (1 + psi)

    R = grid.Rc
    Sigma = (R / Rd) ** (-gamma) * np.exp(-(R / Rd) ** (2 - gamma))

    # Normalize to the requested disc mass (AccretionDisc.Mtot() integrates
    # 2*pi*R*Sigma over the grid in cgs, so this works even though `eos` is
    # not defined yet -- we only need Sigma for the mass integral here).
    disc = AccretionDisc(grid, star, eos=None, Sigma=Sigma)
    Sigma *= Mdisk / disc.Mtot()

    gas = HybridWindModel(psi, lambda_DW)

    for _ in range(100):
        eos = make_eos(eos_params, star, alpha_SS, kappa=kappa, psi=psi, e_rad=e_rad)
        eos.set_grid(grid)
        eos.update(0, Sigma)

        disc = AccretionDisc(grid, star, eos, Sigma)
        v_r = gas.viscous_velocity(disc, Sigma)
        Mdot_actual = disc.Mdot(v_r)[0]      # Msun/yr

        alpha = 0.5 * (alpha + alpha * Mdot_target / Mdot_actual)
        alpha_SS = alpha / (1 + psi)

    return disc, eos, Sigma, alpha, alpha_SS, lambda_DW


# Dispatch table: add an entry here (and a matching function above, or
# ported over from run_model_discchem_stream.py) to support another
# disc-initialization recipe without touching run_model_student.py.
_SETUP_FUNCS = {
    "winds-alpha": winds_alpha_disc,
}


def setup_disc(grid, star, config, kappa):
    """
    Build the initial disc for `config['grid']['type']`.

    This is the single entry point run_model_student.py calls; it just
    looks up the right recipe above and forwards the relevant config
    sections to it.
    """
    grid_type = config["grid"]["type"]
    try:
        func = _SETUP_FUNCS[grid_type]
    except KeyError:
        raise ValueError(
            f"Unknown or unsupported grid type {grid_type!r}. "
            f"This simplified module implements: {list(_SETUP_FUNCS)}. "
            "See run_model_discchem_stream.py for the other five disc-"
            "initialization recipes (Booth-alpha/Rd/Mdot, LBP, winds-Rd/"
            "winds-Mdot) if you need one of those -- port the branch over "
            "into a function here and add it to _SETUP_FUNCS."
        )
    return func(grid, star, config["disc"], config["eos"], config["winds"], kappa)


# ============================================================================
# Grid, star and opacity from the config (moved from run_model_Hof.py, 2026-10-02)
# ============================================================================

def _build_grid_star_kappa(config):
    """
    (Hof) Grid, star and opacity function from the config -- the exact code that used
    to be inline in step 2 of run_model(). First split out (inside run_model_Hof.py) so
    the optional calibration block at the top of run_model() can build them before the
    output filename is known; (2026-10-02) moved to disc_setup_Hof.py, code unchanged,
    so the calibration notebook can build grid/star/kappa without importing run_model_Hof.

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


# ============================================================================
# Steady-state initial guess for alpha (added for Isaac's Hof project)
# ============================================================================

def steady_state_alpha_guess(grid, star, config, kappa):
    """
    Steady-state estimate of the TOTAL alpha (alpha_SS + alpha_DW), used as the
    starting guess for the alpha-calibration loop in winds_alpha_disc()
    when config['disc']['alpha'] == 'SS'.

    Idea: in a steady disc, Mdot = 3 pi nu_SS Sigma (1 + psi), with
    nu_SS = alpha_SS c_s^2 / Omega_K and alpha = alpha_SS (1 + psi), so

        alpha * c_s^2 = Mdot * Omega_K / (3 pi Sigma)                    (*)

    Substituting (*) into the viscous heating term of IrradiatedEOS removes
    alpha from the energy balance, so T(R) can be solved from Mdot alone
    (one Brent solve, no iteration). alpha then follows from (*) with c_s(T).

    Everything below mirrors winds_alpha_disc (Sigma profile) and
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

    # ---- Sigma(R): identical to winds_alpha_disc ----
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
# winds_alpha_disc() can only solve for alpha, and does so with 100
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
# With no "calibration" section, run_model() in run_model_Hof*.py behaves exactly as
# before (setup_disc() damped loop, defined above in this file).
#
# ---- (Hof, 2026-09-28) Solving for psi_DW: hold alpha_SS fixed, not total alpha ----
# Relevant equations, as implemented in the code:
#   viscosity  (EOS gets alpha_SS)     nu    = alpha_SS c_s^2 / Omega        (c_s^2 ∝ T)
#   HybridWindModel.viscous_velocity   v_r   = -3/(Sigma sqrt(R)) d/dR(nu Sigma sqrt(R))   [viscous]
#                                              - (3/2) psi nu / R                             [wind]
#   accretion rate                     Mdot  = -2 pi R Sigma v_r ≈ 3 pi Sigma nu (k + psi),  k = O(1)
#                                            ≈ 3 pi Sigma (c_s^2/Omega) alpha_SS (1 + psi)
#   IrradiatedEOS heating              Q     = e_rad (9/8) alpha_SS c_s^2 Omega Sigma (1 + psi/3)
#
# (a) TOTAL alpha fixed (hold = "alpha", only if set explicitly): alpha_SS (1 + psi) = alpha_tot, so
#       Mdot ≈ 3 pi Sigma (c_s^2/Omega) alpha_tot    -- psi drops out of the prefactor.
#     psi only enters through T, via the heating factor
#       alpha_SS (1 + psi/3) = alpha_tot (1 + psi/3)/(1 + psi),
#     which goes from alpha_tot (psi -> 0) to alpha_tot/3 (psi -> inf): at most a
#     factor 3 in heating, and T ∝ Q^s with s <~ 1/3 (s = 0 where irradiation or the
#     Tmax cap dominate). So Mdot(psi) is BOUNDED (Ben config: only x2.1 over
#     psi = 1e-3..1e3) and d ln Mdot / d ln psi ~ 0.05. Error propagation:
#       delta ln psi = delta ln(Mdot/alpha_tot) / (d ln Mdot / d ln psi) ~ 20 x delta ln alpha
#     -> psi is ill-conditioned: only a narrow window of alpha has a root at all, and
#        the root moves a lot for small changes in alpha.
# (b) alpha_SS fixed (hold = "alpha_SS", DEFAULT for solve_for = "psi_DW" since 2026-09-30;
#     since 2026-10-05 the fixed value calibration.alpha_SS MUST be given explicitly, there
#     is no longer a default derived from the psi guess -- see solve_initial_disc()):
#       Mdot ≈ 3 pi Sigma (c_s^2/Omega) alpha_SS (1 + psi)   -- psi multiplies Mdot directly,
#     and the heating also grows with psi (same sign), so
#       d ln Mdot / d ln psi ≈ psi/(1 + psi) + (small, positive)  -> ~1 for psi >~ 1:
#     monotonic, unbounded (Ben config: x480 over the same range), well-conditioned.
#     Caveat: for psi << 1 the slope -> 0 here too; Mdot cannot constrain very weak winds.
# (b) is also the physical picture: fixed turbulence plus a wind of strength psi,
#     alpha_DW = psi alpha_SS.

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

# (Hof, 2026-09-29) Unit labels for the solvable parameters, used only in the
# solver's printed success / failure messages so every printed value carries its unit.
_SOLVABLE_UNITS = {
    "alpha":  "(dimensionless, total alpha)",
    "M":      "Msun",
    "Rd":     "AU",
    "gamma":  "(dimensionless)",
    "psi_DW": "(dimensionless, alpha_DW/alpha_SS)",
    "e_rad":  "(dimensionless)",
}


def evaluate_initial_disc(grid, star, p, eos_params, kappa):
    """
    Forward model: build the initial gas disc for ONE fixed parameter set and
    measure its inner accretion rate. This is exactly one pass of the loop
    body in winds_alpha_disc() (same Sigma profile, same
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

    # Wind lever arm: identical to winds_alpha_disc (psi -> 0 gives inf;
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


# ---- (Hof, 2026-09-30) "no-solve" mode: calibration.solve_for = "none" ----
# Every other calibration mode adjusts ONE parameter so the initial disc reproduces
# config['disc']['Mdot'] at the inner cell R_c[0] (~0.1 AU). That makes the comparison of
# two runs which differ in the Sigma profile (e.g. gamma = 1 vs gamma = 3/8 at fixed M and
# Rd) implicitly a comparison of alpha as well: the shallower profile has ~30x less gas at
# 0.1 AU, so the calibration raised alpha_SS ~34x (1.36e-3 -> 4.59e-2, M = 0.01 Msun,
# Mdot = 1e-8 Msun/yr, psi = 0.01), and the discs then evolved on viscous times ~100x apart.
# With solve_for = "none", NOTHING is solved: alpha (TOTAL), M, Rd, gamma, psi_DW and e_rad
# are all taken from the config as given, and the initial inner Mdot becomes an OUTPUT
# (printed, and stored in HDF5 attr calib_Mdot). Use this whenever two runs must share the
# same viscosity, so that the only difference between them is the one you changed.
#     "calibration": {"solve_for": "none"}
# config['disc']['Mdot'] is then only a nominal label (it still appears in the output
# filename via output_filename(), but it is NOT the disc's real initial Mdot).

def _fixed_initial_disc(grid, star, config, kappa):
    """
    (Hof, 2026-09-30) Build the initial disc with ALL parameters fixed at their config
    values -- no root-finding -- for config['calibration'] = {"solve_for": "none"}.

    It is one call to evaluate_initial_disc() (same Sigma profile, mass normalisation,
    lambda_DW, EOS and HybridWindModel velocity as every other mode), so the disc built here
    is identical to what the solver would build if it happened to land on these values.

    Returns
    -------
    Same dict shape as solve_initial_disc(), so run_model() can unpack it unchanged:
        'disc', 'eos', 'Sigma' (g/cm^2), 'alpha' (TOTAL alpha, dimensionless),
        'alpha_SS' (dimensionless), 'lambda_DW' (dimensionless),
        'run_config' (deep copy of config, values unchanged, + 'calibration_result'),
        'info'       (same keys as the solving path, plus 't_visc_Rd_yr').
    """
    calib = config['calibration']

    # ---- keys that only make sense when something is being solved: reject, don't ignore ----
    # (silently ignoring e.g. a leftover "bracket" would let the user believe it had an effect)
    unused = [k for k in ('bracket', 'hold', 'alpha_SS', 'n_scan', 'rtol') if k in calib]
    if unused:
        raise ValueError(f"calibration.solve_for = 'none' solves nothing, so calibration keys "
                         f"{unused} have no meaning; remove them (only 'solve_for' is used).")

    # ---- full parameter set, straight from the config (units as in evaluate_initial_disc) ----
    p = {
        'alpha':  config['disc']['alpha'],    # TOTAL alpha = alpha_SS (1 + psi), dimensionless
        'M':      config['disc']['M'],        # Msun
        'Rd':     config['disc']['Rd'],       # AU
        'gamma':  config['disc']['gamma'],    # Sigma power-law index, dimensionless
        'psi_DW': config['winds']['psi_DW'],  # alpha_DW / alpha_SS, dimensionless
        'e_rad':  config['winds']['e_rad'],   # fraction of accretion heating radiated, dimensionless
    }
    for k, v in p.items():
        # 'SS' is rejected: steady_state_alpha_guess() is itself an Mdot calibration, which is
        # exactly what this mode is meant to avoid.
        if isinstance(v, str) or v is None:
            raise ValueError(f"calibration.solve_for = 'none' needs every parameter numeric, got "
                             f"{k} = {v!r}. Give alpha as a number (TOTAL alpha, dimensionless); "
                             f"'SS' is an Mdot calibration and is not allowed in this mode.")
        p[k] = float(v)

    # ---- the ONE forward-model evaluation ----
    res = evaluate_initial_disc(grid, star, p, config['eos'], kappa)
    Mdot = res['Mdot']                                           # Msun / yr, at R_c[0]
    Mdot_nominal = float(config['disc']['Mdot'])                 # Msun / yr, label only
    if not np.isfinite(Mdot):
        raise RuntimeError(f"calibration.solve_for = 'none': the initial disc gives a non-finite "
                           f"Mdot at R_c[0] ({Mdot}); check the parameters {p}.")

    # ---- viscous time at Rd (Lynden-Bell & Pringle), for comparing runs ----
    #   t_nu = Rd^2 / (3 (2 - gamma)^2 nu(Rd))
    # nu = alpha_SS c_s^2 / Omega_K is in code units (AU^2 per code-time unit; code time =
    # yr / 2pi), so t_nu comes out in code-time units and is divided by `yr` (= 2pi) -> years.
    # Note: the EOS nu contains alpha_SS only (the wind torque is not a viscosity), which is
    # the usual definition of the viscous time in the hybrid wind model.
    nu_Rd = float(np.interp(p['Rd'], grid.Rc, res['disc'].nu))   # AU^2 / code-time unit
    t_visc_Rd = p['Rd'] ** 2 / (3 * (2 - p['gamma']) ** 2 * nu_Rd) / yr   # yr

    info = {
        'solve_for': 'none',
        'guess': np.nan,                        # nothing was solved
        'guess_mode': 'fixed',
        'value': np.nan,                        # nothing was solved
        'Mdot_target': Mdot_nominal,            # Msun / yr, NOMINAL only (not imposed)
        'Mdot': float(Mdot),                    # Msun / yr, actual initial Mdot at R_c[0]
        'Mdot_relerr': float(abs(Mdot / Mdot_nominal - 1)),   # dimensionless, informational only
        'n_evals': 1,
        'n_roots': 0,
        'hold': 'alpha',                        # the TOTAL alpha from the config is used as given
        'alpha_SS_fixed': np.nan,               # dimensionless; not used in this mode
        't_visc_Rd_yr': float(t_visc_Rd),       # yr, viscous time at Rd (see above)
    }

    print(f"Calibration: solve_for = 'none' -> NO parameter solved; every value taken from the config.")
    print(f"    total alpha = {p['alpha']:.4e}, alpha_SS = {res['alpha_SS']:.4e} (dimensionless); "
          f"gamma = {p['gamma']:.4g}, M = {p['M']:.4g} Msun, Rd = {p['Rd']:.4g} AU, "
          f"psi_DW = {p['psi_DW']:.4g}, e_rad = {p['e_rad']:.4g}")
    print(f"    initial Mdot at R_c[0] = {grid.Rc[0]:.4g} AU: {Mdot:.4e} Msun/yr "
          f"(nominal disc.Mdot = {Mdot_nominal:.4e} Msun/yr, ratio {Mdot / Mdot_nominal:.3g})")
    print(f"    viscous time at Rd: t_nu = Rd^2 / (3 (2-gamma)^2 nu(Rd)) = {t_visc_Rd:.4e} yr")
    print(f"    NOTE: the output filename's 'Mdot' token is the NOMINAL disc.Mdot; the real initial "
          f"Mdot is stored in HDF5 attr calib_Mdot.")
    if Mdot <= 0:
        # physically possible (net OUTWARD flow at the inner edge when nu*Sigma rises steeply
        # outward there); the run is still well defined, so warn rather than stop
        print(f"    WARNING: Mdot at R_c[0] is <= 0 (net outward flow at the inner edge) for this "
              f"parameter set; the run will still proceed.")

    run_config = copy.deepcopy(config)          # values unchanged (nothing was solved)
    run_config['calibration_result'] = info     # ends up in the logged config json

    return {'disc': res['disc'], 'eos': res['eos'], 'Sigma': res['Sigma'],
            'alpha': p['alpha'], 'alpha_SS': res['alpha_SS'],
            'lambda_DW': res['lambda_DW'], 'run_config': run_config, 'info': info}


def solve_initial_disc(grid, star, config, kappa):
    """
    Root-find ONE disc parameter (config['calibration']['solve_for']) so that
    the initial disc's inner accretion rate equals config['disc']['Mdot'],
    holding every other parameter at its config value.

    config['calibration'] keys
    --------------------------
    solve_for : str, required   -- one of _SOLVABLE_PARAMS ('alpha', 'M', 'Rd',
                                   'gamma', 'psi_DW', 'e_rad'), or (Hof, 2026-09-30)
                                   'none': solve nothing, use every config value as
                                   given (see _fixed_initial_disc(); the other keys
                                   below are then rejected, and disc.Mdot is only a
                                   nominal label -- the real initial Mdot is an output)
    bracket   : [lo, hi], opt.  -- search range in the parameter's own units
                                   (default: the range in _SOLVABLE_PARAMS)
    rtol      : float, opt.     -- max allowed |Mdot/Mdot_target - 1| (default 1e-6)
    n_scan    : int, opt.       -- number of points in the coarse scan (default 31)
    hold      : str, opt.       -- (Hof, only with solve_for = 'psi_DW') which alpha stays
                                   fixed while psi varies: 'alpha_SS' (default since
                                   2026-09-30; viscous alpha fixed, total alpha =
                                   alpha_SS (1 + psi) at each trial psi) or 'alpha' (TOTAL
                                   alpha = disc.alpha fixed; only if set explicitly).
                                   See the equations above _SOLVABLE_PARAMS for why
                                   'alpha_SS' is much better conditioned.
    alpha_SS  : float           -- (Hof, only with hold = 'alpha_SS') the fixed viscous
                                   alpha, dimensionless. (Hof, 2026-10-05) REQUIRED with
                                   hold = 'alpha_SS', i.e. also for solve_for = 'psi_DW'
                                   with no "hold" key (the default hold). There is no
                                   default any more: the old one, disc.alpha / (1 + psi_DW),
                                   depended on the psi GUESS. A missing alpha_SS raises
                                   ValueError. Not needed (and rejected) with hold = 'alpha'.

    The config value of the solved parameter is used ONLY as the initial
    guess (to pick a root if there are several). For 'alpha' it may be 'SS'
    (steady_state_alpha_guess). Every other parameter must be numeric (except
    disc.alpha with hold = 'alpha_SS', which is not used as a fixed value then).

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
    # (Hof, 2026-09-30) solve_for = 'none': solve NOTHING, build the disc from the config
    # values as given (alpha is an input, the initial Mdot an output). See the comment block
    # above _fixed_initial_disc() for why this is needed when comparing e.g. gamma values.
    if name == 'none':
        return _fixed_initial_disc(grid, star, config, kappa)
    if name not in _SOLVABLE_PARAMS:
        raise ValueError(f"calibration.solve_for must be one of {list(_SOLVABLE_PARAMS)} or 'none', got {name!r}")
    section, key, scale, (lo, hi) = _SOLVABLE_PARAMS[name]
    if name == "Rd" and hi is None:
        hi = float(grid.Rc[-1])                                  # AU, outermost cell centre
    # (Hof, 2026-09-29) keep the registry's default range: it also encodes the physical
    # limits (gamma < 2, e_rad < 1, alpha <= 1, ...) and is used in the no-root diagnostic
    # to say whether an extrapolated root would even be physical.
    lo_def, hi_def = lo, hi                                      # parameter's own units
    unit = _SOLVABLE_UNITS.get(name, "")                         # (Hof) label for printed messages
    if calib.get('bracket') is not None:
        lo, hi = (float(v) for v in calib['bracket'])            # parameter's own units
    # (Hof, 2026-09-29) explicit check: a reversed/degenerate bracket would otherwise give a
    # confusing scan (xs decreasing) or a single repeated point.
    if not lo < hi:
        raise ValueError(f"calibration FAILED: bracket must satisfy lo < hi, got [{lo}, {hi}] {unit}")
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
    # ---- (Hof) which alpha is held fixed while solving for psi_DW ----
    #   hold = 'alpha'    : total alpha p['alpha'] fixed (must be requested explicitly for psi_DW)
    #   hold = 'alpha_SS' : viscous alpha fixed at alpha_SS_fixed (dimensionless); the total
    #                       alpha passed to evaluate_initial_disc is alpha_SS_fixed (1 + psi)
    # (Hof, 2026-09-30) DEFAULT CHANGED: when solving for psi_DW the default is now
    # hold = 'alpha_SS' (viscous alpha fixed, well-conditioned; see case (b) above
    # _SOLVABLE_PARAMS). The TOTAL alpha is held fixed only if the config says so
    # explicitly with "hold": "alpha" in the calibration section. For every other
    # solve_for the default stays 'alpha' (hold = 'alpha_SS' is not valid there).
    hold_default = 'alpha_SS' if name == 'psi_DW' else 'alpha'
    hold = calib.get('hold', hold_default)
    hold_src = 'calibration.hold' if 'hold' in calib else 'default'   # for the printed message below
    if hold not in ('alpha', 'alpha_SS'):
        raise ValueError(f"calibration.hold must be 'alpha' or 'alpha_SS', got {hold!r}")
    if name != 'psi_DW' and (hold != 'alpha' or 'alpha_SS' in calib):
        raise ValueError(f"calibration.hold = 'alpha_SS' / calibration.alpha_SS are only valid with "
                         f"solve_for = 'psi_DW' (got solve_for = {name!r}).")
    if hold == 'alpha' and 'alpha_SS' in calib:
        raise ValueError("calibration.alpha_SS given but calibration.hold is not 'alpha_SS'; "
                         "set \"hold\": \"alpha_SS\" to hold the viscous alpha fixed.")
    hold_SS = (hold == 'alpha_SS')
    alpha_SS_fixed = np.nan                                      # dimensionless; NaN = not used
    if hold_SS:
        # (Hof, 2026-10-05) alpha_SS is now REQUIRED whenever hold = 'alpha_SS' (which is
        # also the DEFAULT hold for solve_for = 'psi_DW'). The old fallback
        #     alpha_SS_fixed = disc.alpha / (1 + winds.psi_DW)        (dimensionless)
        # was REMOVED: winds.psi_DW is also the INITIAL GUESS for psi, so that default made
        # the viscous alpha being held fixed -- and therefore the solved psi -- depend on the
        # guess. Brent still converged and met the target Mdot, but to the root of a
        # DIFFERENT problem for every guess (test_calibration_Hof.ipynb, section 3c: with the
        # sweep config, true psi = 0.01, a guess of 1 returned psi = 2.02 and a guess of 100
        # returned psi = 203). With alpha_SS given explicitly, every guess across the whole
        # bracket returned the true psi to machine precision (same notebook, section 3a).
        if calib.get('alpha_SS') is None:
            # For reference only: what the REMOVED default would have used (dimensionless),
            # shown only if disc.alpha and winds.psi_DW are numeric. Not computed for
            # disc.alpha = 'SS' (that would need steady_state_alpha_guess, and the value would
            # still depend on the psi guess).
            ref = ""
            if not isinstance(p['alpha'], str) and not isinstance(p['psi_DW'], str):
                old_default = float(p['alpha']) / (1 + float(p['psi_DW']))   # dimensionless
                ref = (f"\n  For reference, the removed default would have been disc.alpha / (1 + winds.psi_DW) "
                       f"= {float(p['alpha']):.4g} / (1 + {float(p['psi_DW']):.4g}) = {old_default:.4e} "
                       f"(dimensionless). Use that value only if winds.psi_DW is the wind strength you "
                       f"actually intend, not just a starting guess.")
            raise ValueError(
                f"calibration FAILED: solve_for = 'psi_DW' with hold = 'alpha_SS' (from {hold_src}) "
                f"requires an explicit \"alpha_SS\" (the fixed VISCOUS alpha, dimensionless) in the "
                f"calibration section.\n"
                f"  It no longer defaults to disc.alpha / (1 + winds.psi_DW): winds.psi_DW is also the "
                f"initial GUESS for psi, so that default made the fixed alpha_SS, and hence the solved "
                f"psi, depend on the guess (see notebooks/test_calibration_Hof.ipynb, section 3c).\n"
                f"  Fix: add e.g. \"alpha_SS\": 1e-4 to the calibration section, i.e.\n"
                f"      \"calibration\": {{\"solve_for\": \"psi_DW\", \"alpha_SS\": 1e-4}}\n"
                f"  or set \"hold\": \"alpha\" to hold the TOTAL alpha = disc.alpha fixed instead "
                f"(ill-conditioned for psi; see the comment block above _SOLVABLE_PARAMS)." + ref
            )
        alpha_SS_fixed = float(calib['alpha_SS'])                # dimensionless, given explicitly
        alpha_SS_src = 'calibration.alpha_SS'
        if not alpha_SS_fixed > 0:
            raise ValueError(f"calibration: fixed alpha_SS must be > 0, got {alpha_SS_fixed}")
        # (Hof, 2026-09-30) also say whether hold = 'alpha_SS' was chosen by default or explicitly
        print(f"Calibration: holding alpha_SS = {alpha_SS_fixed:.4e} (dimensionless) fixed "
              f"[hold = 'alpha_SS' from {hold_src}; alpha_SS from {alpha_SS_src}]; "
              f"total alpha = alpha_SS (1 + psi) varies with psi. "
              f"Set \"hold\": \"alpha\" in the calibration section to hold the TOTAL alpha instead.")
    elif name == 'psi_DW':
        # (Hof, 2026-09-30) total alpha held fixed only on explicit request ("hold": "alpha")
        # (p['alpha'] printed as-is: a non-numeric 'SS' is rejected with a clear error just below)
        print(f"Calibration: holding TOTAL alpha = {p['alpha']} (dimensionless) fixed "
              f"[hold = 'alpha' from calibration.hold]; alpha_SS = alpha / (1 + psi) varies with psi.")

    for k, v in p.items():
        if hold_SS and k == 'alpha':
            continue          # (Hof) total alpha is derived from alpha_SS_fixed, not used as a fixed input
        if k != name and isinstance(v, str):
            raise ValueError(f"Fixed parameter {k!r} must be numeric when solving for {name!r} "
                             f"(got {v!r}); 'SS' is only allowed for alpha when solve_for = 'alpha'.")
    guess = float(guess)
    # (Hof, 2026-09-29) The guess only chooses among roots found INSIDE the bracket. If it
    # lies outside, the solver cannot return a value near it -- tell the user up front.
    if not lo <= guess <= hi:
        print(f"WARNING calibration: initial guess {name} = {guess:.4g} {unit} is OUTSIDE the "
              f"bracket [{lo:.4g}, {hi:.4g}] {unit}. Only roots inside the bracket can be found; "
              f"widen calibration.bracket if you expect the solution near the guess.")

    def params_at(v):
        """(Hof) Full parameter set with the free parameter = v (its own units). With
        hold = 'alpha_SS' the total alpha is recomputed as alpha_SS_fixed (1 + psi), so
        evaluate_initial_disc's alpha_SS = alpha / (1 + psi) equals alpha_SS_fixed exactly."""
        q = dict(p)
        q[name] = float(v)
        if hold_SS:
            q['alpha'] = alpha_SS_fixed * (1 + float(v))          # total alpha, dimensionless
        return q

    # ---- map parameter value <-> search variable x ----
    if scale == 'log':
        if lo <= 0:
            raise ValueError(f"bracket for log-scaled {name!r} must be > 0, got {[lo, hi]}")
        to_x, from_x = np.log10, (lambda x: 10.0 ** x)
    else:
        to_x, from_x = (lambda v: v), (lambda x: x)

    n_evals = [0]   # mutable counter of forward-model evaluations (list so g() can modify it)
    # (Hof, 2026-09-29) raw Mdot (Msun/yr, may be <= 0) of the most recent g() call. g() must
    # return nan for Mdot <= 0 (log10 undefined), which alone cannot distinguish "Mdot <= 0:
    # net OUTWARD flow at R_c[0]" from "forward model failed"; the scan records this too.
    last_Mdot = [np.nan]

    def g(x):
        """log10(Mdot / Mdot_target) with the free parameter set to from_x(x); nan if Mdot <= 0."""
        q = params_at(from_x(x))      # (Hof) was dict(p) + q[name] = ...; now also handles hold = 'alpha_SS'
        n_evals[0] += 1
        Mdot = evaluate_initial_disc(grid, star, q, config['eos'], kappa)['Mdot']   # Msun / yr
        last_Mdot[0] = Mdot           # (Hof) keep the raw value, sign included
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
    # (Hof, 2026-09-29) also record the raw Mdot at each scan point (Msun/yr; nan if the
    # forward model raised), to classify the unusable points in the no-root diagnostic:
    #   Mdot_scan <= 0      -> net OUTWARD flow at R_c[0] (physical, not a failure)
    #   Mdot_scan nan/inf   -> forward model failed (e.g. EOS temperature solve)
    gs, Mdot_scan = [], []
    with np.errstate(all='ignore'):
        for x in xs:
            last_Mdot[0] = np.nan                                # reset: stays nan if g() raises
            gs.append(g_safe(x))
            Mdot_scan.append(last_Mdot[0])
    gs, Mdot_scan = np.array(gs), np.array(Mdot_scan, dtype=float)
    ok = np.isfinite(gs)
    if not ok.any():
        # (Hof, 2026-09-29) say how many points had Mdot <= 0 (outward flow) vs a failed model
        n_neg = int(np.sum(np.isfinite(Mdot_scan) & (Mdot_scan <= 0)))
        raise RuntimeError(f"calibration FAILED: no usable Mdot for any {name} in [{lo:.4g}, {hi:.4g}] {unit}: "
                           f"{n_neg}/{n_scan} scan points have Mdot <= 0 at R_c[0] (net outward flow), "
                           f"{n_scan - n_neg}/{n_scan} have a failed forward model (exception / non-finite Mdot).")
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
        # ---- (Hof, 2026-09-29) diagnose WHY there is no root inside [lo, hi] ----
        # Three distinct situations, each with its own message:
        #   (a) the sign change is hidden behind scan points where the forward model failed
        #       (g = nan, e.g. the EOS temperature solve did not converge) -> a root probably
        #       exists INSIDE the bracket, in the failed region;
        #   (b) |g| is smallest at one END of the bracket -> Mdot is still heading towards the
        #       target there, so the root most likely lies OUTSIDE the bracket on that side.
        #       Estimate it by linear extrapolation of g(x) from the two outermost usable scan
        #       points (x = log10(value) for 'log' parameters, so this is a power-law
        #       extrapolation of Mdot in the parameter);
        #   (c) |g| is smallest at an INTERIOR scan point -> Mdot has a turning point and
        #       comes closest to the target inside the bracket, then turns away again: the
        #       target is probably unreachable for ANY value of this parameter.
        #   (d) [added after a confusing report] |g| is smallest at the last USABLE point, but
        #       that is not the bracket end: beyond it Mdot <= 0 or the model failed, so Mdot
        #       turns over INSIDE the bracket -> same conclusion as (c), not "widen the bracket".
        fi = np.flatnonzero(ok)                                  # indices of usable (Mdot > 0) scan points
        Mdot_min = Mdot_target * 10 ** np.nanmin(gs)            # Msun / yr, smallest POSITIVE Mdot
        Mdot_max = Mdot_target * 10 ** np.nanmax(gs)            # Msun / yr, largest Mdot
        # (Hof, 2026-09-29) classify the unusable points (see the scan above)
        i_neg = np.flatnonzero(np.isfinite(Mdot_scan) & (Mdot_scan <= 0))   # Mdot <= 0: outward flow
        i_fail = np.flatnonzero(~np.isfinite(Mdot_scan))                     # forward model failed
        msg = [f"calibration FAILED: no {name} in the bracket [{lo:.4g}, {hi:.4g}] {unit} gives "
               f"Mdot = {Mdot_target:.3e} Msun/yr with the other parameters fixed.",
               f"  Positive Mdot reached over the bracket: [{Mdot_min:.3e}, {Mdot_max:.3e}] Msun/yr "
               f"(at {len(fi)}/{n_scan} scan points)."]
        if len(i_neg):
            msg.append(f"  At {len(i_neg)} scan point(s) Mdot at R_c[0] is <= 0 (net OUTWARD flow, not a "
                       f"solver failure): {name} = " + ", ".join(f"{from_x(xs[i]):.3g}" for i in i_neg) +
                       f" {unit} (Mdot down to {np.min(Mdot_scan[i_neg]):.3e} Msun/yr). This happens when "
                       f"nu*Sigma rises outward at the inner edge (viscous factor k = 1 - 2p < 0, see the "
                       f"comment above _SOLVABLE_PARAMS), e.g. a steep inner temperature drop near the "
                       f"Tmax = {config['eos'].get('Tmax')} K cap / dust sublimation, and the wind term "
                       f"(psi_DW = {float(params_at(guess)['psi_DW']):.3g}) is too weak to compensate.")
        if len(i_fail):
            msg.append(f"  At {len(i_fail)} scan point(s) the forward model FAILED (exception / non-finite "
                       f"Mdot): {name} = " + ", ".join(f"{from_x(xs[i]):.3g}" for i in i_fail) + f" {unit}.")

        # (a) sign change between two usable scan points separated by failed (nan) points
        gaps = [(fi[m], fi[m + 1]) for m in range(len(fi) - 1)
                if fi[m + 1] > fi[m] + 1 and gs[fi[m]] * gs[fi[m + 1]] < 0]
        if gaps:
            a, b = gaps[0]
            msg.append(f"  Mdot crosses the target between {name} = {from_x(xs[a]):.4g} and "
                       f"{from_x(xs[b]):.4g} {unit}, but the {b - a - 1} scan point(s) in between are "
                       f"unusable (Mdot <= 0 or model failed, see above). The root is probably in that "
                       f"region; try a narrower bracket there or a larger n_scan.")
            fix = "narrow calibration.bracket around that region or increase n_scan."
        else:
            # g = log10(Mdot/Mdot_target) > 0 everywhere means every Mdot EXCEEDS the target,
            # i.e. the target lies BELOW the reachable range (and vice versa)
            side = 'BELOW' if np.nanmin(gs) > 0 else 'ABOVE'
            msg.append(f"  The target Mdot is {side} every positive Mdot reached in the bracket.")
            i_best = fi[np.argmin(np.abs(gs[fi]))]              # scan point with Mdot closest to target
            # (Hof) (b) needs the closest point to be an actual END of the bracket (scan index
            # 0 or n_scan - 1); being merely the last USABLE point is case (d) below
            if i_best in (0, n_scan - 1) and len(fi) >= 2:
                # (b) closest at a bracket end: extrapolate outwards from that end
                at_lo = (i_best == fi[0])
                i_nb = fi[1] if at_lo else fi[-2]                # usable neighbour, one step inwards
                slope = (gs[i_nb] - gs[i_best]) / (xs[i_nb] - xs[i_best])   # dex of Mdot per unit x
                edge = 'LOWER' if at_lo else 'UPPER'
                slope_lbl = (f"d log10(Mdot) / d log10({name})" if scale == 'log'
                             else f"d log10(Mdot) / d {name}")
                msg.append(f"  Mdot is closest to the target at the {edge} end of the bracket "
                           f"({name} = {from_x(xs[i_best]):.4g} {unit}, Mdot = "
                           f"{Mdot_target * 10 ** gs[i_best]:.3e} Msun/yr), so the solution most likely lies "
                           f"OUTSIDE the bracket, {'below' if at_lo else 'above'} {from_x(xs[i_best]):.4g}. "
                           f"Local sensitivity {slope_lbl} = {slope:.3g}.")
                x_est = xs[i_best] - gs[i_best] / slope if slope != 0 else np.inf   # linear extrapolation of g to 0
                with np.errstate(over='ignore'):
                    v_est = from_x(x_est)                        # parameter's own units (inf if it overflows)
                if not np.isfinite(v_est):
                    msg.append(f"  Mdot is (numerically) flat in {name} at that end: no finite extrapolated "
                               f"root, the target is probably unreachable for ANY {name}.")
                else:
                    width = abs(xs[-1] - xs[0])                  # bracket width in x
                    msg.append(f"  Extrapolated root: {name} ~ {v_est:.4g} {unit} "
                               f"({'log-' if scale == 'log' else ''}linear extrapolation; rough estimate only).")
                    # Mdot flattening out: weak power-law sensitivity (< 0.1, cf. psi_DW with total
                    # alpha fixed) or the estimate is > 3 bracket widths away -> likely saturation
                    if (scale == 'log' and abs(slope) < 0.1) or abs(x_est - xs[i_best]) > 3 * width:
                        msg.append(f"  WARNING: Mdot is only weakly sensitive to {name} here and may "
                                   f"saturate, so the extrapolation is unreliable and the target may be "
                                   f"unreachable for ANY {name}.")
                    # compare to the registry range, which also encodes physical limits
                    lo_lim = lo_def
                    hi_lim = hi_def if hi_def is not None else float(grid.Rc[-1])
                    if not lo_lim <= v_est <= hi_lim:
                        msg.append(f"  NOTE: the estimate is outside the default/physical range "
                                   f"[{lo_lim:.4g}, {hi_lim:.4g}] {unit} for {name} (see _SOLVABLE_PARAMS); "
                                   f"change another parameter rather than widening the bracket.")
                    else:
                        # suggest a bracket that includes the estimate with some margin
                        pad = 0.5 if scale == 'log' else 0.1 * width   # 0.5 dex, or 10% of the bracket
                        new_lo = max(from_x(min(x_est - pad, to_x(lo))), lo_lim)
                        new_hi = min(from_x(max(x_est + pad, to_x(hi))), hi_lim)
                        msg.append(f"  Suggested: \"bracket\": [{new_lo:.4g}, {new_hi:.4g}]   {unit}")
                # (Hof) only point to a suggested bracket if one was actually printed
                fix = ("widen calibration.bracket as suggested, or change another parameter / the target Mdot."
                       if msg[-1].startswith("  Suggested:")
                       else "change another (fixed) parameter or the target Mdot.")
            elif i_best in (fi[0], fi[-1]):
                # (d) closest at the last usable point, but the bracket continues beyond it with
                # Mdot <= 0 / failed points: Mdot turns over inside the bracket
                msg.append(f"  Mdot is closest to the target at {name} = {from_x(xs[i_best]):.4g} "
                           f"{unit} (Mdot = {Mdot_target * 10 ** gs[i_best]:.3e} Msun/yr); beyond it, towards "
                           f"the {'lower' if i_best == fi[0] else 'upper'} bracket end, Mdot is <= 0 or the "
                           f"model failed. Mdot turns over INSIDE the bracket, so widening it will not help: "
                           f"the target is probably unreachable for any {name} with the other parameters fixed.")
                fix = "change another (fixed) parameter or the target Mdot."
            else:
                # (c) closest approach at an interior point: Mdot turns around inside the bracket
                msg.append(f"  Mdot comes closest to the target INSIDE the bracket ({name} = "
                           f"{from_x(xs[i_best]):.4g} {unit}, Mdot = {Mdot_target * 10 ** gs[i_best]:.3e} "
                           f"Msun/yr) and moves away from it on both sides: Mdot turns over, so widening the "
                           f"bracket will not help; the target is probably unreachable for any {name} "
                           f"with the other parameters fixed.")
                fix = "change another (fixed) parameter or the target Mdot."
        # the fixed-total-alpha psi solve is known to be badly constrained (see equations above
        # _SOLVABLE_PARAMS): Mdot varies by only ~x2 over psi = 1e-3..1e3
        if name == 'psi_DW' and not hold_SS:
            msg.append("  HINT: with total alpha fixed (hold = 'alpha'), Mdot depends only weakly and "
                       "boundedly on psi_DW; consider removing \"hold\": \"alpha\" from the calibration "
                       "section (the default for psi_DW is hold = 'alpha_SS').")
        # (Hof) the suggested fix now depends on the case above, instead of always
        # "widen the bracket" (which contradicted cases (c)/(d))
        msg.append(f"  Fix: {fix}")
        raise RuntimeError("\n".join(msg))

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
    n_brent = 0                                                  # (Hof) Brent iterations (0 = exact scan hit)
    if kind == 'exact':
        x_root = xs[j]                                           # scan landed exactly on the root
    else:
        # (Hof, 2026-09-29) full_output=True returns a RootResults object (iteration count);
        # a failure inside Brent (no convergence in maxiter, or the forward model raising at a
        # trial point) is re-raised with the interval and the parameter named.
        try:
            with np.errstate(all='ignore'):
                x_root, brent_res = scipy_brentq(g, xs[j], xs[j + 1], xtol=1e-12, maxiter=200,
                                                 full_output=True)
        except (RuntimeError, ValueError, FloatingPointError, ZeroDivisionError) as err:
            raise RuntimeError(
                f"calibration FAILED: Brent's method did not converge for {name} inside "
                f"[{from_x(xs[j]):.4g}, {from_x(xs[j + 1]):.4g}] {unit} (the scan found a sign change "
                f"there). Underlying error: {err}"
            ) from err
        n_brent = int(brent_res.iterations)
    value = float(from_x(x_root))                                # parameter's own units

    # ---- 4. rebuild the disc at the root and check the residual ----
    q = params_at(value)              # (Hof) consistent total alpha at the root when hold = 'alpha_SS'
    res = evaluate_initial_disc(grid, star, q, config['eos'], kappa)
    relerr = abs(res['Mdot'] / Mdot_target - 1)                  # dimensionless
    if relerr > rtol:
        raise RuntimeError(f"calibration FAILED: Brent converged to {name} = {value:.6g} {unit} but "
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
        'hold': hold,                           # (Hof) 'alpha' (total alpha fixed) or 'alpha_SS' (viscous alpha fixed)
        'alpha_SS_fixed': float(alpha_SS_fixed),  # (Hof) dimensionless; NaN when hold = 'alpha'
    }
    # ---- (Hof, 2026-09-29) success summary ----
    # Local sensitivity of Mdot to the parameter at the root, from the two scan points that
    # bracket it: d log10(Mdot) / d x (x = log10(value) for 'log' parameters). Small values
    # mean the parameter is poorly constrained by Mdot (e.g. psi_DW with total alpha fixed,
    # ~0.05): a small error in Mdot or in the other parameters moves the root a lot.
    if kind == 'interval':
        a, b = xs[j], xs[j + 1]                                  # scan interval containing the root, in x
        sens = (gs[j + 1] - gs[j]) / (b - a)                     # dex of Mdot per unit x
    else:
        a = b = xs[j]
        sens = np.nan                                            # not computed for an exact scan hit
    sens_lbl = f"d log10(Mdot)/d log10({name})" if scale == 'log' else f"d log10(Mdot)/d {name}"
    print(f"Calibration CONVERGED: {name} = {value:.6g} {unit}  (guess {guess:.4g}, "
          f"bracket [{lo:.4g}, {hi:.4g}])")
    print(f"    Mdot = {res['Mdot']:.4e} Msun/yr vs target {Mdot_target:.4e} Msun/yr "
          f"(|rel. err| {relerr:.1e} <= rtol {rtol:.1e})")
    print(f"    root interval [{from_x(a):.4g}, {from_x(b):.4g}] {unit}; "
          f"{'exact scan hit' if kind == 'exact' else f'{n_brent} Brent iterations'}; "
          f"{info['n_evals']} forward-model evaluations; {n_roots} candidate root(s) in bracket")
    print(f"    total alpha = {float(q['alpha']):.4e}, alpha_SS = {res['alpha_SS']:.4e} "
          f"(hold = {hold!r}); local sensitivity {sens_lbl} = {sens:.3g}")
    # warn if Mdot barely responds to the parameter at the root (ill-conditioned solve)
    if np.isfinite(sens) and scale == 'log' and abs(sens) < 0.1:
        print(f"    WARNING: |{sens_lbl}| < 0.1 -- {name} is weakly constrained by Mdot here; a 1% "
              f"change in Mdot shifts {name} by ~{1 / abs(sens):.0f}%.")
    # the root sits in the first/last scan interval: Mdot may re-cross just outside the bracket
    # (interval j spans scan points j..j+1, so the last interval is j = n_scan - 2;
    #  an exact hit uses the scan-point index, whose last value is n_scan - 1)
    if j == 0 or j == (n_scan - 2 if kind == 'interval' else n_scan - 1):
        print(f"    NOTE: the root is in the outermost scan interval of the bracket; other roots "
              f"outside [{lo:.4g}, {hi:.4g}] {unit} would not have been seen.")

    # ---- config copy for the rest of the run, with the solved value written in ----
    run_config = copy.deepcopy(config)
    run_config[section][key] = value
    run_config['disc']['alpha'] = float(q['alpha'])   # numeric total alpha (replaces 'SS' if it was used)
    run_config['calibration_result'] = info           # ends up in the logged config json

    return {'disc': res['disc'], 'eos': res['eos'], 'Sigma': res['Sigma'],
            'alpha': float(q['alpha']), 'alpha_SS': res['alpha_SS'],
            'lambda_DW': res['lambda_DW'], 'run_config': run_config, 'info': info}
