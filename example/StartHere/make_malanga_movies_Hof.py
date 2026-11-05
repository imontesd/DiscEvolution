"""
make_malanga_movies_Hof.py
==========================

(Hof Co-op project, 2026-10-21) Two GIF movies of the Malanga et al. (2026) comparison population
(run_popsynth_malanga_full_Hof.sh output): all discs evolving together from --t-start (default
0.06 Myr = 60 000 yr) to --t-end (default 3.0 Myr, the end of the runs).

Movie 1: log M - log Mdot_*  (animated version of Malanga et al. 2026, Fig. 4b; notebook cell 3/3b)
    * points coloured by the disc's psi_DW = alpha_tot / alpha_SS - 1 (dimensionless; the colour of
      Malanga's Fig. 4b, the disc age, is the same for every disc at a common time, so it is replaced by
      psi as in notebook cell 3b);
    * two viscous isochrones (Malanga Eq. 9 with gamma = 1, t >> t_nu):
        solid  M / Mdot = 2 t_Lupus, t_Lupus = 1.6 Myr fixed (Malanga Table 2 / Sec. 4.3; Fig. 4b's line)
        dashed M / Mdot = 2 t       at the CURRENT time t of the frame (moves with the clock).
Movie 2: log Mdot_* (x) - log Mdot_wind (y)   (the plane of Balogh et al. 2026, Fig. 3)
    * Mdot_wind is not stored; it is computed from mass conservation over a +-DT_WIN window:
        Mdot_wind(t) = -dM/dt - <Mdot_*>,   dM/dt = [M(t+dt) - M(t-dt)] / (2 dt)   [Msun/yr]
      (caveats: dust drifting through the inner edge faster than the gas, and -- for discs with R0
      beyond the 250 AU grid -- gas leaving through the outer boundary, are counted as "wind");
    * colour = psi_DW (same scale as movie 1); spot AREA grows linearly with log M over a FIXED scale
      (log M = -6 -> smallest, 0 -> largest; disc mass spans ~5 decades, so a linear-in-M size is useless);
    * reference lines of constant f_M = Mdot_wind / Mdot_*: f_M = 1 (dashed) and Malanga's
      log f_M = -0.6 (+0.4, -0.3) (Table 2: solid line + shaded band);
    * discs with R0 beyond the grid are drawn HOLLOW (their Mdot_wind includes outer-edge loss);
      moments with Mdot_wind <= 0 cannot be placed on a log axis and are counted in the frame text.
Both movies:
    * log-spaced frame times (early evolution is fast);
    * FIXED axes (from all discs at all frame times), so motion is real motion;
    * GI-flagged discs (initial Toomre Q < 1.5, Balogh et al. 2026 Eq. 48; HDF5 attr GI_flag, or
      recomputed from the t = 0 snapshot for older files) are drawn as triangles, others as circles;
      --exclude-gi drops them entirely (output names get "_noGI");
    * discs below Malanga's detection thresholds (M < 1e-6 Msun or Mdot_* < 1e-12 Msun/yr; Sec. 3, p. 5)
      fade to small grey hollow points; the frame text gives the live number of detected discs.

Usage (with the project venv; run where the data are, e.g. on Nibi):
    # (Hof, 2026-11-03) NOTE: this example folder is the 1 Msun population (every M* = 1 Msun). The movie is drawn
    # from whatever runs are in --data-dir, so for the per-disc-M* population v4 use its own folder instead
    # (.../popsynth/malanga_mstar_aSS1e-5_v4, see the population v4 example further below).
    python make_malanga_movies_Hof.py --data-dir /home/imontesd/projects/def-mbalogh/imontesd/output/DiscEvolution/popsynth/malanga/full
    options: --out-dir DIR (default: DATA_DIR/movies)  --t-start 0.06  --t-end 3.0  [Myr]
             --n-frames 150  --fps 15  --dt-win 0.02 [Myr]  --dpi 100  --exclude-gi
Outputs: <out-dir>/malanga_M_Mdot[_noGI].gif and <out-dir>/malanga_Mdotwind_Mdot[_noGI].gif

(Hof, 2026-11-02) Also works for the per-disc-M* population (population v4, run_popsynth_malanga_mstar_full_Hof.sh):
    * the population CSV may be malanga_draws_Hof_*.csv (1 Msun) OR malanga_mstar_draws_Hof_*.csv (per-disc M*);
    * "R0 beyond the grid" (movie 2, hollow spots) now compares R0 with EACH disc's own outer cell [AU] (per-disc
      grids since population v3), not with the first file's;
    * --psi-scale auto|linear|log (default auto): psi colour scale. auto = log when every psi > 0 and they span more
      than 2 decades (population v4: psi ~ 1-900, since psi grows ~ M*^2), otherwise linear as before (1 Msun population);
    * --movie both|M_Mdot|Mdotwind (default both): which GIF(s) to write;
    * the titles say "per-disc M*" when the CSV's Mstar_Msun is not the same for every disc.
    Full time series of population v4 in the Mdot_* - M plane, coloured by psi (run on Nibi, where the data are):
        python make_malanga_movies_Hof.py --movie M_Mdot --t-start 0.001 --n-frames 200 \
            --data-dir /scratch/imontesd/output/DiscEvolution/popsynth/malanga_mstar_aSS1e-5_v4
    t-start must be > 0 (frames are log-spaced); 0.001 Myr is the first snapshot after t = 0. Caveat: the Mdot_*
    diagnostic needs a few kyr to settle (it can start ~2x off, or negative), so the first frames are a transient.

(Hof, 2026-11-05) Movie 3: log M (x) - log10 f_M (y), with histograms of both on the axes (notebook cell 6i, animated)
    * f_M = Mdot_wind / Mdot_* [dimensionless], the INSTANTANEOUS ratio of the total wind mass-loss rate to the accretion rate
      (Tabone et al. 2022, Eq. 22, p. 5; the f_M of Malanga Eq. 6, p. 5). Unlike movie 2, both rates come from the solver's
      own flux functions (HybridWindModel, the same call as run_model_Hof.build_transport) applied to the stored Sigma_G and T
      profiles, so outer-edge loss and dust drift are NOT counted as wind:
        Mdot_wind = sum over cells of 2 pi R dR s_wind,  s_wind = v_DW Sigma_G / (2 (lambda - 1) R), v_DW = 1.5 psi nu / R
                    (Tabone Eq. 9, p. 3; lambda from Balogh et al. 2026, Eq. 16, p. 5)
        Mdot_*    = 2 pi r_min F[0], the mass flux through the inner edge
    * profiles exist only at the SNAPSHOTS, so the frames ARE the snapshots (0, 0.001, 0.01, 0.05, 0.1, 0.2, ..., 3.0 Myr; NOT
      evenly spaced; --t-start, --n-frames and --dt-win do not apply). No interpolation in time: every frame is exact;
    * top histogram: log M of the detected discs (bars) and of the first frame (dotted outline, fixed);
      right histogram: log10 f_M of the detected discs (bars); solid line = current median; red dashed line = Malanga's
      log f_M = -0.6 (Table 2, p. 7); black outline = the same discs' EXPECTED log10[(R0/r_min)^xi - 1] (CSV column
      f_M_grid_expected). Malanga's f_M assumes the wind starts at R_in = 0.03 au, our grid at r_min >= 0.03 au, so this
      expected value is <= 0.251 and the right comparison for each disc. Bins and axes are FIXED over the movie;
    * frame text: median log10 f_M (16-84 %), its offset from -0.6, and the median per-disc offset from the expected value.
      The same numbers are printed per frame and written to <out-dir>/malanga_fM_M[_noGI]_medians.csv;
    * dispersed discs (Malanga p. 5: M < 1e-6 Msun or Mdot_* < 1e-12 Msun/yr, with Mdot_* from the disk_Mdot_star series; at
      t = 0 only M is tested, as in notebook cell 6) are not drawn, and are left out of both histograms; points outside the
      fixed f_M axis are counted in the frame text;
    * colour = psi, GI triangles and --exclude-gi as in movies 1-2; --fm-fps (default 3) sets its speed (34 frames ~ 11 s);
    * needs the HDF5 attrs grid_rmin_au, grid_rmax_au (population v3 and later) to rebuild each disc's grid.
    python make_malanga_movies_Hof.py --movie fM_M \
        --data-dir /scratch/imontesd/output/DiscEvolution/popsynth/malanga_mstar_aSS1e-5_v4
    --movie all writes the three GIFs; --movie both still means movies 1 and 2 only.

Units (see CLAUDE.md): HDF5 't' [yr], 'disk_Mass' [g] (/ Msun), 'disk_Mdot_star' [Msun/yr], 'R' [AU],
'T' [K], 'Sigma_G' [g/cm^2]. Only the scalar time series (written every 5 steps) are used for M and
Mdot, not the 34 snapshots, so the movie has the full time resolution.
"""

import os
import re
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import h5py
import matplotlib
matplotlib.use("Agg")                                   # no display needed (cluster / batch)
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from DiscEvolution.constants import Msun, AU, Omega0, GasConst   # g, cm, s^-1, erg/mol/K
# (Hof, 2026-11-05) movie 3 only: rebuild each disc's grid and transport model to get the wind and accretion rates
from types import SimpleNamespace
from DiscEvolution.constants import yr                           # code time units per yr (2 pi)
from DiscEvolution.grid import Grid
from DiscEvolution.viscous_evolution import HybridWindModel

# ---- physics / reference values (sources in the module docstring) ----
LUPUS_MEAN_AGE_MYR = 1.6         # Myr, Malanga Table 2 (MHD winds) mean age = t_Lupus of their Fig. 4b
M_DETECT_MSUN = 1e-6             # Msun,    Malanga Sec. 3 detection threshold on M
MDOT_DETECT_MSUN_YR = 1e-12      # Msun/yr, Malanga Sec. 3 detection threshold on Mdot
LOG_FM_MALANGA, LOG_FM_UP, LOG_FM_DOWN = -0.6, 0.4, 0.3   # Malanga Table 2: log f_M and 16-84% errors [dex]
MU_EOS = 2.4                     # IrradiatedEOS default mean molecular weight (GI recompute only)
LOGM_SIZE_RANGE = (-6.0, 0.0)    # log M [Msun] mapped linearly onto spot AREA (movie 2)
AREA_RANGE = (6.0, 220.0)        # points^2, spot area at the two ends of LOGM_SIZE_RANGE
N_WIN = 41                       # points used to average Mdot_* over the +-dt window
# (Hof, 2026-11-05) movie 3 (log M - log10 f_M with histograms)
FM_AXIS_PCT = (0.5, 99.5)        # %, the fixed log10 f_M axis covers these percentiles of all detected (disc, frame) values
N_BINS_FM = 32                   # number of log10 f_M bins over that axis (right histogram)
BIN_LOGM = 0.25                  # dex, log M bin width (top histogram)
HIST_FILL = "#86b6ef"            # histogram bars (light blue: kept apart from the psi purples of the points)
TARGET_RED = "red"               # Malanga log f_M = -0.6 reference line (same colour as notebook cells 3a and 6i)

# ---- colours (one-hue sequential for psi; text in ink, never in series colour) ----
INK, MUTED, GRID, BAND, FADED = "#0b0b0b", "#52514e", "#d9d8d4", "#e7e6e2", "#b9b8b3"
PSI_CMAP = mcolors.LinearSegmentedColormap.from_list("psi", plt.cm.Purples(np.linspace(0.25, 1.0, 256)))


# ============================================================================
# Loading
# ============================================================================

def gi_flag_of(h, d):
    """GI flag of one open file: HDF5 attr if present, else recomputed from the t = 0 snapshot with the
    same function the model uses (disc_setup_Hof.gi_check_toomre; Toomre Q < 1.5, Balogh+26 Eq. 48)."""
    if "GI_flag" in h.attrs:
        return bool(h.attrs["GI_flag"])
    if "Sigma_G" not in h or not len(h["Sigma_G"]) or not np.isclose(h["time_snap"][0], 0.0):
        return False
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from disc_setup_Hof import gi_check_toomre
    R = h["R"][:]                                                            # AU
    cs = (1 / Omega0 / AU) * np.sqrt(GasConst / MU_EOS) * np.sqrt(h["T"][0])   # AU per code-time unit
    Om = np.sqrt(d.Mstar_Msun / R ** 3)                                      # per code-time unit
    return bool(gi_check_toomre(R, h["Sigma_G"][0], cs, Om)["GI_flag"])


def read_population_csv(data_dir):
    """(Hof, 2026-11-05) The population CSV next to the runs, indexed by disc id (shared by both loaders).
    1 Msun population (malanga_draws_Hof_*) OR per-disc-M* population (malanga_mstar_draws_Hof_*)."""
    csvs = (glob.glob(os.path.join(data_dir, "malanga_draws_Hof_*.csv"))
            + glob.glob(os.path.join(data_dir, "malanga_mstar_draws_Hof_*.csv")))
    if len(csvs) != 1:
        sys.exit(f"ERROR: expected exactly one population CSV (malanga_[mstar_]draws_Hof_*.csv) in {data_dir}, found {csvs}")
    return pd.read_csv(csvs[0]).set_index("id")


def load_population(data_dir, t_frames_yr, dt_win_yr):
    """Interpolate every disc onto the frame times (yr). Returns (meta DataFrame, M, Mdot, Mdot_wind),
    the three arrays of shape (n_discs, n_frames) in Msun, Msun/yr, Msun/yr. Each file is reduced to the
    frame grid as soon as it is read, so memory stays small for 500 long time series."""
    # (Hof, 2026-11-02) 1 Msun population (malanga_draws_Hof_*) OR per-disc-M* population (malanga_mstar_draws_Hof_*)
    draws = read_population_csv(data_dir)                                    # (2026-11-05) moved into a shared helper
    files = sorted(glob.glob(os.path.join(data_dir, "hof_malanga_w0_*.h5")))
    if not files:
        sys.exit(f"ERROR: no hof_malanga_w0_*.h5 files in {data_dir}")
    # (Hof, 2026-11-02) the outer grid cell is now read PER DISC below: since population v3 every disc has its own
    # grid (r_max = min(20 R0, 1000 au)), so the first file's grid no longer applies to the others

    meta, Ms, Mds, Mws = [], [], [], []
    off = np.linspace(-1.0, 1.0, N_WIN)                                       # window offsets (x dt)
    for k, f in enumerate(files):
        disc_id = int(re.search(r"hof_malanga_w0_(\d+)_", os.path.basename(f)).group(1))
        d = draws.loc[disc_id]
        with h5py.File(f, "r") as h:
            t = h["t"][:]                                                    # yr
            M = h["disk_Mass"][:] / Msun                                     # Msun
            Md = h["disk_Mdot_star"][:]                                      # Msun / yr
            gi = gi_flag_of(h, d)
            # end of the run [yr]: the last SNAPSHOT is written exactly at the final time, whereas the
            # time series (written every 5 steps) can stop a few steps earlier
            t_end_run = max(float(t[-1]) if len(t) else 0.0,
                            float(h["time_snap"][-1]) * 1e6 if "time_snap" in h and len(h["time_snap"]) else 0.0)
            R_grid_max = float(h["R"][-1])                                   # AU, this disc's outer grid cell
        nan = np.full(len(t_frames_yr), np.nan)
        if len(t) < 2:                                                       # no usable time series
            Ms.append(nan); Mds.append(nan); Mws.append(nan)
        else:
            tf = t_frames_yr
            inside = (tf >= t[0]) & (tf <= t_end_run * (1 + 1e-9))           # frame inside the run
            # (frames between t[-1] and t_end_run use the last time-series values: np.interp clamps)
            Mf, Mdf = np.interp(tf, t, M), np.interp(tf, t, Md)
            # Mdot_wind = -dM/dt - <Mdot_*> over [t - dt, t + dt], window clipped to the run
            lo, hi = np.clip(tf - dt_win_yr, t[0], t[-1]), np.clip(tf + dt_win_yr, t[0], t[-1])
            span = np.where(hi > lo, hi - lo, np.nan)
            tw = lo[:, None] + (hi - lo)[:, None] * (off[None, :] + 1) / 2   # (n_frames, N_WIN) yr
            Md_mean = np.trapezoid(np.interp(tw, t, Md), tw, axis=1) / span   # Msun / yr
            Mw = -(np.interp(hi, t, M) - np.interp(lo, t, M)) / span - Md_mean
            Ms.append(np.where(inside, Mf, np.nan)); Mds.append(np.where(inside, Mdf, np.nan))
            Mws.append(np.where(inside, Mw, np.nan))
        meta.append({"id": disc_id, "psi_DW": d.psi_DW, "R0_au": d.R0_au, "GI_flag": gi,
                     "Mstar_Msun": float(d.get("Mstar_Msun", 1.0)),           # Msun (1 Msun population: 1)
                     "beyond_grid": bool(d.R0_au > R_grid_max)})
        if (k + 1) % 50 == 0 or k + 1 == len(files):
            print(f"  loaded {k + 1}/{len(files)} discs", flush=True)
    return pd.DataFrame(meta), np.array(Ms), np.array(Mds), np.array(Mws)


def snapshot_rates(R, Sigma_G, T, rmin, rmax, psi, lam, alpha_SS, Mstar):
    """
    (Hof, 2026-11-05) Accretion and wind mass-loss rates [Msun/yr] of ONE disc at each of its snapshots, from the solver's own
    flux functions (same method as notebook section 5, channel_rates_p3, and cell 6i). Returns an (n_snap, 2) array with the
    columns Mdot_* (flux through r_min) and Mdot_wind (sum of the wind sink over the cells).
      R [AU] cell centres; Sigma_G [g/cm^2] and T [K] with one row per snapshot; rmin, rmax [AU] grid edges (HDF5 attrs);
      psi, lam [dimensionless] (CSV psi_DW, lambda_DW); alpha_SS [dimensionless] the VISCOUS alpha; Mstar [Msun].
    nu = alpha_SS c_s H exactly as IrradiatedEOS computes it (MU_EOS = its default mean molecular weight).
    """
    grid = Grid(rmin, rmax, len(R), spacing="natural")                       # same call as disc_setup_Hof
    if not np.allclose(grid.Rc, R, rtol=1e-8):
        raise ValueError("rebuilt grid does not match the R stored in the file")
    gas = HybridWindModel(psi, lam, boundary="Zero")                         # same call as run_model_Hof.build_transport
    gas._setup_grid(grid)
    cs0 = (1 / Omega0 / AU) * np.sqrt(GasConst / MU_EOS)                     # code units per K^0.5 (IrradiatedEOS._cs0)
    H0 = cs0 / np.sqrt(Mstar)                                                # IrradiatedEOS._H0
    conv = AU * AU * yr / Msun                                               # [g/cm^2] AU^2 per code time -> Msun/yr
    disc = SimpleNamespace(R=R)
    out = np.full((len(Sigma_G), 2), np.nan)
    for j, (S, Tj) in enumerate(zip(Sigma_G, T)):
        disc.Sigma_G = S
        disc.nu = alpha_SS * (cs0 * np.sqrt(Tj)) * (H0 * np.sqrt(Tj * R ** 3))   # alpha_SS c_s H [code units]
        gas._init_fluxes_visc(disc)
        gas._init_fluxes_wind(disc)
        F = gas._f_visc + gas._f_wind                                        # > 0 at r_min: inflow onto the star
        out[j] = [2 * np.pi * gas._dA[0] * F[0] * conv, np.sum(2 * np.pi * gas._dV * gas._s_wind) * conv]
    return out


def load_fM_population(data_dir, t_end_myr):
    """
    (Hof, 2026-11-05) Movie 3 data, one value per disc and SNAPSHOT (the frames). Returns
      meta     DataFrame: id, psi_DW, GI_flag, Mstar_Msun, f_M_grid_expected [dimensionless]
      t_snap   (n_snap,) Myr, the snapshot times up to t_end_myr (those of the longest run)
      M        (n_discs, n_snap) Msun,    disc mass, from the fine disk_Mass series at the snapshot times
      Md_ser   (n_discs, n_snap) Msun/yr, the disk_Mdot_star series at the snapshot times (detection cut only)
      f_M      (n_discs, n_snap) dimensionless, Mdot_wind / Mdot_* from snapshot_rates (NaN where Mdot_* <= 0)
    NaN after a run's last snapshot.
    """
    draws = read_population_csv(data_dir)
    files = sorted(glob.glob(os.path.join(data_dir, "hof_malanga_w0_*.h5")))
    if not files:
        sys.exit(f"ERROR: no hof_malanga_w0_*.h5 files in {data_dir}")
    meta, per_disc = [], []
    for k, f in enumerate(files):
        disc_id = int(re.search(r"hof_malanga_w0_(\d+)_", os.path.basename(f)).group(1))
        d = draws.loc[disc_id]
        with h5py.File(f, "r") as h:
            a = h.attrs
            if "grid_rmin_au" not in a or "grid_rmax_au" not in a:
                sys.exit(f"ERROR: {os.path.basename(f)} has no grid_rmin_au / grid_rmax_au attrs (needed to rebuild the grid "
                         "for movie 3; population v3 and later have them)")
            t, M, Md = h["t"][:], h["disk_Mass"][:] / Msun, h["disk_Mdot_star"][:]   # yr, Msun, Msun/yr
            n_prof = h["Sigma_G"].shape[0]                                   # snapshots actually written
            ts = h["time_snap"][:n_prof]                                     # Myr
            Sig, T, R = h["Sigma_G"][:n_prof], h["T"][:n_prof], h["R"][:]    # g/cm^2, K, AU
            rmin, rmax = float(a["grid_rmin_au"]), float(a["grid_rmax_au"])  # AU
            alpha_SS = float(a.get("alpha_SS", d.get("alpha_SS", np.nan)))   # the run's own viscous alpha
            gi = gi_flag_of(h, d)
        n = min(len(t), len(M), len(Md))                                     # a killed run can leave unequal lengths
        t, M, Md = t[:n], M[:n], Md[:n]
        Mstar = float(d.get("Mstar_Msun", 1.0))                              # Msun (1 Msun population: 1)
        # snapshots covered by the time series (1e-3 tolerance: the series, written every 5 steps, can end a few steps
        # before the last snapshot of a finished run; np.interp then uses the last value)
        keep = (ts <= t_end_myr * (1 + 1e-9)) & (ts * 1e6 <= (t[-1] if n else -1) * (1 + 1e-3))
        ts, Sig, T = ts[keep], Sig[keep], T[keep]
        rates = snapshot_rates(R, Sig, T, rmin, rmax, d.psi_DW, d.lambda_DW, alpha_SS, Mstar)
        with np.errstate(divide="ignore", invalid="ignore"):
            fM = np.where(rates[:, 0] > 0, rates[:, 1] / rates[:, 0], np.nan)   # dimensionless
        per_disc.append((ts, np.interp(ts * 1e6, t, M), np.interp(ts * 1e6, t, Md), fM))
        meta.append({"id": disc_id, "psi_DW": d.psi_DW, "GI_flag": gi, "Mstar_Msun": Mstar,
                     "f_M_grid_expected": float(d.get("f_M_grid_expected", np.nan))})
        if (k + 1) % 50 == 0 or k + 1 == len(files):
            print(f"  loaded {k + 1}/{len(files)} discs", flush=True)
    # common snapshot grid: the longest run's (every run writes the same snapshot times; checked)
    t_snap = max((p[0] for p in per_disc), key=len)
    arrs = [np.full((len(per_disc), len(t_snap)), np.nan) for _ in range(3)]
    for i, (ts, *vals) in enumerate(per_disc):
        if not np.allclose(ts, t_snap[:len(ts)]):
            sys.exit(f"ERROR: disc {meta[i]['id']} has different snapshot times from the longest run")
        for arr, v in zip(arrs, vals):
            arr[i, :len(ts)] = v
    return pd.DataFrame(meta), t_snap, *arrs


# ============================================================================
# Helpers
# ============================================================================

def safe_log(x):
    """log10 of positive finite values, NaN elsewhere (cannot go on a log axis)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isfinite(x) & (x > 0), np.log10(np.where(x > 0, x, 1.0)), np.nan)


def fixed_limits(*logs, floor=None, pad=0.3):
    """Axis limits covering all finite values in the given log arrays (all frames, all discs), rounded
    out to 0.5 dex; `floor` (log) stops a few vanishing discs from stretching the axis without limit."""
    v = np.concatenate([a[np.isfinite(a)].ravel() for a in logs])
    lo, hi = v.min() - pad, v.max() + pad
    if floor is not None:
        lo = max(lo, floor)
    return np.floor(lo * 2) / 2, np.ceil(hi * 2) / 2


def area_from_logM(logM):
    """Spot area [pt^2], linear in log M over LOGM_SIZE_RANGE (clipped)."""
    f = np.clip((logM - LOGM_SIZE_RANGE[0]) / (LOGM_SIZE_RANGE[1] - LOGM_SIZE_RANGE[0]), 0, 1)
    return AREA_RANGE[0] + f * (AREA_RANGE[1] - AREA_RANGE[0])


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.tick_params(colors=MUTED)


# ============================================================================
# Movies
# ============================================================================

def population_label(meta):
    """(Hof, 2026-11-02) Title fragment: "per-disc M*" if M* differs between discs, else the common M*."""
    ms = meta.Mstar_Msun.to_numpy()
    return r"per-disc $M_*$" if np.ptp(ms) > 0 else rf"$M_* = {ms[0]:g}\,M_\odot$"


PSI_LABEL = r"$\psi_{\rm DW} = \alpha_{\rm tot}/\alpha_{\rm SS} - 1$"


def movie_M_Mdot(meta, M, Md, t_myr, psi_norm, out, fps, dpi):
    """Movie 1: log M - log Mdot_* (animated Malanga Fig. 4b), psi colour, two isochrones."""
    logM, logMd = safe_log(M), safe_log(Md)
    xlim = fixed_limits(logM, floor=np.log10(M_DETECT_MSUN) - 1.5)
    ylim = fixed_limits(logMd, floor=np.log10(MDOT_DETECT_MSUN_YR) - 1.5)
    rgba = PSI_CMAP(psi_norm(meta.psi_DW.to_numpy()))                         # one colour per disc
    gi = meta.GI_flag.to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 7.0), layout="constrained")
    style_axes(ax)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\log(M/{\rm M_\odot})$"); ax.set_ylabel(r"$\log(\dot M_*/{\rm M_\odot\,yr^{-1}})$")
    xs = np.array(xlim)
    ax.plot(xs, xs - np.log10(2 * LUPUS_MEAN_AGE_MYR * 1e6), color=INK, lw=1.5, zorder=1)   # fixed
    iso_now, = ax.plot(xs, xs, color=INK, lw=1.2, ls="--", zorder=1)                        # moving
    faded = ax.scatter([], [], s=12, facecolor="none", edgecolor=FADED, lw=0.8, zorder=2)
    circ = ax.scatter([], [], s=40, marker="o", edgecolor=INK, lw=0.5, zorder=3)
    tri = ax.scatter([], [], s=60, marker="^", edgecolor=INK, lw=0.5, zorder=3)
    # time / count text ABOVE the axes and legend BELOW them: nothing ever hides a disc
    label = ax.text(1.0, 1.01, "", transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=10)
    ax.set_title(rf"DiscEvolution hybrid-wind population ({population_label(meta)}): $\dot M_*$ vs $M$" "\n",
                 loc="left", color=INK, fontsize=11)
    fig.legend(handles=[
        Line2D([], [], color=INK, lw=1.5, label=rf"$M/\dot M = 2\,t_{{\rm Lupus}}$ ({LUPUS_MEAN_AGE_MYR:g} Myr)"),
        Line2D([], [], color=INK, lw=1.2, ls="--", label=r"$M/\dot M = 2\,t$ (current time)"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=7, label="stable (Q ≥ 1.5)"),
        Line2D([], [], marker="^", ls="", mfc="white", mec=INK, ms=8, label="GI-flagged (Q < 1.5)"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=FADED, ms=5, label="below detection (dispersed)")],
        loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=psi_norm, cmap=PSI_CMAP), ax=ax, pad=0.02)
    cb.set_label(PSI_LABEL + ("  (log scale)" if isinstance(psi_norm, mcolors.LogNorm) else ""))

    def update(j):
        x, y = logM[:, j], logMd[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        det = ok & (M[:, j] >= M_DETECT_MSUN) & (Md[:, j] >= MDOT_DETECT_MSUN_YR)
        iso_now.set_ydata(xs - np.log10(2 * t_myr[j] * 1e6))
        faded.set_offsets(np.c_[x[ok & ~det], y[ok & ~det]])
        for art, sel in [(circ, det & ~gi), (tri, det & gi)]:
            art.set_offsets(np.c_[x[sel], y[sel]]); art.set_facecolor(rgba[sel])
        label.set_text(f"t = {t_myr[j]:.3f} Myr  |  detected: {int(det.sum())}/{len(x)}")
        return iso_now, faded, circ, tri, label

    FuncAnimation(fig, update, frames=len(t_myr), blit=False).save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def movie_Mdotwind_Mdot(meta, M, Md, Mw, t_myr, psi_norm, out, fps, dpi):
    """Movie 2: log Mdot_* (x) - log Mdot_wind (y), psi colour, spot area ~ log M, constant-f_M lines."""
    logM, logMd, logMw = safe_log(M), safe_log(Md), safe_log(Mw)
    xlim = fixed_limits(logMd, floor=np.log10(MDOT_DETECT_MSUN_YR) - 1.5)
    ylim = fixed_limits(logMw, floor=np.log10(MDOT_DETECT_MSUN_YR) - 1.5)
    rgba = PSI_CMAP(psi_norm(meta.psi_DW.to_numpy()))
    gi, bg = meta.GI_flag.to_numpy(), meta.beyond_grid.to_numpy()

    fig, ax = plt.subplots(figsize=(8.4, 7.4), layout="constrained")
    style_axes(ax)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\log(\dot M_*/{\rm M_\odot\,yr^{-1}})$")
    ax.set_ylabel(r"$\log(\dot M_{\rm wind}/{\rm M_\odot\,yr^{-1}})$")
    xs = np.array(xlim)
    ax.fill_between(xs, xs + LOG_FM_MALANGA - LOG_FM_DOWN, xs + LOG_FM_MALANGA + LOG_FM_UP, color=BAND,
                    lw=0, zorder=0)
    ax.plot(xs, xs + LOG_FM_MALANGA, color=MUTED, lw=1.3, zorder=1)                 # Malanga f_M
    ax.plot(xs, xs, color=MUTED, lw=1.0, ls="--", zorder=1)                          # f_M = 1
    faded = ax.scatter([], [], s=12, facecolor="none", edgecolor=FADED, lw=0.8, zorder=2)
    arts = {(m, h): ax.scatter([], [], marker=m, lw=0.9 if h else 0.5, zorder=3)
            for m in ("o", "^") for h in (False, True)}                              # (marker, hollow)
    # time / count text ABOVE the axes and legends BELOW them: nothing ever hides a disc
    label = ax.text(1.0, 1.01, "", transform=ax.transAxes, ha="right", va="bottom", color=INK, fontsize=10)
    ax.set_title(r"DiscEvolution hybrid-wind population: wind mass-loss vs stellar accretion" "\n",
                 loc="left", color=INK, fontsize=11)
    size_handles = [Line2D([], [], marker="o", ls="", mfc=FADED, mec=MUTED, ms=np.sqrt(area_from_logM(v)),
                           label=rf"$\log(M/{{\rm M_\odot}}) = {v:g}$") for v in (-4, -2, 0)]
    fig.legend(handles=[
        Line2D([], [], color=MUTED, lw=1.0, ls="--", label=r"$f_M = 1$"),
        Line2D([], [], color=MUTED, lw=1.3, label=rf"Malanga $\log f_M = {LOG_FM_MALANGA}$"),
        Patch(facecolor=BAND, edgecolor="none", label=rf"$+{LOG_FM_UP}/-{LOG_FM_DOWN}$ dex"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=7, label="stable (Q ≥ 1.5)"),
        Line2D([], [], marker="^", ls="", mfc="white", mec=INK, ms=8, label="GI-flagged (Q < 1.5)"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=PSI_CMAP(0.75), mew=1.6, ms=8,
               label=r"hollow: $R_0$ beyond grid"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=FADED, ms=5, label="below detection"),
        *size_handles],                                    # spot size (disc mass) reference sizes
        loc="outside lower center", ncols=4, frameon=False, fontsize=8.5,
        title="lines, markers, and spot size = disc mass", title_fontsize=8.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=psi_norm, cmap=PSI_CMAP), ax=ax, pad=0.02)
    cb.set_label(PSI_LABEL + ("  (log scale)" if isinstance(psi_norm, mcolors.LogNorm) else ""))

    def update(j):
        x, y, lm = logMd[:, j], logMw[:, j], logM[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        det = ok & (M[:, j] >= M_DETECT_MSUN) & (Md[:, j] >= MDOT_DETECT_MSUN_YR)
        faded.set_offsets(np.c_[x[ok & ~det], y[ok & ~det]])
        for (m, hollow), art in arts.items():
            sel = det & (gi if m == "^" else ~gi) & (bg if hollow else ~bg)
            art.set_offsets(np.c_[x[sel], y[sel]])
            art.set_sizes(area_from_logM(lm[sel]) * (1.4 if m == "^" else 1.0))
            if hollow:
                art.set_facecolor("none"); art.set_edgecolor(rgba[sel]); art.set_linewidth(1.6)
            else:
                art.set_facecolor(rgba[sel]); art.set_edgecolor(INK)
        n_neg = int((np.isfinite(Mw[:, j]) & (Mw[:, j] <= 0)).sum())
        label.set_text(f"t = {t_myr[j]:.3f} Myr  |  detected: {int(det.sum())}/{len(x)}"
                       + (f"  |  Mdot_wind ≤ 0 (not shown): {n_neg}" if n_neg else ""))
        return (faded, label, *arts.values())

    FuncAnimation(fig, update, frames=len(t_myr), blit=False).save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def movie_fM_M(meta, t_myr, M, Md, fM, psi_norm, out, fps, dpi):
    """
    (Hof, 2026-11-05) Movie 3: log M (x) - log10 f_M (y) with live histograms of both (module docstring).
    One frame per snapshot. Returns the per-frame summary DataFrame (also written next to the GIF).
    """
    n_disc, n_fr = M.shape
    logM, logf = safe_log(M), safe_log(fM)
    logf_exp = safe_log(meta.f_M_grid_expected.to_numpy())                   # per disc, fixed in time
    # detected (Malanga p. 5) AND f_M defined; at t = 0 only M is tested (the Mdot_* series needs a few kyr to settle)
    with np.errstate(invalid="ignore"):
        det = (np.isfinite(logM) & np.isfinite(logf) & (M >= M_DETECT_MSUN)
               & ((Md >= MDOT_DETECT_MSUN_YR) | (t_myr == 0)[None, :]))
    if not det.any():
        sys.exit("ERROR: movie 3 has no detected disc with a defined f_M at any snapshot")

    # ---- FIXED axes and bins over the whole movie ----
    xlim = fixed_limits(np.where(det, logM, np.nan), pad=0.2)               # log Msun
    xlim = (max(xlim[0], np.log10(M_DETECT_MSUN) - 0.25), xlim[1])
    lo, hi = np.percentile(logf[det], FM_AXIS_PCT)                           # robust: a few wild discs cannot stretch it
    lo = min(lo, LOG_FM_MALANGA, np.nanmin(logf_exp)) - 0.1
    hi = max(hi, LOG_FM_MALANGA, np.nanmax(logf_exp)) + 0.1
    ylim = (np.floor(lo * 10) / 10, np.ceil(hi * 10) / 10)                   # dimensionless (log10 f_M), 0.1 dex steps
    xbins = np.arange(np.floor(xlim[0] / BIN_LOGM) * BIN_LOGM, xlim[1] + BIN_LOGM, BIN_LOGM)
    ybins = np.linspace(*ylim, N_BINS_FM + 1)
    hx = np.array([np.histogram(logM[det[:, j], j], xbins)[0] for j in range(n_fr)])        # counts per frame
    hy = np.array([np.histogram(logf[det[:, j], j], ybins)[0] for j in range(n_fr)])
    hexp = np.array([np.histogram(logf_exp[det[:, j]], ybins)[0] for j in range(n_fr)])     # same discs, expected value
    hx0 = hx[0]                                                              # first frame, fixed outline
    rgba = PSI_CMAP(psi_norm(meta.psi_DW.to_numpy()))
    gi = meta.GI_flag.to_numpy()

    # ---- per-frame summary (frame text, stdout and CSV) ----
    rows = []
    for j in range(n_fr):
        v, e = logf[det[:, j], j], logf_exp[det[:, j]]
        ok = np.isfinite(e)
        rows.append({"t_Myr": t_myr[j], "n_detected": int(det[:, j].sum()),
                     "median_log_fM": np.median(v), "p16_log_fM": np.percentile(v, 16), "p84_log_fM": np.percentile(v, 84),
                     "median_minus_target_dex": np.median(v) - LOG_FM_MALANGA,
                     "median_code_minus_expected_dex": np.median(v[ok] - e[ok]) if ok.any() else np.nan,
                     "n_outside_fM_axis": int(((v < ylim[0]) | (v > ylim[1])).sum())}
                    if v.size else {"t_Myr": t_myr[j], "n_detected": 0})
    summ = pd.DataFrame(rows)

    # ---- layout: top histogram / scatter / right histogram / colour bar; frame text top right ----
    fig = plt.figure(figsize=(10.0, 8.4), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[4.0, 1.4, 0.13], height_ratios=[1.35, 4.0])
    ax = fig.add_subplot(gs[1, 0])
    ax_t = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_r = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_info = fig.add_subplot(gs[0, 1:]); ax_info.axis("off")
    cax = fig.add_subplot(gs[1, 2])
    for a in (ax, ax_t, ax_r):
        style_axes(a)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(r"$\log(M/{\rm M_\odot})$")
    ax.set_ylabel(r"$\log_{10} f_M = \log_{10}(\dot M_{\rm wind}/\dot M_*)$")
    ax_t.tick_params(labelbottom=False); ax_r.tick_params(labelleft=False)
    ax_t.set_ylabel("discs"); ax_r.set_xlabel("discs")
    ax_t.set_ylim(0, 1.08 * max(hx.max(), 1))
    ax_r.set_xlim(0, 1.08 * max(hy.max(), hexp.max(), 1))
    # Malanga's log f_M = -0.6 (Table 2, p. 7): dashed, on the scatter AND on the f_M histogram
    for a in (ax, ax_r):
        a.axhline(LOG_FM_MALANGA, color=TARGET_RED, ls="--", lw=1.3, zorder=4)
    # dynamic histograms (bars) + reference outlines
    bars_t = ax_t.bar(xbins[:-1], hx[0], width=np.diff(xbins), align="edge", color=HIST_FILL, edgecolor="white", lw=0.6)
    ax_t.stairs(hx0, xbins, color=MUTED, lw=1.0, ls=":", zorder=3)
    bars_r = ax_r.barh(ybins[:-1], hy[0], height=np.diff(ybins), align="edge", color=HIST_FILL, edgecolor="white", lw=0.6)
    exp_out = ax_r.stairs(hexp[0], ybins, orientation="horizontal", color=INK, lw=1.0, zorder=3)
    med_line = ax_r.axhline(np.nan, color=INK, lw=1.8, zorder=5)
    # points
    circ = ax.scatter([], [], s=34, marker="o", edgecolor=INK, lw=0.5, zorder=3)
    tri = ax.scatter([], [], s=52, marker="^", edgecolor=INK, lw=0.5, zorder=3)
    info = ax_info.text(0.0, 1.0, "", transform=ax_info.transAxes, ha="left", va="top", color=INK, fontsize=9.5,
                        family="monospace", linespacing=1.5)
    fig.suptitle(rf"DiscEvolution hybrid-wind population ({population_label(meta)}): wind / accretion ratio vs disc mass",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.legend(handles=[
        Line2D([], [], color=TARGET_RED, lw=1.3, ls="--", label=rf"Malanga $\log f_M = {LOG_FM_MALANGA}$ (Table 2)"),
        Line2D([], [], color=INK, lw=1.8, label=r"median $\log_{10} f_M$ (now)"),
        Line2D([], [], color=INK, lw=1.0, label=r"expected, same discs: $\log_{10}[(R_0/r_{\rm min})^\xi-1]$"),
        Line2D([], [], color=MUTED, lw=1.0, ls=":", label=rf"$\log M$ at the first frame ($t$ = {t_myr[0]:g} Myr)"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=7, label="stable (Q ≥ 1.5)"),
        Line2D([], [], marker="^", ls="", mfc="white", mec=INK, ms=8, label="GI-flagged (Q < 1.5)")],
        loc="outside lower center", ncols=3, frameon=False, fontsize=8.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=psi_norm, cmap=PSI_CMAP), cax=cax)
    cb.set_label(PSI_LABEL + ("  (log scale)" if isinstance(psi_norm, mcolors.LogNorm) else ""))

    def update(j):
        x, y, sel_det = logM[:, j], logf[:, j], det[:, j]
        for art, sel in [(circ, sel_det & ~gi), (tri, sel_det & gi)]:
            art.set_offsets(np.c_[x[sel], y[sel]]); art.set_facecolor(rgba[sel])
        for b, h in zip(bars_t, hx[j]):
            b.set_height(h)
        for b, h in zip(bars_r, hy[j]):
            b.set_width(h)
        exp_out.set_data(hexp[j])
        r = summ.iloc[j]
        if r.n_detected:
            med_line.set_ydata([r.median_log_fM] * 2)
            info.set_text(f"t = {t_myr[j]:.3f} Myr  (snapshot {j + 1}/{n_fr})\n"
                          f"detected: {int(r.n_detected)}/{n_disc}\n"
                          f"median log f_M: {r.median_log_fM:+.3f}\n"
                          f"  16-84 %: {r.p16_log_fM:+.3f} to {r.p84_log_fM:+.3f}\n"
                          f"median - ({LOG_FM_MALANGA:g}): {r.median_minus_target_dex:+.3f} dex\n"
                          f"median (code - expected): {r.median_code_minus_expected_dex:+.3f} dex"
                          + (f"\noutside the f_M axis: {int(r.n_outside_fM_axis)}" if r.n_outside_fM_axis else ""))
        else:
            med_line.set_ydata([np.nan] * 2)
            info.set_text(f"t = {t_myr[j]:.3f} Myr  (snapshot {j + 1}/{n_fr})\ndetected: 0/{n_disc}")
        return (circ, tri, exp_out, med_line, info, *bars_t, *bars_r)

    FuncAnimation(fig, update, frames=n_fr, blit=False).save(out, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return summ


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Movies of the Malanga comparison population (see module docstring).")
    ap.add_argument("--data-dir", required=True, help="folder with hof_malanga_w0_*.h5 and the population CSV")
    ap.add_argument("--out-dir", default=None, help="output folder (default: <data-dir>/movies)")
    ap.add_argument("--t-start", type=float, default=0.06, help="first frame time [Myr] (default 0.06)")
    ap.add_argument("--t-end", type=float, default=3.0, help="last frame time [Myr] (default 3.0)")
    ap.add_argument("--n-frames", type=int, default=150, help="number of (log-spaced) frames (default 150)")
    ap.add_argument("--fps", type=int, default=15, help="frames per second (default 15 -> 10 s)")
    ap.add_argument("--dt-win", type=float, default=0.02, help="half-width of the Mdot_wind window [Myr]")
    ap.add_argument("--dpi", type=int, default=100, help="GIF resolution (default 100)")
    ap.add_argument("--exclude-gi", action="store_true", help="drop GI-flagged discs (adds _noGI to names)")
    # (Hof, 2026-11-02) see the module docstring
    ap.add_argument("--psi-scale", choices=["auto", "linear", "log"], default="auto",
                    help="psi colour scale (default auto: log if all psi > 0 and they span > 2 decades)")
    # (Hof, 2026-11-05) fM_M = movie 3; all = movies 1-3; both = movies 1-2 (unchanged meaning)
    ap.add_argument("--movie", choices=["both", "all", "M_Mdot", "Mdotwind", "fM_M"], default="both",
                    help="which GIF(s) to write (both = movies 1-2, all = movies 1-3)")
    ap.add_argument("--fm-fps", type=int, default=3,
                    help="frames per second of movie 3 (default 3: its 34 snapshot frames last ~11 s)")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "movies")
    os.makedirs(out_dir, exist_ok=True)
    # (Hof, 2026-11-05) movies 1-2 use the fine time series on log-spaced frames; movie 3 the snapshot profiles.
    # Each data set is loaded only if one of its movies is requested.
    want_12 = args.movie in ("both", "all", "M_Mdot", "Mdotwind")
    want_3 = args.movie in ("all", "fM_M")
    meta = meta3 = None
    if want_12:
        t_myr = np.logspace(np.log10(args.t_start), np.log10(args.t_end), args.n_frames)   # log-spaced, Myr
        print(f"Loading discs from {args.data_dir} ({args.n_frames} frames, {args.t_start}-{args.t_end} Myr)")
        meta, M, Md, Mw = load_population(args.data_dir, t_myr * 1e6, args.dt_win * 1e6)
    if want_3:
        print(f"Loading snapshot profiles for movie 3 from {args.data_dir} (0-{args.t_end} Myr)")
        meta3, t3_myr, M3, Md3, fM3 = load_fM_population(args.data_dir, args.t_end)

    if args.exclude_gi:
        if want_12:
            keep = ~meta.GI_flag.to_numpy()
            print(f"Excluding {int((~keep).sum())} GI-flagged discs")
            meta, M, Md, Mw = meta[keep].reset_index(drop=True), M[keep], Md[keep], Mw[keep]
        if want_3:
            keep3 = ~meta3.GI_flag.to_numpy()
            meta3, M3, Md3, fM3 = meta3[keep3].reset_index(drop=True), M3[keep3], Md3[keep3], fM3[keep3]
    suffix = "_noGI" if args.exclude_gi else ""
    # (Hof, 2026-11-02) psi colour scale, the same in both movies: linear as before, or log for a population whose
    # psi spans several decades (population v4, psi ~ M*^2: a linear scale would leave almost every disc pale)
    psi_all = (meta if meta is not None else meta3).psi_DW                   # (2026-11-05) same discs in both loaders
    psi_min, psi_max = psi_all.min(), psi_all.max()                          # dimensionless
    use_log = (args.psi_scale == "log"
               or (args.psi_scale == "auto" and psi_min > 0 and psi_max / psi_min > 100))
    if use_log and psi_min <= 0:
        sys.exit("ERROR: --psi-scale log needs every psi > 0 (this population has floored discs, psi = 0)")
    psi_norm = (mcolors.LogNorm(vmin=psi_min, vmax=psi_max) if use_log
                else mcolors.Normalize(vmin=psi_min, vmax=psi_max))
    print(f"psi colour scale: {'log' if use_log else 'linear'}, {psi_min:.3g}-{psi_max:.3g}")

    outs = []
    if args.movie in ("both", "all", "M_Mdot"):
        outs.append(os.path.join(out_dir, f"malanga_M_Mdot{suffix}.gif"))
        print(f"Writing {outs[-1]}"); movie_M_Mdot(meta, M, Md, t_myr, psi_norm, outs[-1], args.fps, args.dpi)
    if args.movie in ("both", "all", "Mdotwind"):
        outs.append(os.path.join(out_dir, f"malanga_Mdotwind_Mdot{suffix}.gif"))
        print(f"Writing {outs[-1]}"); movie_Mdotwind_Mdot(meta, M, Md, Mw, t_myr, psi_norm, outs[-1], args.fps, args.dpi)
    if want_3:                                                               # (2026-11-05) movie 3 + its per-frame summary
        outs.append(os.path.join(out_dir, f"malanga_fM_M{suffix}.gif"))
        print(f"Writing {outs[-1]}")
        summ = movie_fM_M(meta3, t3_myr, M3, Md3, fM3, psi_norm, outs[-1], args.fm_fps, args.dpi)
        csv_out = os.path.join(out_dir, f"malanga_fM_M{suffix}_medians.csv")
        summ.to_csv(csv_out, index=False, float_format="%.5g")
        print("Movie 3, per frame: median log10 f_M, its offset from Malanga's -0.6 and from each disc's expected value [dex]")
        print(summ.to_string(index=False, float_format=lambda v: f"{v:+.3f}" if abs(v) < 10 else f"{v:.3g}"))
        print(f"  {csv_out}")
    for o in outs:
        print(f"  {o}  ({os.path.getsize(o) / 1e6:.1f} MB)")
