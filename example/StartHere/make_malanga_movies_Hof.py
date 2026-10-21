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
    python make_malanga_movies_Hof.py --data-dir /home/imontesd/projects/def-mbalogh/imontesd/output/DiscEvolution/popsynth/malanga/full
    options: --out-dir DIR (default: DATA_DIR/movies)  --t-start 0.06  --t-end 3.0  [Myr]
             --n-frames 150  --fps 15  --dt-win 0.02 [Myr]  --dpi 100  --exclude-gi
Outputs: <out-dir>/malanga_M_Mdot[_noGI].gif and <out-dir>/malanga_Mdotwind_Mdot[_noGI].gif

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

# ---- physics / reference values (sources in the module docstring) ----
LUPUS_MEAN_AGE_MYR = 1.6         # Myr, Malanga Table 2 (MHD winds) mean age = t_Lupus of their Fig. 4b
M_DETECT_MSUN = 1e-6             # Msun,    Malanga Sec. 3 detection threshold on M
MDOT_DETECT_MSUN_YR = 1e-12      # Msun/yr, Malanga Sec. 3 detection threshold on Mdot
LOG_FM_MALANGA, LOG_FM_UP, LOG_FM_DOWN = -0.6, 0.4, 0.3   # Malanga Table 2: log f_M and 16-84% errors [dex]
MU_EOS = 2.4                     # IrradiatedEOS default mean molecular weight (GI recompute only)
LOGM_SIZE_RANGE = (-6.0, 0.0)    # log M [Msun] mapped linearly onto spot AREA (movie 2)
AREA_RANGE = (6.0, 220.0)        # points^2, spot area at the two ends of LOGM_SIZE_RANGE
N_WIN = 41                       # points used to average Mdot_* over the +-dt window

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


def load_population(data_dir, t_frames_yr, dt_win_yr):
    """Interpolate every disc onto the frame times (yr). Returns (meta DataFrame, M, Mdot, Mdot_wind),
    the three arrays of shape (n_discs, n_frames) in Msun, Msun/yr, Msun/yr. Each file is reduced to the
    frame grid as soon as it is read, so memory stays small for 500 long time series."""
    csvs = glob.glob(os.path.join(data_dir, "malanga_draws_Hof_*.csv"))
    if len(csvs) != 1:
        sys.exit(f"ERROR: expected exactly one malanga_draws_Hof_*.csv in {data_dir}, found {csvs}")
    draws = pd.read_csv(csvs[0]).set_index("id")
    files = sorted(glob.glob(os.path.join(data_dir, "hof_malanga_w0_*.h5")))
    if not files:
        sys.exit(f"ERROR: no hof_malanga_w0_*.h5 files in {data_dir}")
    with h5py.File(files[0], "r") as h:
        R_grid_max = float(h["R"][-1])                                       # AU, outer grid cell

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
                     "beyond_grid": bool(d.R0_au > R_grid_max)})
        if (k + 1) % 50 == 0 or k + 1 == len(files):
            print(f"  loaded {k + 1}/{len(files)} discs", flush=True)
    return pd.DataFrame(meta), np.array(Ms), np.array(Mds), np.array(Mws)


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
    ax.set_title(r"DiscEvolution hybrid-wind population ($M_* = 1\,M_\odot$): $\dot M_*$ vs $M$" "\n",
                 loc="left", color=INK, fontsize=11)
    fig.legend(handles=[
        Line2D([], [], color=INK, lw=1.5, label=rf"$M/\dot M = 2\,t_{{\rm Lupus}}$ ({LUPUS_MEAN_AGE_MYR:g} Myr)"),
        Line2D([], [], color=INK, lw=1.2, ls="--", label=r"$M/\dot M = 2\,t$ (current time)"),
        Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=7, label="stable (Q ≥ 1.5)"),
        Line2D([], [], marker="^", ls="", mfc="white", mec=INK, ms=8, label="GI-flagged (Q < 1.5)"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=FADED, ms=5, label="below detection (dispersed)")],
        loc="outside lower center", ncols=3, frameon=False, fontsize=9)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=psi_norm, cmap=PSI_CMAP), ax=ax, pad=0.02)
    cb.set_label(r"$\psi_{\rm DW} = \alpha_{\rm tot}/\alpha_{\rm SS} - 1$")

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
    cb.set_label(r"$\psi_{\rm DW} = \alpha_{\rm tot}/\alpha_{\rm SS} - 1$")

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
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.data_dir, "movies")
    os.makedirs(out_dir, exist_ok=True)
    t_myr = np.logspace(np.log10(args.t_start), np.log10(args.t_end), args.n_frames)   # log-spaced, Myr
    print(f"Loading discs from {args.data_dir} ({args.n_frames} frames, {args.t_start}-{args.t_end} Myr)")
    meta, M, Md, Mw = load_population(args.data_dir, t_myr * 1e6, args.dt_win * 1e6)

    if args.exclude_gi:
        keep = ~meta.GI_flag.to_numpy()
        print(f"Excluding {int((~keep).sum())} GI-flagged discs")
        meta, M, Md, Mw = meta[keep].reset_index(drop=True), M[keep], Md[keep], Mw[keep]
    suffix = "_noGI" if args.exclude_gi else ""
    psi_norm = mcolors.Normalize(vmin=meta.psi_DW.min(), vmax=meta.psi_DW.max())   # same in both movies

    out1 = os.path.join(out_dir, f"malanga_M_Mdot{suffix}.gif")
    print(f"Writing {out1}"); movie_M_Mdot(meta, M, Md, t_myr, psi_norm, out1, args.fps, args.dpi)
    out2 = os.path.join(out_dir, f"malanga_Mdotwind_Mdot{suffix}.gif")
    print(f"Writing {out2}"); movie_Mdotwind_Mdot(meta, M, Md, Mw, t_myr, psi_norm, out2, args.fps, args.dpi)
    for o in (out1, out2):
        print(f"  {o}  ({os.path.getsize(o) / 1e6:.1f} MB)")
