#!/bin/bash
#SBATCH --account=def-mbalogh
#SBATCH --time=1-00:00:00            # (2026-11-12, sim-time step 2) 7 d, was 24 h: the disc 244 test ran ~0.08 s/step at
                                     # dt ~0.93 yr -> 3 Myr ~ 3.2e6 steps ~ 70 h, 5 Myr ~ 5.4e6 steps ~ 5 d (if dt stays
                                     # ~1 yr). Applies to EACH array task. Lower it once the timing CSVs give the real cost.
#SBATCH --array=0-1                  # (2026-11-12, sim-time step 2) task 0 -> T_END_LIST[0] = 3 Myr, task 1 -> [1] = 5 Myr
#SBATCH --cpus-per-task=1            # ONE disc, one serial run_model_Hof.py process
#SBATCH --mem=2G                     # RAM (not disk); the popsynth runs use 500 MB each, planets add little
#SBATCH --job-name=hof_add-planets_pilotv2
#SBATCH --output=/scratch/imontesd/output/DiscEvolution/logs/%x_%A_%a.out   # %A = array job ID, %a = task (0 = 3 Myr, 1 = 5 Myr);
                                                                             # folder must exist BEFORE sbatch
#SBATCH --mail-user=imontesdeocahof@gmail.com
#SBATCH --mail-type=END,FAIL
#
# run_add-planets_pilot_Hof.sh
# ----------------------------
# (Hof, 2026-11-12) Planet-population PILOT run for ONE disc of the v5 population (default: disc 244, pilot v2).
#
# WHY disc 244: pilot v1 (disc 208) sits at the 95th-98th percentile of the v5 population in M*, M0, R0 and psi
# (M* = 0.98 Msun, M0 = 0.077 Msun, R0 = 95 au, psi = 476). Disc 244 is one of the discs closest to the population
# MEDIAN in all four (M* = 0.196 Msun, M0 = 0.0039 Msun, R0 = 5.1 au, psi = 27.5), so it tests what the bulk of the
# population does (Hof_figures_add-planets.ipynb, seeding section; discussion of 2026-11-12).
#
# WHAT IT DOES
#   1. Reads row DISC_ID of the v5 draws CSV (config/malanga/malanga_mstar_draws_Hof_N500_seed1_xirmin.csv).
#   2. Builds the planet seeds exactly as the seeding cell of Hof_figures_add-planets.ipynb: N_P orbital periods
#      log-spaced from P_MIN_YR to P_MAX_YR (default 30 periods, 1 day to 1000 yr), converted with Kepler's third law
#          a [AU] = (M*/Msun)^(1/3) (P/yr)^(2/3)      (= (G M* P^2 / 4 pi^2)^(1/3) in code units, G = 1, yr = 2 pi)
#      and keeps only the seeds INSIDE the grid, r_min <= a <= r_max (a seed outside the grid would sit outside the
#      simulated disc). All seeds get Mp = "SI" (inserted when planetesimals form at their radius) and implant_time 0.
#   3. Writes a per-run config = BASE_CONFIG (default: the pilot v1 config, config/add-planets/
#      DiscConfig_Hof_add-planets-baseline.json, so ONLY the disc and the seeds differ from pilot v1) with:
#          star.M [Msun], star.R [Rsun], star.T_eff [K]       <- Mstar_Msun, Rstar_Rsun, Teff_K
#          grid.rmin, grid.rmax [AU]                          <- rmin_au, rmax_au
#          disc.M [Msun], disc.Rd [AU]                        <- M0_Msun, R0_au
#          winds.psi_DW, winds.e_rad [dimensionless]          <- psi_DW, e_rad
#          calibration.alpha_SS [dimensionless]               <- alpha_SS  (hold = "alpha_SS": TOTAL alpha =
#                                                                alpha_SS (1 + psi), written to disc.alpha for clarity;
#                                                                the code derives it itself and ignores disc.alpha)
#          planets.Rp [AU], planets.Mp, planets.implant_time  <- the seeds of step 2 (all three lists the SAME length)
#          simulation.run_name, simulation.output_dir         <- RUN_NAME, OUTDIR
#      and the stale "calibration_result" block copied into the base config removed (it is recomputed by the run).
#      The config is saved next to the output ($OUTDIR/config_used/) so the run can be reproduced from it alone.
#   4. Runs run_model_Hof.py --config <that file>. The same values are ALSO passed as command-line overrides (--M, --Rd,
#      --psi_DW, --e_rad, --alpha_SS, --Mstar, --Rstar, --Teff, --rmin, --rmax), as run_popsynth_malanga_full_Hof.sh
#      does for the population, so the run log prints every per-disc value ("Overriding ...").
#
# NOTE on the pilot v1 config: its planets.implant_time has 16 entries while planets.Rp / Mp have 20. build_planets()
# pairs them with zip() (run_model_Hof.py, line ~870), which stops at the SHORTEST list, so the 4 outermost pilot v1
# seeds (80.8, 123, 187, 285 au) were silently DROPPED, not "never inserted". This script builds all three lists from
# the same seed array and checks their lengths, so this cannot happen here.
#
# USAGE (Nibi)
#   mkdir -p /scratch/imontesd/output/DiscEvolution/logs      # once: the #SBATCH --output folder must exist
#   sbatch run_add-planets_pilot_Hof.sh                       # disc 244 -> popsynth/add-planets/pilot_v2
#   DISC_ID=387 RUN_TAG=pilotv3 OUTDIR=/scratch/.../pilot_v3 sbatch --export=ALL run_add-planets_pilot_Hof.sh
#   DRY_RUN=1 bash run_add-planets_pilot_Hof.sh               # write + print the config and command, run nothing
#   T_END_MYR=0.001 bash run_add-planets_pilot_Hof.sh         # quick smoke test: stop after 0.001 Myr
#
# Restart safety: run_model_Hof.py skips the run if its output file already exists and is complete.
#
# (Hof, 2026-11-12, sim-time step 2) TIMING TRIAL: the same disc to TWO end times at once
#   * Slurm job ARRAY (--array=0-1): task i runs to T_END_MYR = T_END_LIST[i] (default "3 5" Myr), so the 3 Myr and
#     5 Myr runs start together on separate CPUs. An explicit T_END_MYR in the environment wins over the array.
#   * RUN_TAG gets the suffix "_<T_END>Myr" (pilotv2_3Myr, pilotv2_5Myr): the output filename has no end-time token
#     (output_filename() in run_model_Hof.py), so without it the two runs would write the SAME .h5 file.
#   * Snapshot times: the base config's list (0, 0.001, 0.01, 0.05, then every 0.1 Myr to 3 Myr) is cut at T_END and,
#     beyond its last entry, EXTENDED with the same 0.1 Myr spacing up to T_END. So both runs share every snapshot up
#     to 3 Myr: the timestep is only ever cut to land on a snapshot, hence the two runs take IDENTICAL steps to 3 Myr
#     and differ only in the ETA they predict (t_end = 3 vs 5 Myr) -- the comparison this trial is for.
#   * Timing stream ON (STREAM_TIMING=1): simulation.stream_timing = true and stream_timing_every = STREAM_EVERY
#     (default 1 = every step) are written into the per-run config, so run_model_Hof.py writes
#     <output>_timing.csv (step, t, dt, dt_phys, wall time, ETA; see its "sim-time step 1" change-log entry).
#     ~75 bytes per row: ~240 MB (3 Myr) / ~400 MB (5 Myr) at STREAM_EVERY=1; STREAM_EVERY=10 makes it 10x smaller.
#   Usage:
#     sbatch run_add-planets_pilot_Hof.sh                                   # both runs (array tasks 0 and 1)
#     SLURM_ARRAY_TASK_ID=1 DRY_RUN=1 bash run_add-planets_pilot_Hof.sh     # check the 5 Myr config + command
#     T_END_LIST="0.0003 0.0005" sbatch run_add-planets_pilot_Hof.sh        # short test of the pair
#
# (Hof, 2026-11-13) WALL_LIMIT_HOURS: optional wall-clock limit per run [h], written into the per-run config as
#   simulation.wall_limit_hours (run_model_Hof.py, change log 2026-11-13: the run stops after that many hours, saving
#   all output so far, and sets HDF5 attr stopped_by_wall_limit). Empty (default) = no limit, as for the timing trial.
#   Also used by run_popsynth_add-planets_v5_Hof.sh, which calls THIS script once per disc (DISC_ID, RUN_TAG,
#   T_END_MYR, OUTDIR, BASE_CONFIG, WALL_LIMIT_HOURS, STREAM_TIMING from its environment), so the seed and config
#   logic exists in one place only.

set -euo pipefail

# ---- settings (all overridable from the environment) ----
DISC_ID="${DISC_ID:-244}"                    # "id" column of the draws CSV
RUN_TAG="${RUN_TAG:-pilotv2}"                # goes into run_name -> output filename prefix
SCRIPT_DIR="${SCRIPT_DIR:-/home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere}"
DRAWS_CSV="${DRAWS_CSV:-$SCRIPT_DIR/config/malanga/malanga_mstar_draws_Hof_N500_seed1_xirmin.csv}"
BASE_CONFIG="${BASE_CONFIG:-$SCRIPT_DIR/config/add-planets/DiscConfig_Hof_add-planets-baseline.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/imontesd/output/DiscEvolution}"
OUTDIR="${OUTDIR:-$OUTPUT_ROOT/popsynth/add-planets/pilot_v2}"
P_MIN_YR="${P_MIN_YR:-0.00273972602739726}"  # yr, shortest seed period (1 day = 1/365 yr)
P_MAX_YR="${P_MAX_YR:-1000}"                 # yr, longest seed period
N_P="${N_P:-30}"                             # number of log-spaced periods (before dropping off-grid seeds)
T_END_MYR="${T_END_MYR:-}"                   # Myr; empty = keep the base config's snapshot times (to 3 Myr), unless
                                             # this is an array task (then T_END_LIST[task], see below)
T_END_LIST="${T_END_LIST:-3 5}"              # (2026-11-12, sim-time step 2) Myr, end time per array task (space-separated)
STREAM_TIMING="${STREAM_TIMING:-1}"          # (2026-11-12, sim-time step 2) 1 = write the per-step timing CSV, 0 = off
STREAM_EVERY="${STREAM_EVERY:-1}"            # (2026-11-12, sim-time step 2) steps per timing-CSV row (1 = every step)
WALL_LIMIT_HOURS="${WALL_LIMIT_HOURS:-}"     # (2026-11-13) h, wall-clock limit per run; empty = none
DRY_RUN="${DRY_RUN:-0}"                      # 1 = write and print the config and command, run nothing

# (2026-11-12, sim-time step 2) array task -> end time; then tag the run with it so the 3 and 5 Myr outputs differ
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "$T_END_MYR" ]]; then
    read -r -a _T_ENDS <<< "$T_END_LIST"
    if (( SLURM_ARRAY_TASK_ID >= ${#_T_ENDS[@]} )); then
        echo "ERROR: array task $SLURM_ARRAY_TASK_ID but T_END_LIST has ${#_T_ENDS[@]} entries ($T_END_LIST)" >&2; exit 1
    fi
    T_END_MYR="${_T_ENDS[$SLURM_ARRAY_TASK_ID]}"
fi
if [[ -n "$T_END_MYR" ]]; then RUN_TAG="${RUN_TAG}_${T_END_MYR}Myr"; fi   # e.g. pilotv2_5Myr
RUN_NAME="${RUN_NAME:-hof_add-planets_${RUN_TAG}}"   # moved below the T_END suffix (was in the settings block)

# ---- environment (as run_popsynth_malanga_full_Hof.sh) ----
module load StdEnv/2023 python/3.12 2>/dev/null || true      # no-op where `module` does not exist (local tests)
VENV="${VENV:-/project/def-mbalogh/imontesd/DiscEvolution/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then source "$VENV/bin/activate"; fi

for f in "$DRAWS_CSV" "$BASE_CONFIG" "$SCRIPT_DIR/run_model_Hof.py"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f" >&2; exit 1; }
done
mkdir -p "$OUTDIR/config_used"
GEN_CONFIG="$OUTDIR/config_used/DiscConfig_Hof_add-planets-${RUN_TAG}_disc${DISC_ID}.json"

# ---- steps 1-3: read the disc, build the seeds, write the config; prints the CLI overrides on its LAST line ----
OVERRIDES_LINE=$(python3 - "$DRAWS_CSV" "$BASE_CONFIG" "$GEN_CONFIG" "$DISC_ID" "$RUN_NAME" "$OUTDIR" \
                          "$P_MIN_YR" "$P_MAX_YR" "$N_P" "$T_END_MYR" "$STREAM_TIMING" "$STREAM_EVERY" \
                          "$WALL_LIMIT_HOURS" <<'EOF'
import csv, json, math, sys
(draws_csv, base_config, gen_config, disc_id, run_name, outdir,
 p_min, p_max, n_p, t_end, stream_timing, stream_every, wall_limit_hours) = sys.argv[1:]
disc_id, p_min, p_max, n_p = int(disc_id), float(p_min), float(p_max), int(n_p)

# 1. the disc's row of the draws CSV
rows = [r for r in csv.DictReader(open(draws_csv)) if int(r["id"]) == disc_id]
if len(rows) != 1:
    sys.exit(f"ERROR: {len(rows)} rows with id = {disc_id} in {draws_csv}")
r = rows[0]
Ms, Rs, Teff = float(r["Mstar_Msun"]), float(r["Rstar_Rsun"]), float(r["Teff_K"])     # Msun, Rsun, K
rmin, rmax = float(r["rmin_au"]), float(r["rmax_au"])                                 # AU
M0, R0 = float(r["M0_Msun"]), float(r["R0_au"])                                       # Msun, AU
psi, erad, aSS = float(r["psi_DW"]), float(r["e_rad"]), float(r["alpha_SS"])          # dimensionless

# 2. seeds: N_P periods log-spaced in [p_min, p_max] yr -> a [AU] = M*^(1/3) P^(2/3); keep r_min <= a <= r_max
P = [p_min * (p_max / p_min) ** (k / (n_p - 1)) for k in range(n_p)]                 # yr (= np.geomspace)
a_all = [Ms ** (1 / 3) * p ** (2 / 3) for p in P]                                     # AU
a_in = [a for a in a_all if rmin <= a <= rmax]                                        # AU, seeds inside the grid
if not a_in:
    sys.exit(f"ERROR: no seed inside the grid [{rmin}, {rmax}] AU for disc {disc_id}")

# 3. per-run config
cfg = json.load(open(base_config))
cfg.pop("calibration_result", None)                       # stale copy from a previous run; recomputed by the run
cfg["star"].update({"M": Ms, "R": Rs, "T_eff": Teff})
cfg["grid"].update({"rmin": rmin, "rmax": rmax})
cfg["disc"].update({"M": M0, "Rd": R0, "alpha": aSS * (1 + psi)})   # alpha: TOTAL, for clarity only (hold = alpha_SS)
cfg["winds"].update({"psi_DW": psi, "e_rad": erad})
if cfg.get("calibration", {}).get("hold") != "alpha_SS":
    sys.exit("ERROR: base config must have calibration.hold = 'alpha_SS' (fixed viscous alpha, as the v5 population)")
cfg["calibration"]["alpha_SS"] = aSS
cfg["planets"]["Rp"] = [float(f"{a:.6g}") for a in a_in]  # AU
cfg["planets"]["Mp"] = ["SI"] * len(a_in)
cfg["planets"]["implant_time"] = [0] * len(a_in)
n = {k: len(cfg["planets"][k]) for k in ("Rp", "Mp", "implant_time")}
if len(set(n.values())) != 1:                             # zip() in build_planets would silently drop planets
    sys.exit(f"ERROR: planet lists differ in length: {n}")
cfg["simulation"]["run_name"] = run_name
cfg["simulation"]["output_dir"] = outdir
if t_end:                                                 # cut the snapshot list at T_END_MYR [Myr] ...
    t_end = float(t_end)
    snaps = cfg["simulation"]["t_interval"]               # Myr, explicit list in the base config
    keep = [t for t in snaps if t < t_end]
    # (2026-11-12, sim-time step 2) ... and EXTEND it past the base list's last time with the base list's last spacing
    # (0.1 Myr), so a 5 Myr run gets 3.1, 3.2, ..., 4.9 Myr snapshots and shares every snapshot of the 3 Myr run.
    step = snaps[-1] - snaps[-2]                          # Myr, 0.1 for the pilot v1 base config
    k = 1
    while snaps[-1] + k * step < t_end - 1e-9:
        keep.append(round(snaps[-1] + k * step, 6))       # Myr; round() avoids 3.3000000000000003-style values
        k += 1
    cfg["simulation"]["t_interval"] = keep + [t_end]
# (2026-11-12, sim-time step 2) per-step timing CSV (run_model_Hof.py, sim-time step 1); in the config, not the CLI,
# so config_used/ alone reproduces the run
cfg["simulation"]["stream_timing"] = stream_timing == "1"
cfg["simulation"]["stream_timing_every"] = int(stream_every)   # steps per CSV row
# (2026-11-13) wall-clock limit [h] (run_model_Hof.py stops the run and keeps its output); None = no limit
cfg["simulation"]["wall_limit_hours"] = float(wall_limit_hours) if wall_limit_hours else None
cfg["simulation"].pop("eta_abort_hours", None)            # (2026-11-13) old key, read by nothing; avoid confusion
cfg["_provenance_Hof"] = {                                # ignored by run_model_Hof.py; for the reader
    "made_by": "run_add-planets_pilot_Hof.sh", "draws_csv": draws_csv, "disc_id": disc_id,
    "base_config": base_config, "seed_periods_yr": [p_min, p_max, n_p],
    "n_seeds_in_grid": len(a_in), "n_seeds_dropped_inside_rmin": sum(a < rmin for a in a_all),
    "n_seeds_dropped_beyond_rmax": sum(a > rmax for a in a_all)}
json.dump(cfg, open(gen_config, "w"), indent=2)

print(f"Disc {disc_id}: M* = {Ms:.4g} Msun, R* = {Rs:.4g} Rsun, Teff = {Teff:.4g} K | M0 = {M0:.4g} Msun, "
      f"R0 = {R0:.4g} au | psi = {psi:.4g}, e_rad = {erad:.6g}, alpha_SS = {aSS:.3g} | grid {rmin:.4g}-{rmax:.4g} au",
      file=sys.stderr)
print(f"Seeds: {len(a_in)} of {n_p} periods inside the grid ({a_in[0]:.4g}-{a_in[-1]:.4g} au); dropped "
      f"{sum(a < rmin for a in a_all)} inside r_min, {sum(a > rmax for a in a_all)} beyond r_max", file=sys.stderr)
print("Rp [au] = " + ", ".join(f"{a:.4g}" for a in a_in), file=sys.stderr)
_ti = cfg["simulation"]["t_interval"]                     # (sim-time step 2) snapshot summary, Myr
print(f"Snapshots: {len(_ti)} times, {_ti[0]} ... {_ti[-1]} Myr (last five: {_ti[-5:]}); timing stream "
      f"{'ON, every ' + stream_every + ' step(s)' if stream_timing == '1' else 'OFF'}; wall-clock limit "
      f"{wall_limit_hours + ' h' if wall_limit_hours else 'none'}", file=sys.stderr)
# last line on stdout: the CLI overrides (same values as in the config), read by bash below
print(f"--M {M0!r} --Rd {R0!r} --psi_DW {psi!r} --e_rad {erad!r} --alpha_SS {aSS!r} "
      f"--Mstar {Ms!r} --Rstar {Rs!r} --Teff {Teff!r} --rmin {rmin!r} --rmax {rmax!r}")
EOF
)
read -r -a OVERRIDES <<< "$OVERRIDES_LINE"

CMD=(python3 "$SCRIPT_DIR/run_model_Hof.py" --config "$GEN_CONFIG" --run_name "$RUN_NAME"
     --output_dir "$OUTDIR" "${OVERRIDES[@]}")
echo "Config written: $GEN_CONFIG"
echo "Output folder:  $OUTDIR"
echo "Command:        ${CMD[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1: not running."
    exit 0
fi
echo "Start: $(date -Is)"
"${CMD[@]}"
echo "End:   $(date -Is)"
