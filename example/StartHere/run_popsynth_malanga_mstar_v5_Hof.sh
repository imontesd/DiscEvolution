#!/bin/bash
#SBATCH --account=def-mbalogh
#SBATCH --time=0-8:00:00            # same request as the v4 full run (run_popsynth_malanga_mstar_full_Hof.sh); same
                                     # grids and the same 500 discs, only the wind parameters differ. The job ends when
                                     # the last disc ends, so unused time costs nothing
#SBATCH --cpus-per-task=20          # one run per CPU; must fit on ONE node (NPROC below follows this value)
#SBATCH --mem-per-cpu=500M           # 500 MB per CPU -> 150 x 0.5 GB = 75 GB for the job (RAM, not disk); as v4
#SBATCH --job-name=hof_malanga_mstar_v5
#SBATCH --output=/scratch/imontesd/output/DiscEvolution/logs/malanga-comparison/%x_%j.out   # this script's own echo output (folder must exist BEFORE sbatch)
#SBATCH --mail-user=imontesdeocahof@gmail.com
#SBATCH --mail-type=END,FAIL
#
# run_popsynth_malanga_mstar_v5_Hof.sh
# ------------------------------------
# (Hof, 2026-11-09) FULL run of the per-disc-M* Malanga et al. (2026) population, VERSION 5: all 500 discs of
# config/malanga/malanga_mstar_draws_Hof_N500_seed1_xirmin.csv. A copy of run_popsynth_malanga_mstar_full_Hof.sh
# (the v4 run, left unchanged so v4 stays reproducible) with only these differences:
#   * DRAWS_CSV  -> the "_xirmin" population CSV;
#   * OUTDIR     -> popsynth/malanga_mstar_aSS1e-5_v5,  LOGDIR -> logs/malanga-comparison/mstar_aSS1e-5_v5
#     (exported here, so run_popsynth_malanga_full_Hof.sh uses them instead of its POPULATION=mstar "_v4" defaults);
#   * the population check also requires the v5 column xi_Malanga_Rin, and f_M_grid_expected = 10^-0.6 for every
#     wind disc, i.e. xi = ln(1 + f_M) / ln(R0 / r_min) (Tabone et al. 2022, Eq. 22, p. 5; Malanga Table 2, p. 7).
# The run itself (job list, run_one, population guard, provenance, summary) is still done by
# run_popsynth_malanga_full_Hof.sh, so there is one copy of that logic.
#
# Population version 5 = version 4 with ONE change (sample_malanga_mstar_Hof.py, module docstring point 7):
#   the ejection index is calibrated on each disc's OWN grid,
#       xi = ln(1 + f_M) / ln(R0 / r_min)        (v4: ln(1 + f_M) / ln(R0 / 0.03 au), Malanga Eq. 6, p. 5)
#   so that the steady-state wind/accretion ratio the grid can produce, (R0 / r_min)^xi - 1 (Tabone et al. 2022,
#   Eq. 22, p. 5, with r_in = r_min), equals Malanga's f_M = 10^-0.6 (Table 2, p. 7) for EVERY disc (v4: median
#   0.226, down to 0.150). e_rad (relation D) and lambda_DW (Balogh et al. 2026, Eq. 16, p. 5) follow from xi.
#   M*, R*, Teff, M0, R0, alpha, psi, r_min, r_max and t_age are IDENTICAL to v4 disc by disc; the 151 discs on the
#   0.03 au r_min floor have the same e_rad too (their runs are the same as in v4, but are re-run so the whole
#   population comes from one code version and one folder).
#
# Output (scratch):
#     /scratch/imontesd/output/DiscEvolution/popsynth/malanga_mstar_aSS1e-5_v5/       .h5 per disc, config_log/,
#                                                         population CSV + .json, run_provenance_<jobid>.txt
#     /scratch/imontesd/output/DiscEvolution/logs/malanga-comparison/mstar_aSS1e-5_v5/          job list, joblog,
#                                                         run_logs/<run_name>.out/.err
#     /scratch/imontesd/output/DiscEvolution/logs/malanga-comparison/hof_malanga_mstar_v5_<jobid>.out
#
# Restart: run_model_Hof.py skips discs that are already complete, so if the job hits the time limit,
# submit this script again. Only unfinished discs run, and each of those restarts from t = 0.
#
# Usage on Nibi:
#   cd /home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere
#   git pull
#   mkdir -p /scratch/imontesd/output/DiscEvolution/logs/malanga-comparison
#   DRY_RUN=1 bash run_popsynth_malanga_mstar_v5_Hof.sh | head -20    # check the commands, run nothing
#   sbatch run_popsynth_malanga_mstar_v5_Hof.sh

set -euo pipefail

# Absolute path: Slurm runs a spooled COPY of this file, so its own location cannot be used to find the code
SCRIPT_DIR="${SCRIPT_DIR:-/home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere}"
DRAWS_CSV="${DRAWS_CSV:-$SCRIPT_DIR/config/malanga/malanga_mstar_draws_Hof_N500_seed1_xirmin.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/imontesd/output/DiscEvolution}"
OUTDIR="${OUTDIR:-$OUTPUT_ROOT/popsynth/malanga_mstar_aSS1e-5_v5}"                  # NEW folder: never mixed with v4
LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs/malanga-comparison/mstar_aSS1e-5_v5}"
N_EXPECTED="${N_EXPECTED:-500}"      # discs in the population (only lowered for local tests)
R0_RMIN_MIN=5                        # dimensionless, both radii in au; sample_malanga_mstar_Hof.py, R0_RMIN_MIN
LOG_FM_MALANGA=-0.6                  # Malanga et al. (2026) Table 2, p. 7: log10 f_M (dimensionless)

# Stop here, before any CPU time is used, if this checkout does not have population version 5
# (e.g. `git pull` was forgotten). Columns are found by NAME, as in the job list of the launcher.
[[ -f "$DRAWS_CSV" ]] || { echo "ERROR: not found: $DRAWS_CSV" >&2; exit 1; }
awk -F, -v n_expected="$N_EXPECTED" -v ratio_min="$R0_RMIN_MIN" -v logfm="$LOG_FM_MALANGA" '
    function abs(x) { return x < 0 ? -x : x }
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i
              if (!("xi_Malanga_Rin" in col) || !("R0_over_rmin" in col) || !("f_M_grid_expected" in col)) {
                  print "ERROR: this CSV has no xi_Malanga_Rin column: not population version 5 (git pull?)"
                  bad = 1; exit 1 }
              fm = exp(logfm * log(10))                                    # 10^-0.6 = 0.2512
              next }
    { n++
      if ($col["R0_over_rmin"] + 0 < ratio_min) low++
      if ($col["Mstar_Msun"] + 0 < 0.1 || $col["Mstar_Msun"] + 0 > 1.4) outside++    # Msun
      # v5: xi calibrated on the disc own r_min [au], hence (R0/r_min)^xi - 1 = f_M for every disc with a wind
      # (psi > 0). Written with 8 sf, so compare relatively; a v4-style CSV (xi from 0.03 au) fails here
      if ($col["psi_DW"] + 0 > 0 && abs($col["f_M_grid_expected"] / fm - 1) > 1e-6) off_fm++ }
    END { if (bad) exit 1
          if (n != n_expected) { print "ERROR: " n " discs in the CSV, expected " n_expected; exit 1 }
          if (low)             { print "ERROR: " low " disc(s) with R0 / r_min < " ratio_min; exit 1 }
          if (outside)         { print "ERROR: " outside " disc(s) with M* outside 0.1-1.4 Msun"; exit 1 }
          if (off_fm)          { print "ERROR: " off_fm " wind disc(s) with (R0/r_min)^xi - 1 != 10^" logfm \
                                       " (xi not calibrated on r_min: v4 CSV?)"; exit 1 }
          print "Population check OK: " n " discs, R0 / r_min >= " ratio_min ", M* in 0.1-1.4 Msun, " \
                "xi on r_min -> R0, (R0/r_min)^xi - 1 = 10^" logfm }
' "$DRAWS_CSV" >&2

# The launcher reads all of these from the environment (POPULATION=mstar picks the per-M* config,
# DiscConfig_Hof_malanga_mstar.json; OUTDIR and LOGDIR override its "_v4" folder defaults).
export POPULATION=mstar
export SCRIPT_DIR DRAWS_CSV OUTPUT_ROOT OUTDIR LOGDIR
export NPROC="${SLURM_CPUS_PER_TASK:-${NPROC:-20}}"     # runs at once = CPUs actually granted by Slurm


# (Hof, 2026-11-09) PILOT ONLY: 20 discs (17 off the 0.03 au r_min floor, spanning the v4 f_M deficit; 3 floor controls
# 153, 269, 333 that must reproduce v4 exactly). IDS is read ONLY by run_popsynth_malanga_Hof.sh (the pilot launcher);
# run_popsynth_malanga_full_Hof.sh ignores it on purpose and would run all 500 discs.
export IDS="6 23 30 72 128 153 199 245 263 267 269 306 312 321 333 368 373 379 403 497"

# PILOT: the pilot launcher (honours IDS). For the FULL v5 run, delete the IDS line above and call
# run_popsynth_malanga_full_Hof.sh here instead (and restore the full-run #SBATCH request).
bash "$SCRIPT_DIR/run_popsynth_malanga_Hof.sh"
