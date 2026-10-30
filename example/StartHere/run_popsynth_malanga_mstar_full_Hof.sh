#!/bin/bash
#SBATCH --account=def-mbalogh
#SBATCH --time=2-10:00:00            # pilot 3 estimate ~7.5 h wall on 150 slots, slowest disc 7.46 h; the
                                     # job ends when the last disc ends, so unused time costs nothing
#SBATCH --cpus-per-task=150          # one run per CPU; must fit on ONE node (NPROC below follows this value)
#SBATCH --mem-per-cpu=500M           # 500 MB per CPU -> 150 x 0.5 GB = 75 GB for the job (RAM, not disk);
                                     # same as pilot 3, where no disc ran out of memory
#SBATCH --job-name=hof_malanga_mstar_full
#SBATCH --output=/scratch/imontesd/output/DiscEvolution/logs/malanga-comparison/%x_%j.out   # this script's own echo output (folder must exist BEFORE sbatch)
#SBATCH --mail-user=imontesdeocahof@gmail.com
#SBATCH --mail-type=END,FAIL
#
# run_popsynth_malanga_mstar_full_Hof.sh
# --------------------------------------
# (Hof, 2026-10-30) FULL run of the per-disc-M* Malanga et al. (2026) population, version 4: all 500
# discs of config/malanga/malanga_mstar_draws_Hof_N500_seed1.csv. Use THIS script for that run.
#
# Why a separate script: run_popsynth_malanga_full_Hof.sh defaults to POPULATION=1msun, so submitting
# it without POPULATION=mstar would start the wrong 500 discs. This file only holds the Slurm request
# for this run and the checks below; the run itself (job list, run_one, population guard, provenance,
# summary) is still done by run_popsynth_malanga_full_Hof.sh, so there is one copy of that logic.
#
# Population version 4 = version 3 (pilot 3) plus the two changes of 2026-10-30:
#   * 0.1 dex Gaussian scatter on log10 M* of each Table A.1 draw (Malanga Sec. 2.4, p. 3), M* in
#     0.1-1.4 Msun;
#   * draws with R0 / r_min < 5 are discarded and the disc gets a new star and a new R0.
# Unchanged: alpha_SS = 1e-5, r_max = min(20 R0, 1000 au), r_min = 0.1 au (R*/3.079 Rsun)^(12/7) in
# [0.03, 0.1] au, nr = 500, 34 snapshots to 3 Myr.
#
# Output (scratch):
#     /scratch/imontesd/output/DiscEvolution/popsynth/malanga_mstar_aSS1e-5_v4/       .h5 per disc, config_log/,
#                                                         population CSV + .json, run_provenance_<jobid>.txt
#     /scratch/imontesd/output/DiscEvolution/logs/malanga-comparison/mstar_aSS1e-5_v4/          job list, joblog,
#                                                         run_logs/<run_name>.out/.err
#     /scratch/imontesd/output/DiscEvolution/logs/malanga-comparison/hof_malanga_mstar_full_<jobid>.out
#
# Restart: run_model_Hof.py skips discs that are already complete, so if the job hits the time limit,
# submit this script again. Only unfinished discs run, and each of those restarts from t = 0.
#
# Usage on Nibi:
#   cd /home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere
#   git pull
#   mkdir -p /scratch/imontesd/output/DiscEvolution/logs/malanga-comparison
#   DRY_RUN=1 bash run_popsynth_malanga_mstar_full_Hof.sh | head -20    # check the commands, run nothing
#   sbatch run_popsynth_malanga_mstar_full_Hof.sh

set -euo pipefail

# Absolute path: Slurm runs a spooled COPY of this file, so its own location cannot be used to find the code
SCRIPT_DIR="${SCRIPT_DIR:-/home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere}"
DRAWS_CSV="${DRAWS_CSV:-$SCRIPT_DIR/config/malanga/malanga_mstar_draws_Hof_N500_seed1.csv}"
N_EXPECTED="${N_EXPECTED:-500}"      # discs in the population (only lowered for local tests)
R0_RMIN_MIN=5                        # dimensionless, both radii in au; sample_malanga_mstar_Hof.py, R0_RMIN_MIN

# Stop here, before any CPU time is used, if this checkout does not have population version 4
# (e.g. `git pull` was forgotten). Columns are found by NAME, as in the job list of the launcher.
[[ -f "$DRAWS_CSV" ]] || { echo "ERROR: not found: $DRAWS_CSV" >&2; exit 1; }
awk -F, -v n_expected="$N_EXPECTED" -v ratio_min="$R0_RMIN_MIN" '
    NR == 1 { for (i = 1; i <= NF; i++) col[$i] = i
              if (!("Mstar_table_Msun" in col) || !("R0_over_rmin" in col) || !("Mstar_Msun" in col)) {
                  print "ERROR: this CSV has no Mstar_table_Msun / R0_over_rmin column: not population version 4 (git pull?)"
                  bad = 1; exit 1 }
              next }
    { n++
      if ($col["R0_over_rmin"] + 0 < ratio_min) low++
      if ($col["Mstar_Msun"] + 0 < 0.1 || $col["Mstar_Msun"] + 0 > 1.4) outside++ }   # Msun
    END { if (bad) exit 1
          if (n != n_expected) { print "ERROR: " n " discs in the CSV, expected " n_expected; exit 1 }
          if (low)             { print "ERROR: " low " disc(s) with R0 / r_min < " ratio_min; exit 1 }
          if (outside)         { print "ERROR: " outside " disc(s) with M* outside 0.1-1.4 Msun"; exit 1 }
          print "Population check OK: " n " discs, R0 / r_min >= " ratio_min ", M* in 0.1-1.4 Msun" }
' "$DRAWS_CSV" >&2

# The launcher reads all of these from the environment. Its POPULATION=mstar defaults give the _v4 folders.
export POPULATION=mstar
export SCRIPT_DIR DRAWS_CSV
export NPROC="${SLURM_CPUS_PER_TASK:-${NPROC:-150}}"     # runs at once = CPUs actually granted by Slurm

bash "$SCRIPT_DIR/run_popsynth_malanga_full_Hof.sh"
