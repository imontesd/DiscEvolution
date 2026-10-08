#!/bin/bash
#SBATCH --account=def-mbalogh
#SBATCH --time=0-00:10:00            # adjust to the expected runtime
#SBATCH --cpus-per-task=1            # must equal NPROC below
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=hof_popsynthtest
#SBATCH --output=%x_%j.out           # this script's own echo output
#SBATCH --mail-user=imontesdeocahof@gmail.com
#SBATCH --mail-type=END,FAIL
#
# run_popsynth_Hof.sh
# -------------------
# Launch a grid of DiscEvolution runs (run_model_Hof.py) over
# (psi_DW, Mdot, M, Rd, alpha_SS), running up to $NPROC of them at a time.
#
# This is a script to sweep a parameter grid, run in parallel,
# be safe to re-launch after an interruption and skip any
# (psi, Mdot, M, Rd, alpha_SS) combination whose output file exists and is
# marked complete. So this script can just launch everything every time; the
# Python side figures out what's actually left to do.
#
# (2026-10-08) alpha_SS sweep dimension (ALPHA_SS_VALUES, passed as --alpha_SS):
#   needs the config's calibration section to fix the VISCOUS alpha, e.g.
#       "calibration": {"solve_for": "none", "hold": "alpha_SS", "alpha_SS": 1e-4}
#   Each run then uses TOTAL alpha = alpha_SS * (1 + psi_DW) [dimensionless]
#   (disc.alpha in the config is ignored), and the output filename gets an
#   "_aSS<value>" token so runs differing only in alpha_SS don't collide.
#   Leave ALPHA_SS_VALUES="" to NOT pass --alpha_SS at all (then the config's
#   own calibration / disc.alpha settings are used unchanged).
#   Runs whose initial Mdot falls outside [1e-10, 1e-6] Msun/yr (solve_for =
#   "none") are not evolved: a flagged output file is written and the sweep
#   simply continues.
#
# Usage (Nibi):
#   sbatch run_popsynth_Hof.sh        # with the venv active, or add the
#                                     # module load / source lines below
# Usage (laptop, no SLURM):
#   ./run_popsynth_Hof.sh
#
# To run this fully in the background on a laptop, detached from your terminal:
#   nohup setsid ./run_popsynth_Hof.sh > master.log 2>&1 &

set -euo pipefail

# Optional: make the job independent of the submitting shell's environment
# (uncomment on Nibi; use the SAME modules the venv was built with).
# module load StdEnv/2023 python/3.12
# source /project/def-mbalogh/imontesd/DiscEvolution/.venv/bin/activate

# ---------------------------------------------------------------------------
# 1. Parameter grid. Edit these lines to change what gets run.
# ---------------------------------------------------------------------------
PSI_VALUES="10"
MDOT_VALUES="1e-8"
M_VALUES="0.01"
RD_VALUES="50"
# (2026-10-08) fixed viscous alpha per run [dimensionless]; "" = don't pass --alpha_SS
ALPHA_SS_VALUES="1e-4"

# ---------------------------------------------------------------------------
# 2. Config file, where output/logs go, and how many runs at once.
# ---------------------------------------------------------------------------
SCRIPT_DIR="/home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere"

CONFIG_FILE="$SCRIPT_DIR/config/popsynth/DiscConfig_Hof_popsynthtest.json"

OUTDIR="/project/def-mbalogh/imontesd/output/DiscEvolution/popsynth/test"
LOGDIR="/project/def-mbalogh/imontesd/output/DiscEvolution/logs"
NPROC=1

mkdir -p "$LOGDIR" "$OUTDIR"

# Read run_name back out of the config purely so this script can print it --
# it is NOT used to build a filename here (run_model_Hof.py::output_filename()
# is the single source of truth for filenames).
RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['simulation'].get('run_name','run'))")

echo "Config:    $CONFIG_FILE"
echo "Run name:  $RUN_NAME"
echo "Output to: $OUTDIR"
echo "Logs to:   $LOGDIR"
echo "alpha_SS:  ${ALPHA_SS_VALUES:-(not swept; config values used)}"
echo

# (2026-10-08) An empty ALPHA_SS_VALUES still needs ONE value to loop over;
# "none" is a placeholder meaning "don't pass --alpha_SS" (see run_one).
ALPHA_SS_LOOP="${ALPHA_SS_VALUES:-none}"

# ---------------------------------------------------------------------------
# 3. One job per combination.
# ---------------------------------------------------------------------------
run_one() {
    local psi="$1" mdot="$2" M="$3" Rd="$4" aSS="$5"
    local tag="psi${psi}_Mdot${mdot}_M${M}_Rd${Rd}"
    local aSS_args=()
    # (2026-10-08) alpha_SS: add "--alpha_SS <value>" and an "_aSS<value>" log tag,
    # unless the placeholder "none" (ALPHA_SS_VALUES empty) was given
    if [[ "$aSS" != "none" ]]; then
        aSS_args=(--alpha_SS "$aSS")
        tag="psi${psi}_aSS${aSS}_Mdot${mdot}_M${M}_Rd${Rd}"
    fi

    echo "[$(date +%T)] Launching $tag (skips itself if already done -- see .out log)"
    # (2026-10-08) "|| echo ..." keeps one crashed run from stopping the whole sweep under
    # set -e; the Python traceback is in the .err log. (An out-of-range Mdot is NOT a crash:
    # run_model_Hof.py writes a flagged file and exits normally.)
    python3 "$SCRIPT_DIR/run_model_Hof.py" --config "$CONFIG_FILE" \
        --psi_DW "$psi" --Mdot "$mdot" --M "$M" --Rd "$Rd" "${aSS_args[@]}" --output_dir "$OUTDIR" \
        > "$LOGDIR/${tag}.out" 2> "$LOGDIR/${tag}.err" \
        || echo "[$(date +%T)] FAILED $tag -- see $LOGDIR/${tag}.err"
}
export -f run_one
export SCRIPT_DIR CONFIG_FILE OUTDIR LOGDIR

if command -v parallel >/dev/null 2>&1; then
    parallel -j "$NPROC" run_one {1} {2} {3} {4} {5} \
        ::: $PSI_VALUES ::: $MDOT_VALUES ::: $M_VALUES ::: $RD_VALUES ::: $ALPHA_SS_LOOP
else
    # Fallback if GNU parallel isn't installed: a plain bash job-control
    # loop that does the same thing (launch in the background, cap how
    # many run at once with `wait`).
    echo "(GNU parallel not found -- using a plain bash loop instead)"
    count=0
    for psi in $PSI_VALUES; do
      for mdot in $MDOT_VALUES; do
        for M in $M_VALUES; do
          for Rd in $RD_VALUES; do
            for aSS in $ALPHA_SS_LOOP; do
              run_one "$psi" "$mdot" "$M" "$Rd" "$aSS" &
              # (2026-10-08) was ((count++)): that returns status 1 when count is 0,
              # which stopped the script under set -e after the first launch
              count=$((count + 1))
              if ((count % NPROC == 0)); then wait; fi
            done
          done
        done
      done
    done
    wait
fi

echo
echo "[$(date +%T)] All simulations complete."
