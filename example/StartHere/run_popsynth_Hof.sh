#!/bin/bash
#SBATCH --account=def-mbalogh
#SBATCH --time=0-09:00:00            # adjust to the expected runtime (whole sweep, see NOTE below)
#SBATCH --cpus-per-task=162          # must equal NPROC below, and fit on ONE node
#SBATCH --mem-per-cpu=8G             # NOTE: total = 162 x 8G = 1296G -- must fit on ONE node (check!)
#SBATCH --job-name=hof_popsynth1.00
#SBATCH --output=%x_%j.out           # this script's own echo output
#SBATCH --mail-user=imontesdeocahof@gmail.com
#SBATCH --mail-type=END,FAIL
#
# run_popsynth_Hof.sh
# -------------------
# Launch a grid of DiscEvolution runs (run_model_Hof.py) over
# (psi_DW, alpha_SS, M, Rd, e_rad), running up to $NPROC of them at a time.
#
# This is a script to sweep a parameter grid, run in parallel,
# be safe to re-launch after an interruption and skip any
# combination whose output file exists and is marked complete. So this
# script can just launch everything every time; the Python side figures
# out what's actually left to do.
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
# (2026-10-08) e_rad sweep dimension (ERAD_VALUES, passed as --e_rad, dimensionless).
#   e_rad is already in the output filename ("_erad<value>").
#
# Grid size: 4 psi x 3 alpha_SS x 3 M x 3 Rd x 3 e_rad = 324 runs
#   -> with NPROC = 162: two "waves" of 162 runs (the fallback loop waits for a whole
#      wave to finish before starting the next; GNU parallel starts a new run as soon
#      as any slot frees up). The --time limit covers BOTH waves.
#   NOTE: for psi = 0 (no wind) e_rad has no effect, so the 3 e_rad values give the
#   same physics -- 27 of the 324 runs are duplicates (different filenames).
#
# Usage (Nibi):
#   sbatch run_popsynth_Hof.sh        # with the venv active, or uncomment the
#                                     # module load / source lines below

set -euo pipefail

# Optional: make the job independent of the submitting shell's environment
# (uncomment on Nibi; use the SAME modules the venv was built with).
# module load StdEnv/2023 python/3.12
# source /project/def-mbalogh/imontesd/DiscEvolution/.venv/bin/activate

# ---------------------------------------------------------------------------
# 1. Parameter grid. Edit these lines to change what gets run.
# ---------------------------------------------------------------------------
PSI_VALUES="0 10 100 1000"           # psi_DW = alpha_DW / alpha_SS, dimensionless
ALPHA_SS_VALUES="1e-4 1e-3 1e-2"     # fixed viscous alpha, dimensionless; "" = don't pass --alpha_SS
M_VALUES="0.001 0.01 0.1"            # initial disc mass, Msun
RD_VALUES="25 50 100"                # characteristic radius, AU
ERAD_VALUES="0.8 0.9 1.0"            # wind radiative-loss efficiency e_rad, dimensionless
# (Mdot is NOT swept: with solve_for = "none" the initial Mdot is an output; disc.Mdot in the
#  config is only the nominal filename label.)

# An empty ALPHA_SS_VALUES still needs ONE value to loop over; "none" is a placeholder
# meaning "don't pass --alpha_SS" (see run_one).
ALPHA_SS_LOOP="${ALPHA_SS_VALUES:-none}"

# ---------------------------------------------------------------------------
# 2. Config file, where output/logs go, and how many runs at once.
# ---------------------------------------------------------------------------
SCRIPT_DIR="/home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere"

CONFIG_FILE="$SCRIPT_DIR/config/popsynth/DiscConfig_Hof_popsynth1.00.json"

OUTDIR="/project/def-mbalogh/imontesd/output/DiscEvolution/popsynth/test"   # NOTE: still the "test" folder
LOGDIR="/project/def-mbalogh/imontesd/output/DiscEvolution/logs"
NPROC=162

mkdir -p "$LOGDIR" "$OUTDIR"

# Read run_name back out of the config purely so this script can print it --
# it is NOT used to build a filename here (run_model_Hof.py::output_filename()
# is the single source of truth for filenames).
RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['simulation'].get('run_name','run'))")

echo "Config:    $CONFIG_FILE"
echo "Run name:  $RUN_NAME"
echo "Output to: $OUTDIR"
echo "Logs to:   $LOGDIR"
echo "psi:       $PSI_VALUES"
echo "alpha_SS:  ${ALPHA_SS_VALUES:-(not swept; config values used)}"
echo "M [Msun]:  $M_VALUES"
echo "Rd [AU]:   $RD_VALUES"
echo "e_rad:     $ERAD_VALUES"
echo

# ---------------------------------------------------------------------------
# 3. One job per combination.
# ---------------------------------------------------------------------------
# Arguments, in this order (must match the parallel ::: lists and the loops below):
#   $1 psi   $2 alpha_SS (or "none")   $3 M   $4 Rd   $5 e_rad
run_one() {
    local psi="$1" aSS="$2" M="$3" Rd="$4" erad="$5"
    local tag="psi${psi}_M${M}_Rd${Rd}_erad${erad}"
    local aSS_args=()
    # alpha_SS: add "--alpha_SS <value>" and an "_aSS<value>" log tag, unless the
    # placeholder "none" (ALPHA_SS_VALUES empty) was given
    if [[ "$aSS" != "none" ]]; then
        aSS_args=(--alpha_SS "$aSS")
        tag="psi${psi}_aSS${aSS}_M${M}_Rd${Rd}_erad${erad}"
    fi

    echo "[$(date +%T)] Launching $tag (skips itself if already done -- see .out log)"
    # "|| echo ..." keeps one crashed run from stopping the whole sweep under set -e;
    # the Python traceback is in the .err log. (An out-of-range Mdot is NOT a crash:
    # run_model_Hof.py writes a flagged file and exits normally.)
    python3 "$SCRIPT_DIR/run_model_Hof.py" --config "$CONFIG_FILE" \
        --psi_DW "$psi" "${aSS_args[@]}" --M "$M" --Rd "$Rd" --e_rad "$erad" --output_dir "$OUTDIR" \
        > "$LOGDIR/${tag}.out" 2> "$LOGDIR/${tag}.err" \
        || echo "[$(date +%T)] FAILED $tag -- see $LOGDIR/${tag}.err"
}
export -f run_one
export SCRIPT_DIR CONFIG_FILE OUTDIR LOGDIR

if command -v parallel >/dev/null 2>&1; then
    # 5 argument lists, in run_one's argument order
    parallel -j "$NPROC" run_one {1} {2} {3} {4} {5} \
        ::: $PSI_VALUES ::: $ALPHA_SS_LOOP ::: $M_VALUES ::: $RD_VALUES ::: $ERAD_VALUES
else
    # Fallback if GNU parallel isn't installed: a plain bash job-control
    # loop that does the same thing (launch in the background, cap how
    # many run at once with `wait`).
    echo "(GNU parallel not found -- using a plain bash loop instead)"
    count=0
    for psi in $PSI_VALUES; do
      for aSS in $ALPHA_SS_LOOP; do
        for M in $M_VALUES; do
          for Rd in $RD_VALUES; do
            for erad in $ERAD_VALUES; do
              run_one "$psi" "$aSS" "$M" "$Rd" "$erad" &
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
