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
# run_popsynth_student.sh
# ------------------------
# Launch a grid of DiscEvolution runs (run_model_student.py) over
# (psi_DW, Mdot, M, Rd), running up to $NPROC of them at a time.
#
# This is a script to sweep a parameter grid, run in parallel, 
# be safe to re-launch after an interruption and skip any (psi, Mdot, M, Rd)
# combination whose output file exists and is marked complete. So this
# script can just launch everything every time; the Python side figures
# out what's actually left to do. 
#
# Usage:
#   ./run_popsynth_student.sh
#
# To run this fully in the background, detached from your terminal (so it
# keeps going after you close your laptop or log out of an ssh session):
#   nohup setsid ./run_popsynth_student.sh > master.log 2>&1 &
# See GETTING_STARTED.md for what each of those pieces does.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Parameter grid. Edit these four lines to change what gets run.
# ---------------------------------------------------------------------------
PSI_VALUES="10"
MDOT_VALUES="1e-8"
M_VALUES="0.01"
RD_VALUES="50"

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
# it is NOT used to build a filename here (see header comment above).
RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['simulation'].get('run_name','run'))")

echo "Config:    $CONFIG_FILE"
echo "Run name:  $RUN_NAME"
echo "Output to: $OUTDIR"
echo "Logs to:   $LOGDIR"
echo

# ---------------------------------------------------------------------------
# 3. One job per combination.
# ---------------------------------------------------------------------------
run_one() {
    local psi="$1" mdot="$2" M="$3" Rd="$4"
    local tag="psi${psi}_Mdot${mdot}_M${M}_Rd${Rd}"

    echo "[$(date +%T)] Launching $tag (skips itself if already done -- see .out log)"
    python3 "$SCRIPT_DIR/run_model_Hof.py" --config "$CONFIG_FILE" \
        --psi_DW "$psi" --Mdot "$mdot" --M "$M" --Rd "$Rd" --output_dir "$OUTDIR" \
        > "$LOGDIR/${tag}.out" 2> "$LOGDIR/${tag}.err"
}
export -f run_one
export SCRIPT_DIR CONFIG_FILE OUTDIR LOGDIR

if command -v parallel >/dev/null 2>&1; then
    parallel -j "$NPROC" run_one {1} {2} {3} {4} \
        ::: $PSI_VALUES ::: $MDOT_VALUES ::: $M_VALUES ::: $RD_VALUES
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
            run_one "$psi" "$mdot" "$M" "$Rd" &
            ((count++))
            if ((count % NPROC == 0)); then wait; fi
          done
        done
      done
    done
    wait
fi

echo
echo "[$(date +%T)] All simulations complete."
