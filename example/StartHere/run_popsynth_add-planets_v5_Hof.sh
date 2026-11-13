#!/bin/bash
#SBATCH --account=def-mbalogh
#SBATCH --array=0-166                # 167 tasks; task k runs the discs at CSV rows k, k + 167, k + 334 (3 discs, the last
                                     # tasks only 2: 167 x 3 = 501 >= 500). N_TASKS below must equal the array size
#SBATCH --cpus-per-task=1            # per TASK: one serial run_model_Hof.py at a time (167 CPUs in total)
#SBATCH --mem=2G                     # per TASK (RAM, not disk); the pilot runs need well under this
#SBATCH --time=7-00:00:00            # per TASK: 3 discs x WALL_LIMIT_HOURS = 3 x 48 h = 144 h worst case < 168 h, leaving
                                     # 24 h for module loading, Python start-up and margin. The guard below checks it
#SBATCH --job-name=hof_add-planets_v5
#SBATCH --output=/scratch/imontesd/output/DiscEvolution/logs/add-planets/%x_%A_%a.out   # %A = array job ID, %a = task;
                                                                                        # folder must exist BEFORE sbatch
#SBATCH --mail-user=imontesdeocahof@gmail.com
#SBATCH --mail-type=END,FAIL
#
# run_popsynth_add-planets_v5_Hof.sh
# ----------------------------------
# (Hof, 2026-11-13) PLANET POPULATION run: all 500 discs of the v5 population
# (config/malanga/malanga_mstar_draws_Hof_N500_seed1_xirmin.csv), each with its OWN planet seeds.
#
# WHAT EACH DISC GETS (built by run_add-planets_pilot_Hof.sh, called once per disc so there is ONE copy of the logic):
#   * seeds = the seeding of Hof_figures_add-planets.ipynb (2026-11-12): 30 orbital periods log-spaced from 1 day to
#     1000 yr, a [AU] = (M*/Msun)^(1/3) (P/yr)^(2/3), keeping only r_min <= a <= r_max; all Mp = "SI" (inserted where
#     planetesimals form), implant_time 0; Rp, Mp, implant_time of equal length (checked);
#   * star, grid, M0, R0, psi_DW, e_rad, alpha_SS from the disc's CSV row (per-run config in $OUTDIR/config_used/);
#   * everything else from BASE_CONFIG = config/add-planets/DiscConfig_Hof_add-planets.json (chosen 2026-11-13):
#     pb_gas_f = 0.05, planetesimal diameter "SI" (rho_pltsml 2.0), e_init / i_init "eq", drag, VS_embryo, VS_pltsml,
#     DF on. NOTE: the pilots (v1, v2 timing trial) used DiscConfig_Hof_add-planets-baseline.json (pb_gas_f 0.1,
#     diameter 200), so the trial's runtimes are only a guide for these runs;
#   * end time T_END_MYR = 5 Myr (default, chosen 2026-11-13): the base snapshot list (to 3 Myr) extended every 0.1 Myr;
#   * wall-clock limit WALL_LIMIT_HOURS = 48 h (run_model_Hof.py, change log 2026-11-13): a disc still running after
#     48 h is stopped, keeping all its output, with HDF5 attrs stopped_by_wall_limit = True and t_stop_yr. It is still
#     marked complete, so a re-submission does NOT re-run it;
#   * timing stream OFF (STREAM_TIMING=0): no per-step CSV in production.
#
# OUTPUT (scratch)
#   $OUTDIR  = /scratch/imontesd/output/DiscEvolution/popsynth/add-planets/v5_5Myr/
#              one .h5 per disc: hof_add-planets_v5_disc<NNN>_5Myr_gamma1_psi..._...h5, config_used/, config_log/
#   $LOGDIR  = /scratch/imontesd/output/DiscEvolution/logs/add-planets/v5_5Myr/
#              run_logs/disc<NNN>.out (each disc's full run_model_Hof.py output; appended, one block per attempt),
#              status/task<k>.tsv (one line per disc and attempt: id, exit code, state, start, end), where
#              state = ok | STOPPED (48 h limit) | skipped (.h5 already complete) | FAILED (non-zero exit)
#   Slurm log per task: logs/add-planets/hof_add-planets_v5_<jobID>_<task>.out
#
# AFTER THE RUN: stopped discs have HDF5 attr stopped_by_wall_limit = True ("STOPPED" in status/*.tsv); discs killed
# by Slurm or crashed have no complete attr (exit code != 0 in status/*.tsv).
#
# RESTART: run_model_Hof.py skips a disc whose .h5 is complete (finished OR stopped by the 48 h limit), so submitting
# this script again only runs unfinished discs, each from t = 0. `sbatch --array=12,40 ...` re-runs only those tasks.
#
# USAGE (Nibi)
#   cd /home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere
#   git pull
#   mkdir -p /scratch/imontesd/output/DiscEvolution/logs/add-planets          # the #SBATCH --output folder
#   SLURM_ARRAY_TASK_ID=0 DRY_RUN=1 bash run_popsynth_add-planets_v5_Hof.sh   # task 0's discs: configs + commands only
#   sbatch run_popsynth_add-planets_v5_Hof.sh

set -euo pipefail

# ---- settings (all overridable from the environment) ----
# Absolute path: Slurm runs a spooled COPY of this file, so its own location cannot be used to find the code
SCRIPT_DIR="${SCRIPT_DIR:-/home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere}"
DRAWS_CSV="${DRAWS_CSV:-$SCRIPT_DIR/config/malanga/malanga_mstar_draws_Hof_N500_seed1_xirmin.csv}"
BASE_CONFIG="${BASE_CONFIG:-$SCRIPT_DIR/config/add-planets/DiscConfig_Hof_add-planets.json}"
T_END_MYR="${T_END_MYR:-5}"                  # Myr, end time of every disc
WALL_LIMIT_HOURS="${WALL_LIMIT_HOURS:-48}"   # h, per-disc wall-clock limit (run_model_Hof.py stops + saves the run)
STREAM_TIMING="${STREAM_TIMING:-0}"          # 0 = no per-step timing CSV (production); 1 = on
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/imontesd/output/DiscEvolution}"
OUTDIR="${OUTDIR:-$OUTPUT_ROOT/popsynth/add-planets/v5_${T_END_MYR}Myr}"
LOGDIR="${LOGDIR:-$OUTPUT_ROOT/logs/add-planets/v5_${T_END_MYR}Myr}"
N_TASKS="${N_TASKS:-${SLURM_ARRAY_TASK_COUNT:-167}}"   # array size (= #SBATCH --array count); discs are dealt out in it
TASK="${SLURM_ARRAY_TASK_ID:-0}"             # this task (0 when run by hand, e.g. for DRY_RUN)
N_EXPECTED="${N_EXPECTED:-500}"              # discs in the CSV (only lowered for local tests)
JOB_HOURS="${JOB_HOURS:-168}"                # h, = #SBATCH --time (7 d); used only by the time guard below
DRY_RUN="${DRY_RUN:-0}"                      # 1 = write + print each disc's config and command, run nothing

[[ -f "$DRAWS_CSV" ]] || { echo "ERROR: not found: $DRAWS_CSV" >&2; exit 1; }
[[ -f "$BASE_CONFIG" ]] || { echo "ERROR: not found: $BASE_CONFIG" >&2; exit 1; }
[[ -f "$SCRIPT_DIR/run_add-planets_pilot_Hof.sh" ]] || { echo "ERROR: run_add-planets_pilot_Hof.sh not in $SCRIPT_DIR" >&2; exit 1; }

# ---- the disc IDs (CSV column "id", found by NAME), and this task's share: rows TASK, TASK + N_TASKS, ... ----
# Dealing every N_TASKS-th row (not a contiguous block) mixes the discs across tasks.
mapfile -t ALL_IDS < <(awk -F, 'NR == 1 { for (i = 1; i <= NF; i++) if ($i == "id") c = i; next } { print $c }' "$DRAWS_CSV")
if (( ${#ALL_IDS[@]} != N_EXPECTED )); then
    echo "ERROR: ${#ALL_IDS[@]} discs in $DRAWS_CSV, expected $N_EXPECTED" >&2; exit 1
fi
MY_IDS=()
for (( i = TASK; i < ${#ALL_IDS[@]}; i += N_TASKS )); do MY_IDS+=("${ALL_IDS[$i]}"); done
PER_TASK_MAX=$(( (${#ALL_IDS[@]} + N_TASKS - 1) / N_TASKS ))   # discs on the busiest task (3 for 500 / 167)

# ---- time guard: the busiest task must fit in the job time even if EVERY disc hits the wall-clock limit ----
if [[ -n "$WALL_LIMIT_HOURS" ]] && awk -v n="$PER_TASK_MAX" -v w="$WALL_LIMIT_HOURS" -v j="$JOB_HOURS" \
        'BEGIN { exit !(n * w > j) }'; then
    echo "ERROR: $PER_TASK_MAX discs x $WALL_LIMIT_HOURS h > job time $JOB_HOURS h: Slurm could kill a task mid-disc" \
         "(that disc would have no final snapshot and would re-run on resubmission). Raise N_TASKS or #SBATCH --time." >&2
    exit 1
fi

mkdir -p "$OUTDIR" "$LOGDIR/run_logs" "$LOGDIR/status"
STATUS="$LOGDIR/status/task${TASK}.tsv"
echo "Task $TASK of $N_TASKS: discs ${MY_IDS[*]}  (T_END = $T_END_MYR Myr, wall limit = ${WALL_LIMIT_HOURS:-none} h)"
echo "Code: $(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)  base config: $BASE_CONFIG"
echo "Output: $OUTDIR   logs: $LOGDIR"

# ---- run this task's discs one after another ----
for id in "${MY_IDS[@]}"; do
    tag=$(printf "v5_disc%03d" "$id")                        # run_name hof_add-planets_v5_disc<NNN>_<T_END>Myr
    log="$LOGDIR/run_logs/$(printf "disc%03d" "$id").out"
    start=$(date -Is)
    echo "[$start] disc $id -> $log"
    # The disc log is APPENDED to (a re-submission must not overwrite the log of an earlier attempt, e.g. the one that
    # says STOPPED); n0 = its length before this attempt, so the state below is read from THIS attempt's lines only.
    n0=$( { wc -l < "$log"; } 2>/dev/null || echo 0 )
    echo "===== attempt $start (Slurm job ${SLURM_ARRAY_JOB_ID:-local}, task $TASK) =====" >> "$log"
    # env -u SLURM_ARRAY_TASK_ID: the single-disc script would otherwise read this task number as ITS OWN array
    # index (its 3 vs 5 Myr trial). T_END_MYR is passed explicitly, so it is not needed there anyway.
    set +e                                                   # one failed disc must not stop the task's other discs
    env -u SLURM_ARRAY_TASK_ID \
        DISC_ID="$id" RUN_TAG="$tag" T_END_MYR="$T_END_MYR" OUTDIR="$OUTDIR" \
        SCRIPT_DIR="$SCRIPT_DIR" DRAWS_CSV="$DRAWS_CSV" BASE_CONFIG="$BASE_CONFIG" \
        WALL_LIMIT_HOURS="$WALL_LIMIT_HOURS" STREAM_TIMING="$STREAM_TIMING" DRY_RUN="$DRY_RUN" \
        bash "$SCRIPT_DIR/run_add-planets_pilot_Hof.sh" >> "$log" 2>&1
    rc=$?
    set -e
    end=$(date -Is)
    this=$(tail -n +"$((n0 + 1))" "$log")                     # this attempt's output only
    # state: STOPPED = 48 h limit hit; skipped = .h5 already complete (finished or stopped earlier, see older attempts
    # in the log / its HDF5 attrs); ok = finished; FAILED = non-zero exit (crash; the .h5 has no complete attr)
    if grep -q "STOPPED by wall-clock limit" <<< "$this"; then state=STOPPED
    elif grep -q "Skipping -- output already complete" <<< "$this"; then state=skipped
    elif (( rc == 0 )); then state=ok
    else state=FAILED; fi
    echo "[$end] disc $id: $state (exit $rc)"
    if [[ "$DRY_RUN" == "1" ]]; then
        sed -n '/^Command:/p' "$log"                         # dry run: show the command; no status line written
    else
        # status line (appended, so a re-submission adds lines; the LAST line per disc is its latest attempt)
        printf "%s\t%s\t%s\t%s\t%s\n" "$id" "$rc" "$state" "$start" "$end" >> "$STATUS"
    fi
done
if [[ -f "$STATUS" ]]; then echo "Task $TASK status so far: $(cut -f3 "$STATUS" | sort | uniq -c | tr '\n' ' ')"; fi
