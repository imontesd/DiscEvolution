#!/bin/bash
#SBATCH --account=def-mbalogh
#SBATCH --time=0-05:00:00            # pilot: slowest disc 1.2 h; 500 discs / 192 slots = 3 waves (~1.5-2 h)
#SBATCH --cpus-per-task=150          # must equal NPROC below, and fit on ONE node
#SBATCH --mem-per-cpu=500M           # 500 MB per CPU -> 192 x 0.5 GB = 96 GB for the job (RAM, not disk);
                                     # measured peak ~120 MB per run (/usr/bin/time -v, typical and most
                                     # compact disc), so ~23 GB in use with 192 runs at once
#SBATCH --job-name=hof_malanga_full
#SBATCH --output=/scratch/imontesd/output/DiscEvolution/logs/malanga-comparison/%x_%j.out   # this script's own echo output (folder must exist BEFORE sbatch)
#SBATCH --mail-user=imontesdeocahof@gmail.com
#SBATCH --mail-type=END,FAIL
#
# run_popsynth_malanga_full_Hof.sh
# --------------------------------
# (Hof, 2026-10-20) FULL Malanga et al. (2026) MHD-wind population run: ALL discs in the population
# CSV (500 for malanga_draws_Hof_N500_seed1.csv), up to $NPROC at a time. A copy of
# run_popsynth_malanga_Hof.sh (keep using that one for pilots / subsets), with these differences:
#   * no disc selection: the FIRST_ID / LAST_ID / IDS options were REMOVED, so every row of the CSV is
#     run (if any of those variables is set in the environment, e.g. left over from the pilot, it is
#     IGNORED and a warning is printed); the script checks the job list has one line per CSV row;
#   * Slurm resources sized from the pilot: --mem-per-cpu 500M (was 3800M), --time 4 h (was 8 h),
#     job name hof_malanga_full;
#   * provenance: git commit / branch / uncommitted-changes state of the code, host, Slurm job id and
#     start time are printed to the Slurm output AND saved as OUTDIR/run_provenance_<jobid>.txt, since
#     the .h5 files do not record the code version;
#   * before launching it reports output files ALREADY in OUTDIR (e.g. pilot discs): those are
#     skipped by run_model_Hof.py, and files lacking the GI attrs come from older code -- move them
#     away first if the whole population must come from one code version;
#   * the GNU-parallel joblog is named joblog_full_<population>.tsv (separate from pilot joblogs).
# Everything else (paths on scratch, log layout, population guard, restart safety, run_one) is
# identical to run_popsynth_malanga_Hof.sh; see that file for the full description:
#   one run_model_Hof.py run per CSV row with --run_name hof_malanga_w0_<id> --M <M0 [Msun]>
#   --Rd <R0 [au]> --psi_DW <psi> --e_rad <e_rad> --alpha_SS <alpha_SS>, config DiscConfig_Hof_malanga.json.
#
# Output layout (scratch; same as the project output folder):
#     /scratch/imontesd/output/DiscEvolution/
#     |-- popsynth/malanga/                .h5 per disc, config_log/, population CSV + .json,
#     |                                    run_provenance_<jobid>.txt                          (OUTDIR)
#     `-- logs/malanga-comparison/         Slurm output, job list, joblog                     (LOGDIR)
#         `-- run_logs/                    per-disc .out/.err                                 (RUNLOGDIR)
# Scratch is PURGED after a while: once the job has FINISHED, copy the results to project:
#     rsync -av /scratch/imontesd/output/DiscEvolution/ /project/def-mbalogh/imontesd/output/DiscEvolution/
#
# Restart safety: run_model_Hof.py skips any disc whose output file exists with attr
# complete = True, so re-submitting this script only runs what is left. (Since the filename
# carries the COMPUTED initial Mdot, each skipped disc first rebuilds its initial disc, ~1 s.)
#
# Runtime (pilot, 16 discs incl. the extremes): median 0.14 h, max 1.2 h (compact disc); 500 discs on
# 192 slots -> ~1.5-2 h wall time; --time = 4 h leaves margin.
#
# Usage (Nibi; first: git pull, and mkdir -p the scratch folders above):
#   sbatch run_popsynth_malanga_full_Hof.sh                      # all discs in the population CSV
#   DRY_RUN=1 bash run_popsynth_malanga_full_Hof.sh | head       # check the commands, run nothing
#   (paths can still be overridden from the environment, e.g. OUTPUT_ROOT=..., for local tests)

set -euo pipefail

module load StdEnv/2023 python/3.12 2>/dev/null || true          # no-op where `module` does not exist (local tests)
VENV="${VENV:-/project/def-mbalogh/imontesd/DiscEvolution/.venv}"
if [[ -f "$VENV/bin/activate" ]]; then source "$VENV/bin/activate"; fi

# ---------------------------------------------------------------------------
# 1. Settings (each one overridable from the environment)
# ---------------------------------------------------------------------------
SCRIPT_DIR="${SCRIPT_DIR:-/home/imontesd/projects/def-mbalogh/imontesd/DiscEvolution/example/StartHere}"
CONFIG_FILE="${CONFIG_FILE:-$SCRIPT_DIR/config/popsynth/DiscConfig_Hof_malanga.json}"
DRAWS_CSV="${DRAWS_CSV:-$SCRIPT_DIR/config/malanga/malanga_draws_Hof_N500_seed1.csv}"   # from sample_malanga_Hof.py (population files live in config/malanga/)
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/imontesd/output/DiscEvolution}"   # (Hof, 2026-10-19) ALL outputs on scratch ($SCRATCH = /scratch/imontesd),
                                 # same layout as /project/def-mbalogh/imontesd/output/DiscEvolution
OUTDIR="${OUTDIR:-$OUTPUT_ROOT/popsynth/malanga}"           # .h5 files + config_log/ + population copy
LOGS_ROOT="$OUTPUT_ROOT/logs"   # shared logs folder: nothing is written DIRECTLY here
LOGDIR="${LOGDIR:-$LOGS_ROOT/malanga-comparison}"   # this sweep's folder: Slurm output, job list, joblog
RUNLOGDIR="$LOGDIR/run_logs"     # (Hof, 2026-10-19) per-disc .out/.err logs, one pair per disc
NPROC="${NPROC:-150}"            # runs at once; keep equal to --cpus-per-task above
# (Hof, 2026-10-20) no FIRST_ID / LAST_ID / IDS here: the full run always takes EVERY disc
for v in FIRST_ID LAST_ID IDS; do
    if [[ -n "${!v:-}" ]]; then
        echo "WARNING: $v='${!v}' is set in the environment but IGNORED: this script runs ALL discs." >&2
    fi
done
DRY_RUN="${DRY_RUN:-0}"          # 1 = print the run_model_Hof.py commands, run nothing

# ---------------------------------------------------------------------------
# 2. Pre-flight checks
# ---------------------------------------------------------------------------
for f in "$CONFIG_FILE" "$DRAWS_CSV" "$SCRIPT_DIR/run_model_Hof.py"; do
    [[ -f "$f" ]] || { echo "ERROR: not found: $f" >&2; exit 1; }
done
# (Hof, 2026-10-19) never write logs directly into the shared logs/ folder (LOGS_ROOT): LOGDIR must
# be a subfolder of it (or somewhere else entirely, e.g. a local test folder). realpath -m resolves
# trailing slashes, "..", and symlinks without requiring the folders to exist yet.
if [[ "$(realpath -m "$LOGDIR")" == "$(realpath -m "$LOGS_ROOT")" ]]; then
    echo "ERROR: LOGDIR = $LOGDIR is the shared logs folder itself; use a subfolder (default:" >&2
    echo "       $LOGS_ROOT/malanga-comparison). Check for an exported LOGDIR: echo \$LOGDIR" >&2
    exit 1
fi
mkdir -p "$OUTDIR" "$LOGDIR" "$RUNLOGDIR"

# Keep each population with its outputs, and never mix two populations in one OUTDIR. A population
# CSV already in OUTDIR is compatible with DRAWS_CSV only if one is a line-by-line PREFIX of the
# other (identical files included): same seed, only N changed. sample_malanga_Hof.py draws one row
# per disc, so raising N with the same seed keeps discs 0..N_old-1 identical and only appends new
# ones -- their finished runs are then reused (skipped). Anything else (other seed, other Table 2
# values, other alpha_SS, ...) is a DIFFERENT population -> stop and use a new OUTDIR.
CSV_BASE="$(basename "$DRAWS_CSV")"
is_prefix() {   # is_prefix A B: true if file A equals the first lines of file B
    head -n "$(wc -l < "$1")" "$2" | cmp -s - "$1"
}
for existing in "$OUTDIR"/malanga_draws_Hof_*.csv; do
    [[ -e "$existing" ]] || continue
    if ! is_prefix "$existing" "$DRAWS_CSV" && ! is_prefix "$DRAWS_CSV" "$existing"; then
        echo "ERROR: $OUTDIR already holds a different population ($(basename "$existing"))." >&2
        echo "       Use another OUTDIR for $CSV_BASE (or remove the old outputs first)." >&2
        exit 1
    fi
done
if [[ "$DRY_RUN" != 1 ]]; then                   # a dry run writes nothing into OUTDIR
    cp "$DRAWS_CSV" "$OUTDIR/"
    if [[ -f "${DRAWS_CSV%.csv}.json" ]]; then cp "${DRAWS_CSV%.csv}.json" "$OUTDIR/"; fi
fi

# (Hof, 2026-10-20) provenance: which code produced these files (the .h5 files do not record it)
GIT_COMMIT=$(git -C "$SCRIPT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)
GIT_BRANCH=$(git -C "$SCRIPT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
GIT_DIRTY=$(git -C "$SCRIPT_DIR" status --porcelain 2>/dev/null | wc -l)
PROVENANCE="job ${SLURM_JOB_ID:-local} on $(hostname), started $(date '+%F %T')
code: $SCRIPT_DIR  branch $GIT_BRANCH  commit $GIT_COMMIT  uncommitted changes: $GIT_DIRTY file(s)
config: $CONFIG_FILE
population: $DRAWS_CSV"
echo "$PROVENANCE"
if [[ "$GIT_DIRTY" -gt 0 ]]; then
    echo "WARNING: the code has $GIT_DIRTY uncommitted change(s); commit $GIT_COMMIT alone does not describe it." >&2
fi
if [[ "$DRY_RUN" != 1 ]]; then
    echo "$PROVENANCE" > "$OUTDIR/run_provenance_${SLURM_JOB_ID:-local}.txt"
fi

# (Hof, 2026-10-20) outputs already in OUTDIR are SKIPPED by run_model_Hof.py (if complete). Report them,
# and how many lack the GI attrs (= written by older code, e.g. the pilot), so a mixed-version population
# is a deliberate choice: move those files away before submitting if every disc must use this code.
python3 - "$OUTDIR" <<'EOF'
import glob, sys
import h5py
files = sorted(glob.glob(sys.argv[1] + "/hof_malanga_w0_*.h5"))
old = []
for f in files:
    try:
        with h5py.File(f, "r") as h:
            if "GI_flag" not in h.attrs:
                old.append(f.split("/")[-1].split("_gamma")[0])
    except OSError:
        pass                                    # unreadable (e.g. incomplete) -> will simply be re-run
print(f"Existing outputs in OUTDIR: {len(files)} (complete ones are skipped)")
if old:
    print(f"WARNING: {len(old)} of them have no GI attrs (older code): {', '.join(old[:20])}"
          + (" ..." if len(old) > 20 else ""))
EOF

# Job list: one line per disc, "run_name M0 R0 psi e_rad alpha_SS", read BY COLUMN NAME (so a
# change in the sampler's column order cannot pass the wrong value), values copied VERBATIM from
# the CSV (4 significant figures; 6 for e_rad). ALL rows (no selection in the full run).
JOBLIST="$LOGDIR/joblist_full_${CSV_BASE%.csv}.txt"
python3 - "$DRAWS_CSV" > "$JOBLIST" <<'EOF'
import csv, sys
path = sys.argv[1]
need = ["id", "run_name", "M0_Msun", "R0_au", "psi_DW", "e_rad", "alpha_SS"]
with open(path, newline="") as f:
    rows = csv.DictReader(f)
    missing = [c for c in need if c not in rows.fieldnames]
    if missing:
        sys.exit(f"ERROR: {path} has no column(s) {missing}")
    for r in rows:
        print(r["run_name"], r["M0_Msun"], r["R0_au"], r["psi_DW"], r["e_rad"], r["alpha_SS"])
EOF
NJOBS=$(wc -l < "$JOBLIST")
NROWS=$(( $(wc -l < "$DRAWS_CSV") - 1 ))          # CSV rows minus the header
if [[ "$NJOBS" -ne "$NROWS" || "$NJOBS" -eq 0 ]]; then
    echo "ERROR: job list has $NJOBS lines but the population CSV has $NROWS discs" >&2; exit 1
fi

echo "Config:     $CONFIG_FILE"
echo "Population: $DRAWS_CSV"
echo "Discs:      $NJOBS (ALL discs in the population), $NPROC at a time"
echo "Output to:  $OUTDIR"
echo "Logs to:    $LOGDIR  (per-disc .out/.err in $RUNLOGDIR)"
[[ "$DRY_RUN" == 1 ]] && echo "DRY RUN: commands are printed, nothing is run"
echo

# ---------------------------------------------------------------------------
# 3. One run per disc
# ---------------------------------------------------------------------------
# Arguments (order of the job-list columns):
#   $1 run_name   $2 M0 [Msun]   $3 R0 [au]   $4 psi_DW [-]   $5 e_rad [-]   $6 alpha_SS [-]
run_one() {
    local rn="$1" M="$2" Rd="$3" psi="$4" erad="$5" aSS="$6"
    local cmd=(python3 "$SCRIPT_DIR/run_model_Hof.py" --config "$CONFIG_FILE" --run_name "$rn"
               --M "$M" --Rd "$Rd" --psi_DW "$psi" --e_rad "$erad" --alpha_SS "$aSS"
               --output_dir "$OUTDIR")
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "${cmd[*]}"
        return 0
    fi
    echo "[$(date +%T)] Launching $rn (skips itself if already complete -- see run_logs/$rn.out)"
    # "|| ..." keeps one crashed run from stopping the sweep under set -e (traceback in .err);
    # `return 1` still reports the failure to GNU parallel's joblog.
    "${cmd[@]}" > "$RUNLOGDIR/$rn.out" 2> "$RUNLOGDIR/$rn.err" \
        || { echo "[$(date +%T)] FAILED $rn -- see $RUNLOGDIR/$rn.err"; return 1; }
}
export -f run_one
export SCRIPT_DIR CONFIG_FILE OUTDIR LOGDIR RUNLOGDIR DRY_RUN

if command -v parallel >/dev/null 2>&1; then
    # one disc per job-list line, columns -> {1}..{6}; starts the next disc as soon as a slot frees
    parallel -j "$NPROC" --colsep ' ' --joblog "$LOGDIR/joblog_full_${CSV_BASE%.csv}.tsv" \
        run_one {1} {2} {3} {4} {5} {6} :::: "$JOBLIST" || true    # failures are summarised below
else
    # Fallback without GNU parallel: background jobs, at most NPROC at once; `wait -n` (bash >= 4.3)
    # returns as soon as ANY job ends, so a new disc starts whenever a slot frees.
    echo "(GNU parallel not found -- using a bash job-control loop instead)"
    running=0
    while read -r rn M Rd psi erad aSS; do
        run_one "$rn" "$M" "$Rd" "$psi" "$erad" "$aSS" &
        running=$((running + 1))
        if ((running >= NPROC)); then wait -n || true; running=$((running - 1)); fi
    done < "$JOBLIST"
    wait || true
fi

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------
[[ "$DRY_RUN" == 1 ]] && exit 0
echo
echo "[$(date +%T)] All discs finished."
# per selected disc: is there an output file (any computed-Mdot token), and is its .err log non-empty?
# (a non-empty .err is not necessarily a failure -- e.g. numpy warnings -- but worth a look)
n_h5=0; n_err=0
while read -r rn _; do
    if compgen -G "$OUTDIR/${rn}_gamma*.h5" > /dev/null; then n_h5=$((n_h5 + 1)); fi
    if [[ -s "$RUNLOGDIR/$rn.err" ]]; then n_err=$((n_err + 1)); fi
done < "$JOBLIST"
echo "Output files present: $n_h5 / $NJOBS discs"
echo "Discs with a non-empty .err log (worth a look): $n_err"
JOBLOG="$LOGDIR/joblog_full_${CSV_BASE%.csv}.tsv"
if [[ -f "$JOBLOG" ]]; then
    # joblog column 7 = exit value, last column = the command (run_one <run_name> ...)
    awk -F'\t' 'NR > 1 && $7 != 0 {split($NF, c, " "); print "  FAILED (exit " $7 "): " c[2]}' "$JOBLOG"
fi
echo
echo "Next: copy the results from scratch to project before scratch is purged:"
echo "  rsync -av ${OUTPUT_ROOT}/ /project/def-mbalogh/imontesd/output/DiscEvolution/"
