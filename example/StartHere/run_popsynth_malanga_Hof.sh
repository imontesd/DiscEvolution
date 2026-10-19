#!/bin/bash
#SBATCH --account=def-mbalogh
#SBATCH --time=0-08:00:00            # whole population (all waves); see "Runtime" below
#SBATCH --cpus-per-task=192          # must equal NPROC below, and fit on ONE node
#SBATCH --mem-per-cpu=3800M          # 3.8 GB per CPU -> 192 x 3.8 GB = 730 GB <= 748 GB (one node)
#SBATCH --job-name=hof_malanga
#SBATCH --output=/project/def-mbalogh/imontesd/output/DiscEvolution/logs/malanga-comparison/%x_%j.out   # this script's own echo output (folder must exist BEFORE sbatch, see Usage)
#SBATCH --mail-user=imontesdeocahof@gmail.com
#SBATCH --mail-type=END,FAIL
#
# run_popsynth_malanga_Hof.sh
# ---------------------------
# (Hof, 2026-10-19) Launch the Malanga et al. (2026) MHD-wind disc population: one
# run_model_Hof.py run per row of the CSV written by sample_malanga_Hof.py, up to $NPROC at a
# time. Each row is one randomly drawn disc (M* = 1 Msun) with its own
#     --run_name hof_malanga_w0_<id>   --M <M0 [Msun]>   --Rd <R0 [au]>
#     --psi_DW <psi [-]>   --e_rad <e_rad [-]>   --alpha_SS <alpha_SS [-]>
# (see sample_malanga_Hof.py for how these are drawn and mapped). The config
# (DiscConfig_Hof_malanga.json) fixes everything else: solve_for = "none" with alpha_SS held fixed
# (TOTAL alpha = alpha_SS (1 + psi)), no initial-Mdot restriction, 34 snapshots to 3.0 Myr.
#
# Differences from run_popsynth1.00_hof.sh:
#   * the parameters come from a CSV (one row per disc), not from nested loops over a grid;
#   * every setting below can be overridden from the environment (VAR=value ./script.sh), so the
#     same file is used for local tests and the Step 6 pilot without editing it;
#   * FIRST_ID / LAST_ID run only part of the population (pilot, or discs added by raising N);
#   * DRY_RUN=1 prints the commands without running them;
#   * ALL logs go to logs/malanga-comparison/ (LOGDIR), never directly into logs/ itself:
#       logs/malanga-comparison/            this script's own Slurm output (hof_malanga_<jobid>.out),
#                                           the job list and the GNU-parallel joblog
#       logs/malanga-comparison/run_logs/   (Hof, 2026-10-19) one .out/.err per disc, named after it,
#                                           e.g. hof_malanga_w0_0042.out/.err (RUNLOGDIR)
#     The script stops if LOGDIR resolves to the top-level logs/ folder (LOGS_ROOT), e.g. because a
#     LOGDIR variable exported in the submitting shell overrode the default;
#   * GNU parallel writes a joblog (exit code + wall time per disc), independent of the HDF5 attrs;
#   * the fallback loop (no GNU parallel) keeps all NPROC slots busy with `wait -n` instead of
#     waiting for a whole wave to finish;
#   * the population CSV (+ .json) is copied into OUTDIR, and the script refuses to write a
#     DIFFERENT population into an OUTDIR that already holds one.
#
# Restart safety: run_model_Hof.py skips any disc whose output file exists with attr
# complete = True, so re-submitting this script only runs what is left. (Since the filename
# carries the COMPUTED initial Mdot, each skipped disc first rebuilds its initial disc, ~1 s.)
#
# Runtime: popsynth1.00 runs to 3 Myr with total alpha ~1e-3 - 1e-2 took ~0.3-0.5 h each (median);
# the Malanga discs have total alpha ~2e-3 - 1.1e-2, so expect ~0.3-0.5 h per disc. With
# N = 500 and NPROC = 192 that is 3 waves (192 + 192 + 116), ~1.5 h; --time = 8 h is a wide margin.
# The Step 6 pilot (FIRST_ID=0 LAST_ID=9) measures the real value.
#
# Usage:
#   sbatch run_popsynth_malanga_Hof.sh                                # whole population on Nibi
#   FIRST_ID=0 LAST_ID=9 sbatch run_popsynth_malanga_Hof.sh           # pilot: discs 0-9 only
#   DRY_RUN=1 NPROC=2 OUTDIR=/tmp/x LOGDIR=/tmp/x/logs bash run_popsynth_malanga_Hof.sh   # local check
#   (sbatch passes the submitting shell's environment to the job by default, so the VAR=value
#    prefixes work with sbatch too.)

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
OUTDIR="${OUTDIR:-/project/def-mbalogh/imontesd/output/DiscEvolution/popsynth/malanga}"
LOGS_ROOT="/project/def-mbalogh/imontesd/output/DiscEvolution/logs"   # shared logs folder: nothing is written DIRECTLY here
LOGDIR="${LOGDIR:-$LOGS_ROOT/malanga-comparison}"   # this sweep's folder: Slurm output, job list, joblog
RUNLOGDIR="$LOGDIR/run_logs"     # (Hof, 2026-10-19) per-disc .out/.err logs, one pair per disc
NPROC="${NPROC:-192}"            # runs at once; keep equal to --cpus-per-task above
FIRST_ID="${FIRST_ID:-}"         # first disc id to run (inclusive); empty = from the start of the CSV
LAST_ID="${LAST_ID:-}"           # last disc id to run (inclusive); empty = to the end of the CSV
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

# Job list: one line per disc, "run_name M0 R0 psi e_rad alpha_SS", read BY COLUMN NAME (so a
# change in the sampler's column order cannot pass the wrong value), values copied VERBATIM from
# the CSV (4 significant figures; 6 for e_rad), filtered to FIRST_ID..LAST_ID.
JOBLIST="$LOGDIR/joblist_${CSV_BASE%.csv}.txt"
python3 - "$DRAWS_CSV" "$FIRST_ID" "$LAST_ID" > "$JOBLIST" <<'EOF'
import csv, sys
path, first, last = sys.argv[1], sys.argv[2], sys.argv[3]
need = ["id", "run_name", "M0_Msun", "R0_au", "psi_DW", "e_rad", "alpha_SS"]
with open(path, newline="") as f:
    rows = csv.DictReader(f)
    missing = [c for c in need if c not in rows.fieldnames]
    if missing:
        sys.exit(f"ERROR: {path} has no column(s) {missing}")
    for r in rows:
        i = int(r["id"])
        if (first and i < int(first)) or (last and i > int(last)):
            continue
        print(r["run_name"], r["M0_Msun"], r["R0_au"], r["psi_DW"], r["e_rad"], r["alpha_SS"])
EOF
NJOBS=$(wc -l < "$JOBLIST")
[[ "$NJOBS" -gt 0 ]] || { echo "ERROR: no discs selected (FIRST_ID=$FIRST_ID, LAST_ID=$LAST_ID)" >&2; exit 1; }

echo "Config:     $CONFIG_FILE"
echo "Population: $DRAWS_CSV"
echo "Discs:      $NJOBS (ids ${FIRST_ID:-start}..${LAST_ID:-end}), $NPROC at a time"
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
    parallel -j "$NPROC" --colsep ' ' --joblog "$LOGDIR/joblog_${CSV_BASE%.csv}.tsv" \
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
echo "[$(date +%T)] All selected discs finished."
# per selected disc: is there an output file (any computed-Mdot token), and is its .err log non-empty?
# (a non-empty .err is not necessarily a failure -- e.g. numpy warnings -- but worth a look)
n_h5=0; n_err=0
while read -r rn _; do
    if compgen -G "$OUTDIR/${rn}_gamma*.h5" > /dev/null; then n_h5=$((n_h5 + 1)); fi
    if [[ -s "$RUNLOGDIR/$rn.err" ]]; then n_err=$((n_err + 1)); fi
done < "$JOBLIST"
echo "Output files present: $n_h5 / $NJOBS selected discs"
echo "Selected discs with a non-empty .err log (worth a look): $n_err"
JOBLOG="$LOGDIR/joblog_${CSV_BASE%.csv}.tsv"
if [[ -f "$JOBLOG" ]]; then
    # joblog column 7 = exit value, last column = the command (run_one <run_name> ...)
    awk -F'\t' 'NR > 1 && $7 != 0 {split($NF, c, " "); print "  FAILED (exit " $7 "): " c[2]}' "$JOBLOG"
fi
