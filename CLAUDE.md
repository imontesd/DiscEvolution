# CLAUDE.md

Guidance for Claude Code when working in this repository.

---

## User guidelines

<!-- Add your permanent instructions here. They are loaded into context every session. -->

- Never make changes to any files outside of home/imontesd/DiscEvolution. Only make direct changes to files with Hof in their filename (i.e. run_model_Hof.py, Hof_figures.ipynb, DiscConfig_Hof_baseline.json, etc.)
- When making direct changes, include extense comments so that it is easy for the user to review changes made. Include units whenever relevant for traceability
-

---

## Project context

DiscEvolution is a protoplanetary-disc evolution code (gas viscous + disc-wind
transport, two-population dust growth + radial drift, C/O chemistry, planetesimal
formation, Bitsch planet growth/migration).

### Isaac's Hof Co-op project (Fall 2026)

Work lives in `example/StartHere/`:

| File | Purpose |
|---|---|
| `run_model_Hof.py` | Modified copy of `run_model_student.py` for this project. Change log is in its module docstring, keep up to date with any changes with respect to run_model_student.py |
| `config/DiscConfig_Hof_baseline.json` | Baseline run config. `grid.type = "winds-alpha"`. |
| `notebooks/Hof_figures.ipynb` | Analysis / figures. `load_data()` reads the HDF5 output; helper cell defines the post-processing functions. Keep well-commented, especially with any new changes|
| `HOF_PROJECT_GUIDE.md` | Project guide (referenced from the notebook). Do not make any changes to this file, use as reference only|

`run_model_Hof.py` writes `erad` into the output filename (`..._psiX_eradY_MdotZ_...h5`)
so it can be recovered for the wind mass-loss-rate calculation. The config key is
`e_rad` (underscore); the filename token is `erad` (no underscore). `load_data()`
regex-extracts both `psi` and `erad` from the filename.

## Units (the thing that trips everyone up)

Code works in `G = AU = Msun = 1`; the time unit is *not* seconds/years — `yr = 2π`
is the conversion factor. See `DiscEvolution/constants.py` (everything CGS) and the
`run_model_Hof.py` module docstring.

HDF5 output datasets:

| Dataset | Units |
|---|---|
| `t` | **years** |
| `time_snap` | **Myr** |
| `disk_Mass` | **grams** (divide by `Msun`) |
| `disk_Mdot_star`, `disk_Mdot_p` | M☉/yr |
| `Sigma_G`, `Sigma_dust`, `Sigma_pebbles` | g cm⁻² |
| `T` | K |
| `Sigma_pebble_size` | cm |
| `Vdrift` | code units (AU per 2π yr); ÷2π → AU/yr |
| `R` | AU |
| chemistry abundances | dimensionless mass ratio relative to H (not normalised) |
| planet `Mcs`/`Mes`/`M_iso`/`M_transition` | M⊕ |

`disk_Mass` and `t` are the two easy-to-misuse ones: `t` is years while
`time_snap` is Myr; `disk_Mass` is grams while `disk_Mdot_star` is already M☉/yr.

Profile datasets (`Sigma_G`, `T`, ...) have **one row per `time_snap` entry**, not
per `t`. Plot `Mgas_tsnap`/`R50_tsnap` vs `t_snap`, and `Mdisc_t`/`Mdot_t` vs `t`.

## Environment gotchas

- **NumPy is 2.5.2** in `.venv` — `np.trapz` was removed; use `np.trapezoid`.
- The venv Python is `/home/imontesd/DiscEvolution/.venv/bin/python`.
- **VS Code does not hot-reload `.ipynb` files.** After Claude edits the notebook,
  close/reopen it or run "Revert File" before running cells — and don't save from a
  stale editor tab (it clobbers the edit).

## Conventions

- Notebook variables use a `_tsnap` suffix for series aligned to `time_snap`.
- `run_model_Hof.py::output_filename()` is the single source of truth for output
  filenames. Config keys consumed there (`psi_DW`, `e_rad`, `Mdot`, `M`, `Rd`,
  and `alpha` for `winds-alpha`) are all required — a missing one is a `KeyError`.
