# Architecture (proposal, Stage 1)

## Layering rule
`velocity_analysis`, `geometry`, `gather_manager`, `picking`, `qc`, `export`, `database`, `data_io`
are **pure Python/numpy, no GUI imports** (enforced by a test). `ui` and `visualization` depend on them,
never the reverse. Domain objects are plain dataclasses.

## Core domain objects
- `TraceIndex` — per-trace header table (structured numpy array / memmap): trace seq, FFID, CMP, offset,
  sx, sy, rx, ry, scalars, etc. Built once by a scan pass, cached next to the project (not next to the SEG-Y).
- `Gather` — samples [ntr x nt], dt, offsets (signed, metres), cmp ids, contributing trace ids, flags.
- `AnalysisParams` — vmin, vmax, dv, t0 start/end, window length, smoothing, NMO stretch-mute %,
  interpolation, offset range/mute, supergather spec. Hashable; stored with every result.
- `Semblance` — array [nt0 x nv], axes, params hash, input-data fingerprint.
- `VelocityFunction` — identity (location, velocity type, units), ordered `Pick(t_ms, v)`; versions are immutable rows.

## Velocity-analysis engine (`velocity_analysis`)
Method v1: **hyperbolic-NMO semblance** (Neidell & Taner 1971).
For each trial v and zero-offset time t0: t_i = sqrt(t0^2 + x_i^2 / v^2); amplitude a_i(t_i) by linear
interpolation (option: cubic); NMO-stretch mute where (t_i - t0)/t0 > threshold;
S(t0, v) = sum_{window} (sum_i a_i)^2 / ( N * sum_{window} sum_i a_i^2 ), N = live traces after mute.
Smoothing is an optional, explicit, parameterised step. Offsets use header offset with scalar/units checked;
no sign assumption (x^2 only). Vectorised per-v over t0 (memory ~ ntr*nt per v), chunked over v; optional numba later.
**Velocity picked = stacking (NMO) velocity under the hyperbolic assumption.** It approximates RMS velocity only for
small offset/layered-earth assumptions; interval velocity is *derived* (Dix) in a separate, explicit, opt-in function
and is never stored as the picked type. Every function and export carries `velocity_type`.
Other methods (e.g. non-hyperbolic / eta) are out of scope for v1; the engine interface is a registry so they can be added.

## Geometry / gather preparation
CMP from header if trusted; otherwise computed from coordinates (apply scalar, check units) and binned with
user-set bin size/azimuth for 2D line. Source gathers -> CMP: index-only sort (no data copy); gathers extracted lazily by
reading just the needed traces. Validation report lists missing/ambiguous headers *only if they affect picking*.
CMP ids are taken from the actual index (non-contiguous OK); requested-but-missing ids are reported.
Supergather: neighbour window defined by number of CMPs *or* distance, separate from analysis spacing; contributing
CMPs displayed; optional offset-duplicate handling (sum/mix/none) is a user choice. Lateral dip warning is a QC aid, not automatic.

## Persistence (`database`)
SQLite (WAL, foreign keys on, `PRAGMA user_version` + ordered migration scripts, backup before migrating).
Tables: project, survey, source_file (path, size, mtime, hash of headers), line, location (cmp, x, y, status),
analysis_config (JSON + hash), semblance_ref (optional cache path), velocity_function (identity),
function_version (immutable, parent_version, provenance: manual|copied|interpolated|imported, source ids, approval, notes, timestamp),
pick (version_id, t_ms, v, ord), review_note, draft (autosave). Approved versions are never updated in place; edits create new versions.

## UI (`ui`, `visualization`)
Three linked panels sharing one `SelectionModel` (cursor t/v, selected pick id). Background work via worker threads/processes
with progress + cancel; results arrive by signal/queue. Undo/redo via command pattern in `picking` (UI-independent).

## Export
`ExportAdapter` interface: `name, velocity_type, write(functions, options)`. CSV adapter first (header block with project,
velocity type, units, version ids). Package-specific adapters only after real spec/sample files exist.

## Key technical risks
1. Header chaos in field SEG-Y (scalars, offsets in feet/m, bad CMP) -> validation + user override.
2. Interactive rendering speed of wiggle/VA + large semblance image -> GUI toolkit choice (D1).
3. Semblance cost: ntr x nt x nv; supergathers multiply it -> estimator + chunking + cache.
4. Correctness of semblance vs reference -> synthetic known-answer tests + comparison with a trusted package output (needs your data).
5. Import of third-party semblance/velocity files: formats unknown until samples are seen.
6. Version/provenance integrity under crashes -> transactions + draft recovery.
