# Validation status / known limitations

Last run: Stage 2 (reader, index, gather extraction, semblance engine). Results below are actual.

| Item | Status |
|---|---|
| Package layout / imports | smoke tests pass |
| SEG-Y reader (rev 0/1, big-endian, fixed length; formats 1,2,3,5,8) | synthetic round-trip tests pass; real file 700-750 reads correctly |
| IBM float decode | known values (-118.625, 1.0, 0) + round-trip within 2e-6 relative |
| Header index, CMP lookup, gaps, offset cross-check | synthetic tests + real file (6 CMPs, fold 344-398) pass |
| Gather extraction | passes on synthetic and CMP 700/750 of real file (finite, no dead traces) |
| Source file never modified | checksum/mtime tests pass |
| NMO, semblance, param validation, cost estimate | 22 synthetic known-answer + brute-force-reference tests pass |
| Conditioning, supergather, GeoEast-style stack power | 14 synthetic tests pass (73 total); see CMP 210 notes below |
| Semblance on real CMP 700-750 | runs (~1 s/CMP); panel dominated by data top-mute edge and noise; NOT validated, see notes |
| Supergathers, CMP binning from geometry | NOT implemented |
| Pick model, navigation, SQLite versions/drafts/named velocities/overwrite+audit/migrations v1->v2, file identity, CSV export | 32 + 12 tests pass (immutability, approved protection, audit, copy, per-name drafts, migration of a real v1 project, moved-folder re-attach) |
| Qt picking workstation (3 panels, pick/move/delete, undo, copy, save, recover) | 40 headless GUI tests pass incl. real mouse click and realistic-timing drags in every panel, time-axis limits, CVS picking, display options; run on real CMP 210-215 file (~10 s per CMP, full 6500 ms); NOT yet run on the user's desktop |
| Constant-velocity stacks | engine + GUI tests pass (event strongest in strip nearest true velocity); not compared with GeoEast stack segments |
| Interpolation between CMPs, version compare UI, supergather display, GeoEast export | NOT implemented |
| Comparison with trusted package results | NOT done (no trusted semblance supplied yet) |

## Known limitations
- Big-endian only; no variable-length traces; formats 4, 6, 7, 9+ unsupported (clear error raised).
- Units are inferred: coordinate-units flag is 0 in the test file; metres is supported only by the header-offset vs coordinate-distance agreement (<1 m).
- Index cache staleness uses file size + mtime; copying a file changes mtime and forces a rebuild (safe).

## Real-data notes (CMP 700, 2026-10)
- The file carries a top mute (zeros above a sloping line, ~450 ms at 1269 m to ~2550 ms at 7270 m offset).
  With default settings the strongest semblance lies on that mute edge, not on reflections.
- Restricting offsets (<= 4000 m), stretch 50 % and an 80 ms gate gave a coherent ~3200 m/s feature near 950 ms;
  deeper than ~1300 ms no clear peak. A 500 ms AGC outside the engine changed almost nothing.
- Cause (data state vs method limits) is undetermined. Needs the processing state of the file and a trusted
  semblance for comparison.

## Real-data notes: CMP 210-215 file (second dataset, 2 ms, 3500 samples, fold 600, offsets 16-15188 m)
- Data are already top-muted (first non-zero sample ~100 ms near offset, ~3800 ms at 14 km); all traces live.
- GeoEast-style stack power (bandpass 7-11-45-55, equalise 300 ms, 3-CMP groups) shows a ridge along the processor's
  picks between ~650 and ~1300 ms; deeper (1300-2300 ms, picks 3.4-4.1 km/s) no ridge is visible, and strong
  low-velocity (<2500 m/s) energy appears at all times.
- Quantitative check vs the 9 picks <= 2300 ms (percentile of row-normalised power at the pick velocity):
  median 44 % (1 CMP), 57 % (3 CMPs), 67 % (6 CMPs); random velocity functions give 50 % (95th pct 74 %).
  Conclusion: NOT statistically distinguishable from random on this metric; NOT validated.
- Resolved: the file's CDP numbers were relabelled by the user (original CDP 200-205 -> 210-215) so GeoEast would plot
  them; header coordinates belong to the original CDPs and are irrelevant. The gathers ARE the GeoEast input.
- Unknown: contents of the mute file, how many CMPs the job really used (manual 5.6b caps it by the CMP increment, 5),
  GeoEast's stack normalisation/smoothing.

### Comparison with the GeoEast CMP 210 spectrum recovered from a screenshot (colours -> relative values)
Axis calibration verified: predicted pick positions fall on the screenshot's pick dots. Recovered values are
relative display colours (jet inverse), approximate by nature.
- Inside GeoEast's own spectrum the processor's picks sit high (median 85th percentile of the best cell within +-120 m/s).
- My stack power, median per-row Spearman correlation with GeoEast for t >= 900 ms: 3 CMPs 0.41-0.51; 1 CMP 0.24-0.37
  (3 CMPs clearly better). Stretch 50 % best of 20/50/100 %. 300 ms equalisation made almost no difference.
- Whole-panel agreement is poor (raw Spearman negative, row-normalised ~0.2). Structural differences:
  GeoEast shows an almost dead zone ~200-900 ms for v < 5000 m/s with a sharp lower edge at ~900 ms and strong
  5000-6000 m/s energy at 300-900 ms; mine does not. GeoEast's deep part is uniformly bright (per-row display scaling).
- Cause unknown (candidates: how the mute is applied inside VelocityAna, stretch/NMO limits, display scaling).

## Note on testing mouse drags
pyqtgraph ignores scene mouse-move events arriving faster than its `mouseRateLimit` (100 per second). Synthetic events fired
back-to-back are therefore silently dropped and a drag never starts - this made an early test script report "drag does not move
the view" although it does. Drag tests now use realistic 12 ms spacing. (A drag failure reported by the user on a real desktop
could not be reproduced; touch-pad scrolling, arrow keys and exact view boxes were added as alternatives.)

## Schema v2 (named velocities) - verified on a real project
A project database written by the previous program (schema v1, picks on CMP 210 and 212) was opened with the new code: migrated to v2,
all versions and picks intact, a `.bak-v1` backup written, foreign-key check clean. A copied project folder (new path) initially showed
its picks as missing because files were identified by path; files are now identified by content (headers hash + size) and old duplicate
entries are merged; verified on the real copied project.
