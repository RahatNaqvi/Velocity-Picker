# Staged plan
0. (done) scaffold, docs.
1. Core data: SEG-Y scan/index (D2), header validation report, CMP index (existing + computed), gather extraction, supergather. Synthetic SEG-Y generator for tests.
2. Engine: NMO + semblance + parameter validation + cost estimate; known-answer synthetic tests (hyperbolic events, noise, multiples); benchmark.
3. Picking core (no GUI): pick model, undo/redo, navigation, copy-previous/selected. Then three-panel UI on top.
4. SQLite + versions + provenance + autosave/recovery + CSV export.
5. QC warnings, interpolation, NMO preview, neighbour overlay.
6. Real-data validation, existing-semblance importers (per samples), packaging, limitations log.
Each stage: runnable, tests run, results recorded in docs/VALIDATION.md.
