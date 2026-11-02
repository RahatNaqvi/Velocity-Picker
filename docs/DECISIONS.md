# Decisions needing owner input
D1 GUI toolkit: PySide6+pyqtgraph (recommended) vs Tkinter+matplotlib.
D2 SEG-Y backend: segyio (fast, C-backed; strict about some files) vs own reader on numpy memmap (full control, more tests).
D3 Initial geometry: 2D land line only, or 3D (CMP bins/inline-xline) from the start?
D4 Picked velocity type: stacking velocity (recommended default) — confirm; is Dix interval conversion wanted in v1?
D5 Target downstream package(s) for export, and a sample velocity file from each.
D6 Units: metres/ms internally; need feet support?
D7 Existing-semblance import: which package(s) produce the files you have?
D8 Supergather defaults and whether offset-duplicate handling is needed in v1.
D9 Target OS and Python version for packaging (Ubuntu/Python 3.12 assumed).
