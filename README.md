# velpick — Seismic Velocity Picking Workstation

**Status:** first working picking workstation (see `docs/GUI.md`). Spectra are not yet validated against GeoEast; see `docs/VALIDATION.md`.

## Documentation

- `docs/ARCHITECTURE.md` — assessment, architecture, risks
- `docs/ENGINE.md` — velocity analysis engine assumptions
- `docs/DECISIONS.md` — items needing owner input
- `docs/PLAN.md` — staged plan
- `docs/SAMPLE_DATA.md` — files needed for validation
- `docs/VALIDATION.md` — known limitations and status log
- `docs/GUI.md` — picking workstation GUI

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,plot]"
pytest
```

## Command-line tools

Run from the project folder with the venv active. Values are in ms, m and m/s.

```bash
python -m velpick.data_io.inspect_cli /full/path/to/file.sgy

python scripts/run_stackpower.py /full/path/to/file.sgy \
    --cmps 210 211 212 --vmin 1500 --vmax 6000 --dv 40 \
    --t0-start 100 --t0-end 2500 \
    --picks examples/picks_cmp210.txt --out cmp210.png

python scripts/run_semblance.py /full/path/to/file.sgy \
    --cmp 700 --vmin 1500 --vmax 6000 --dv 50 \
    --t0-start 100 --t0-end 4000 --window 40 --out cmp700.png
```

## Picking workstation (GUI)

```bash
pip install -e ".[gui,dev,gui-test,plot]"
velpick /full/path/to/gathers.sgy --line 42
```

Options:

- `--line LINE` — line number label for the data (default 0)
- `--db DB` — project database path (default: next to the SEG-Y file)

The SEG-Y file is opened read-only. Picks are stored in the project database.

If Qt reports it cannot load the `xcb` platform plugin on Ubuntu:

```bash
sudo apt install libxcb-cursor0
```
