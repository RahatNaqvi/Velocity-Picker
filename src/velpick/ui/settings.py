"""Analysis settings shown in the dialog and stored with every saved version (provenance)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields


@dataclass
class AnalysisSettings:
    # The defaults mirror the user's GeoEast job (stack power, 3 CMPs, 1500-6000 m/s at 40 m/s, bandpass
    # 7-11-45-55, 300 ms equalisation). They are starting values, not universal processing defaults.
    method: str = "stackpower"            # "stackpower" or "semblance"
    vmin: float = 1500.0
    vmax: float = 6000.0
    dv: float = 40.0
    t0_start: float = 100.0
    t0_end: float = 6500.0                 # clamped to the record length; GeoEast job used 6500
    n_cmps: int = 3                        # CMPs used for one spectrum (stack power; multiples of 3 above 3)
    line_interval: float = 25.0            # spectrum line spacing, ms
    stretch: float = 50.0                  # NMO stretch mute %, stack power (assumption, see docs/ENGINE.md)
    filter2d: bool = True
    bandpass: bool = True
    f1: float = 7.0
    f2: float = 11.0
    f3: float = 45.0
    f4: float = 55.0
    eq_ms: float = 300.0                   # equalisation window, 0 = off
    window_ms: float = 40.0                # semblance gate
    sem_stretch: float = 30.0              # NMO stretch mute %, semblance
    row_norm: bool = True                  # display only: divide each time line by its mean/max
    max_offset: float = 10000.0            # traces beyond this |offset| are not used (0 = no limit); GeoEast job: 10000
    cvs_enabled: bool = True               # stack-segment panel (constant-velocity stacks)
    cvs_strips: int = 21                   # number of segments = constant velocities (GeoEast: 'number of stack segments')
    cvs_cmps: int = 3                      # traces per segment = neighbouring CMPs stacked separately (GeoEast job: 25)
    cvs_vmin: float = 0.0                  # velocity range of the segments; 0 = use the analysis vmin / vmax
    cvs_vmax: float = 0.0
    cvs_dt_ms: float = 8.0                 # time sampling of the stacks (display only)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

    def key(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class DisplaySettings:
    """How the panels LOOK (never changes the data, the analysis or the saved picks). Stored in the project database."""
    # ---- gather panel
    g_mode: str = "Wiggle + VA"            # or "Image"
    g_bandpass: bool = False               # display filter on a copy (Ormsby, Hz)
    g_f1: float = 7.0
    g_f2: float = 11.0
    g_f3: float = 45.0
    g_f4: float = 55.0
    g_agc_ms: float = 500.0                # 0 = off
    g_gain: float = 1.0                    # amplitude multiplier (bigger = bigger wiggles / darker image)
    g_clip: float = 98.0                   # percentile that maps to full amplitude
    g_max_traces: int = 90                 # wiggle traces drawn (evenly decimated)
    # ---- velocity-stack panel
    c_bandpass: bool = False
    c_f1: float = 7.0
    c_f2: float = 11.0
    c_f3: float = 45.0
    c_f4: float = 55.0
    c_agc_ms: float = 500.0
    c_gain: float = 1.0
    c_clip: float = 98.0
    c_axis: str = "velocity"               # top-axis labels: "velocity" (m/s per segment) or "trace" (GeoEast style: 1 at each segment)
    # ---- spectrum panel
    s_cmap: str = "jet"                    # jet, viridis, inferno, magma, gray, seismic
    s_clip_low: float = 0.0                # percentile mapped to the lowest colour
    s_clip_high: float = 99.5              # percentile mapped to the highest colour
    s_gamma: float = 1.0                   # colour contrast curve
    s_pick_size: float = 10.0
    # ---- colours (hex)
    col_pick: str = "#ffffff"              # current picks on the spectrum
    col_prev: str = "#00dcff"              # previous CMP's picks
    col_next: str = "#ff00ff"              # next CMP's picks
    col_saved: str = "#ffffff"             # last saved version (dotted, while editing)
    col_cvs: str = "#e61e1e"               # triangles on the velocity stacks
    neigh_width: float = 3.0
    # ---- layout / mouse
    w_spec: int = 5                        # relative panel widths
    w_gather: int = 5
    w_cvs: int = 5
    wheel_scrolls: bool = False            # False: wheel zooms time, Shift scrolls; True: wheel scrolls, Ctrl zooms time

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DisplaySettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})
