# Velocity-analysis engine (Stage 2)

Units: time ms, offset m, velocity m/s. The engine cannot detect file offsets in feet; convert first.

## What is computed
Picked quantity = **stacking (NMO) velocity** under the hyperbolic model tau = sqrt(t0^2 + x^2/v^2).
It is *not* RMS or interval velocity. RMS ~ stacking velocity only for small offset / layered earth;
interval velocity needs a separate, explicit Dix conversion (not implemented).

## Steps (per trial velocity v)
1. Select traces: not dead (all-zero), within optional |offset| limits.
2. NMO (`nmo.py`): tau for every (trace, t0); source sample (tau - t_start)/dt; linear interpolation.
   Invalid (set to 0, not counted) where tau is outside the record or (tau - t0)/t0 > stretch mute %.
3. Semblance (`semblance.py`): S = sum_gate(sum_i a)^2 / sum_gate(n * sum_i a^2), n = live traces per sample;
   boxcar gate of 2h+1 output samples (h = round(window / (2*step))), truncated at the ends.
   S in [0,1]; S = 0 where live traces < `min_live_traces` (a single trace would give S = 1).
4. Optional boxcar smoothing along time/velocity (off by default, explicit parameters).

## Parameter safeguards (`validate_params`)
Errors block the run (invalid ranges, t0 outside record, too few traces, ...). Warnings do not
(short gate, large stretch %, dv coarse relative to gate, poor far-offset moveout). There are no default
velocity/time ranges or gate: they are survey dependent. `estimate_cost` returns counts and memory, not time.

## Known limitations
- Hyperbolic, single stacking velocity per (t0); no anisotropy/eta, no azimuth handling (wide-azimuth data
  are treated by |offset| only).
- Linear interpolation lowers high-frequency peak amplitude (up to ~10 % for 30 Hz at 4 ms) but not peak position.
- Semblance is energy weighted: strong coherent noise or a sharp data mute dominates the panel.
- No pre-conditioning (AGC, bandpass, noise attenuation) inside the engine.
- Zero (muted) samples in the *data* are treated as data; a data top-mute edge produces a bright arc.
- Time of the maximum is only defined to about one gate length.

## Validation status
- Synthetic known-answer tests: single/multiple events, noise, polarity, dead traces, offset limits, mute,
  resolution behaviour, parameter validation, cancellation.
- Independent slow brute-force implementation in the tests agrees with the vectorised code (atol 2e-4).
  This checks implementation consistency, **not** agreement with any commercial package.
- Real data (CMP 700-750 of line 42): runs in ~1 s per CMP (344-398 traces, 91 velocities, 976 times);
  results are NOT yet validated against trusted semblance (see VALIDATION.md).

## GeoEast-style stack-power spectrum (`stackpower.py`, `conditioning.py`)
Source: GeoEast VelocityAna 3.43 manual (supplied by the user). Implemented from what the manual states:
two-step stacking (groups of 3 adjacent CMPs summed -> absolute value -> matrices added), bandpass then
equalisation (300 ms), spectrum lines every 25 ms (> 6 x sample interval), optional 2D filter, CMP-count rules
(capped by the CMP interval to the next spectrum, multiples of 3 above 3), top-mute interpolation/extrapolation rule.
**Not stated in the manual, therefore assumptions (parameters, flagged in code):** stack normalisation (mean over
live traces), smoothing window (one line interval), 2D-filter kernel (3 x 3), NMO-stretch limit (50 %), display gain
(row mean), order of mute vs filtering. Result: a GeoEast-*like* spectrum, not a numerical reproduction.
Filter operator length (400 ms) is not modelled (FFT zero-phase filter).

## Velocity function fields seen in GeoEast Velocity Browser (screenshot, not a format spec)
Per function: name, version, velocity type (Stack), datum type (Floating Datum), domain (PP Time), replacement
velocity (2000), owner, modified date; per CMP: X, Y, CMP datum statics (ms), final velocity (m/s); table of
Time(ms) / Velocity(m/s) / eta / gamma0. Used only as design guidance for the export adapter; the actual file
format must come from a real exported file.
