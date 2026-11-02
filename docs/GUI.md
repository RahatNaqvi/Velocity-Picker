# velpick workstation - how to use

Start:  `velpick /path/to/gathers.sgy --line 42`   (or `python -m velpick.ui /path/to/gathers.sgy --line 42`)
Created next to the SEG-Y (the SEG-Y itself is only ever read): `<name>.velpick.sqlite` (picks, versions) and
`<name>.velpick-index.npz` (header index cache, safe to delete). `--db` chooses another database path.

## Panels (one shared time axis, ms; velocity axes on top, m/s) - laid out like the usual processing displays
1. Spectrum (stack power or semblance): **pick here**. The previous CMP's saved picks are drawn on it in cyan (circles), the
   next CMP's in magenta (squares), with a legend ("Show neighbour picks", on by default); while you edit, the last saved
   version of this CMP is a white dotted line.
2. Gather (CMP, sorted by offset): wiggle + variable area (default) or image, clip %, display AGC, **NMO preview**
   (applies the current picks to a copy for display only; data and database are never touched).
3. Velocity stacks (constant-velocity stacks): one strip per velocity (default 21), one stack trace per contributing CMP; a
   triangle marks each pick. **Pick here too** (click = add at the strip's velocity, in between strips interpolated).

## Zooming and moving the time window
Mouse wheel over any panel = zoom TIME around the cursor (all panels follow); Ctrl+wheel = zoom velocity; Shift+wheel = scroll.
**Touch-pad users:** View > Display options > "Wheel scrolls time" makes a two-finger swipe scroll the window and Ctrl+swipe zoom.
Other ways to move the zoomed window: drag on empty space in any panel (vertical only unless you zoomed the velocity axis),
drag on the time axis, the vertical scroll bar at the right, Up / Down arrow keys (after clicking in the plots), or type exact
values in Display options > "Time window from / to".
Horizontal (velocity / offset / strips): a horizontal scroll bar under each panel (active when that panel is zoomed horizontally with
Ctrl+wheel), Left / Right arrow keys, Alt+wheel or a sideways touch-pad swipe, and the velocity window boxes in Display options. View > Reset view (Home) zooms back out; the view can never leave the
analysed range. Side panel (locations, picks table, warnings, note): fixed width; F4 or the toolbar button hides / shows it.

## Display options (View > Display options, F6) - look only, saved with the project
Gather and Velocity stacks: display bandpass (Ormsby, 4 corner frequencies), AGC window (0 = off), amplitude gain, clip %,
number of wiggle traces. Spectrum: colour map (jet, viridis, inferno, magma, gray, seismic), lowest / highest colour
percentiles, contrast (gamma), pick marker size. Colours: current picks, previous CMP picks, next CMP picks, last saved
version, stack triangles (colour chooser), neighbour line width. Layout and view: relative panel widths, wheel mode, exact
time and velocity window. These change only how things are drawn: the data, the analysis and the saved picks are never touched.
(The analysis conditioning - bandpass / equalisation that goes into the spectrum - is under Analysis > Settings.)

## Picking
Left click = add pick. Left drag on a pick = move it (one undo step). Right click on a pick = delete. Left drag
elsewhere = pan, wheel = zoom. Table at left: edit time/velocity numerically. Delete = remove selected, Ctrl+Z / Ctrl+Y.
Warnings (large jump, reversal, outside analysed range) are shown but never block.

## CMPs
PgDown / PgUp = next / previous CMP in the analysis sequence, Ctrl+G jump, CMP box, or click the location list
(unprocessed / needs_review / picked / reviewed / approved). Navigate > Set analysis sequence: start, end, interval
(actual CMP ids only; missing ones are reported). Unsaved work is kept as a draft (autosave every 30 s and on
navigation) and offered for recovery.

## Copy
Pick > Copy from previous CMP (Ctrl+Shift+P) / Copy from selected CMP. Copies are marked **needs_review** and are only
written to the database when you save; the source function is never changed. Interpolation between CMPs: NOT built yet.

## Where picks are saved, and velocity names
Everything is saved in ONE SQLite file next to the SEG-Y: `<segy name>.velpick.sqlite` (File > "Show where picks are saved..." shows the
full path and can copy it; it is also the tooltip of the Velocity box). The SEG-Y itself is only ever read. Delete
`<segy name>.velpick-index.npz` freely: it is just a cache.
A project holds one or more NAMED VELOCITIES (like `velo1` in GeoEast). The first one is called `default`. The active name is
shown in the toolbar (Velocity box), in the title bar `[name]` and in the info box; each name has its own functions and its own
numbered version history for every CMP, so the same CMP can carry different functions in different names.
The same data is recognised by its CONTENT, not its path: if you move, copy or rename the folder (or open the file from another
mount point) the picks follow; the info box says so.

Save (toolbar button; the arrow next to it and the Velocity menu show all three):
- **Save new version** (Ctrl+S): a new numbered version of this CMP in the active name. Nothing is overwritten. (default)
- **Overwrite latest version** (Ctrl+Alt+S): replaces the picks of this CMP's latest version after you confirm. The old picks are
  kept in an audit log. **Approved versions are protected** and cannot be overwritten (save a new version instead).
- **Save as new velocity name...** (Ctrl+Shift+S): asks for a name (+ description); optionally copies the latest function of every
  CMP of the current name into it (copies are marked needs_review); makes it active and saves the picks in the editor into it.
  The old name keeps exactly what it had. Names must be unique (case-insensitive) and file-name safe.
Velocity menu also: New empty velocity name, Rename, Manage (list with counts and last change; switch), Versions / restore.
Switching names keeps unsaved edits as a draft of the name you leave. Names are saved when you reopen the project.
Pick > Versions / restore loads an older version into the editor (saved as a new version).
File > Export CSV: choose which velocity name; the file name defaults to `<segy>_<name>.csv` and the header and last column
name it. Package-specific (e.g. GeoEast) export needs a real exported sample: NOT built.

## Analysis
Analysis > Recompute (F5) recalculates the current CMP's spectrum ignoring the cache; it is only needed if you suspect a
stale result (changing Settings recomputes automatically).
Analysis > Settings: method (stackpower = GeoEast-style, or semblance), velocity range/step, times, bandpass,
equalisation, CMPs per spectrum, stretch mute. Defaults copy the user's GeoEast job and are starting values only.
Defaults: 100 to 6500 ms (clamped to the record), offsets up to 10000 m, as in the user's GeoEast job. Computation runs in the
background (Esc cancels, F5 recomputes); on the 2 ms / 3500-sample test data a new CMP takes about 10 s with the full
6500 ms range (spectrum + 21 stack strips) - reduce t0 end, strips or the offset limit in Settings for speed. Results are
cached per CMP and settings; there is no pre-computation of the next CMP yet.

## Status of this version (see VALIDATION.md)
Implemented and tested headlessly with synthetic data (including a real mouse click) and exercised on CMP 210-215.
NOT verified on the user's desktop yet. NOT built: interpolation between CMPs, version-compare screen, supergather
display, QC of lateral changes beyond the neighbour overlay, GeoEast-format export, packaging as an installer.
The stack-power spectrum does not yet reproduce GeoEast numerically (see VALIDATION.md).
