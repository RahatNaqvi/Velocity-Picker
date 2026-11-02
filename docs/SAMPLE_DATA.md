# Sample data needed
1. One 2D land line, CMP-sorted SEG-Y (offset, CDP, scalar headers populated), 100-500 CMPs, with your trusted velocity picks for it.
2. Same line as shot gathers SEG-Y + SPS (S/R/X) so CMP computation can be checked against the CMP-sorted file.
3. Trusted semblance output (image or file) for 3-5 CMPs from your normal package, with the parameters used, to compare my semblance.
4. Example existing velocity-analysis / semblance files from each package you want imported.
5. Example velocity file(s) for each export target + its format spec.
6. A deliberately bad file (wrong scalars, zero offsets, gaps in CMP numbering) if you have one.
Share headers summary (textual header + binary header + a few trace headers) alongside each SEG-Y.
