"""Velocity-analysis engine (pure numpy/scipy, no GUI imports).

Spectra: Neidell-Taner semblance and a GeoEast-style two-step stack-power spectrum.
Conditioning (bandpass, equalisation, top mute) works on copies. Picked quantity is stacking (NMO)
velocity; RMS/interval conversion is not implemented.
"""
from .conditioning import equalize, ormsby_bandpass, top_mute
from .cvs import constant_velocity_stacks
from .nmo import nmo_correct
from .params import (AnalysisParams, CostEstimate, ParamIssue, ParamValidationError,
                     estimate_cost, validate_params)
from .semblance import AnalysisCancelled, Semblance, compute_semblance
from .stackpower import (StackPowerParams, StackPowerSpectrum, compute_stack_power,
                         select_neighbour_cmps, select_spectrum_cmps)

__all__ = ["AnalysisCancelled", "AnalysisParams", "CostEstimate", "ParamIssue", "ParamValidationError",
           "Semblance", "StackPowerParams", "StackPowerSpectrum", "compute_semblance",
           "compute_stack_power", "constant_velocity_stacks", "equalize", "estimate_cost", "nmo_correct", "ormsby_bandpass",
           "select_neighbour_cmps", "select_spectrum_cmps", "top_mute", "validate_params"]
