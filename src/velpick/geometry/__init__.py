"""Header interpretation, coordinate scalars, trace index and validation. CMP binning from geometry: not implemented."""
from .trace_index import (IndexStaleError, Issue, TraceIndex, ValidationReport, apply_scalar)

__all__ = ["IndexStaleError", "Issue", "TraceIndex", "ValidationReport", "apply_scalar"]
