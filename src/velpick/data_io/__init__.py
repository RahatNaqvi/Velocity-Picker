"""SEG-Y reader (read-only, memmap) and, later, importers for existing velocity-analysis products."""
from .segy import (DEFAULT_TRACE_HEADER_MAP, HeaderField, SegyError, SegyFile,
                   SegyFormatError, ibm_to_ieee)

__all__ = ["DEFAULT_TRACE_HEADER_MAP", "HeaderField", "SegyError", "SegyFile",
           "SegyFormatError", "ibm_to_ieee"]
