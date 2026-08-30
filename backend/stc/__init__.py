"""
Syndrome-Trellis Code (STC) experimental module for SecureStegVault v3.

STATUS: Experimental approximation.

A full production STC (Filler, Fridrich et al.) requires a carefully optimized
syndrome-trellis Viterbi implementation with wet-paper coding. This module
provides:

1. A cost-ordered syndrome coding approximation suitable for research
   experimentation and ablation (clearly labelled).
2. Interfaces that a future exact STC encoder/decoder can drop into.

Never claim this is mathematically equivalent to classical STC.
"""

from .encoder import stc_embed_bits, stc_extract_bits
from .cost import pixel_modification_cost

__all__ = ["stc_embed_bits", "stc_extract_bits", "pixel_modification_cost"]
