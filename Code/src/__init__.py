"""Simulation-free conditional paths, dual semicouplings, and weighted matching."""
import os
import sys

# Windows conda MKL and PyTorch can ship different Intel OpenMP DLLs. Use
# sequential MKL for the small CPU linear-algebra operations; torch retains
# its own configured threading. Do not allow duplicate OpenMP runtimes.
if sys.platform == "win32":
    os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
