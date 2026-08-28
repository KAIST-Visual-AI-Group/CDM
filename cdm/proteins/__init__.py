import os
import sys

# DPLM-2's structure tokenizer imports pure-Python helpers from OpenFold, which ships as a
# vendored submodule of DPLM. Installing it with pip would build CUDA kernels we never call
# (and that need nvcc to match the torch CUDA version), so put it on the path directly.
_VENDORED_OPENFOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dplm", "vendor", "openfold")
if os.path.isdir(_VENDORED_OPENFOLD) and _VENDORED_OPENFOLD not in sys.path:
    sys.path.append(_VENDORED_OPENFOLD)
