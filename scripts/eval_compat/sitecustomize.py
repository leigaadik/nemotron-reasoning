"""Process-local compatibility guards for the evaluation runtime.

The platform image contains optional flash_attn and fbgemm_gpu extensions that
are ABI-incompatible with its Torch build. vLLM has built-in fallbacks when
these optional packages are absent, but importing the installed copies fails
before fallback detection can run. The dual-evaluation launcher opts into this
guard through NEMOTRON_EVAL_DISABLE_BROKEN_EXTENSIONS=1.
"""

from __future__ import annotations

import os
import sys


if os.environ.get("NEMOTRON_EVAL_DISABLE_BROKEN_EXTENSIONS") == "1":
    sys.modules["flash_attn"] = None
    sys.modules["fbgemm_gpu"] = None
