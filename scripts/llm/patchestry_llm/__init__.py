# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in the
# LICENSE file found in the root directory of this source tree.
"""LLM refinement stage for patchir-decomp.

Tier 1 rewrites the Ghidra JSON (names, comments, types) and lets
patchir-decomp lift the result; the tool, not this package, decides what
the C means.  Every network call goes through `providers`; nothing here
links against the decompiler.
"""

__version__ = "0.1.0"
