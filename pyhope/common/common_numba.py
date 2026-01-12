#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2024 Numerics Research Group, University of Stuttgart, Prof. Andrea Beck
#
# PyHOPE is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# PyHOPE is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# PyHOPE. If not, see <http://www.gnu.org/licenses/>.

# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
from typing import Any
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
NUMBA_AVAILABLE: bool = False
# ==================================================================================================================================


# Optional numba support
def jit(*args: Any, **kwargs: Any):
    return lambda f: f


def njit(*args: Any, **kwargs: Any):
    return lambda f: f


# Runtime fallback "types" object
class _NumbaTypesStub:
    """ Stub class that always returns itself
    """
    def __getattr__(self, name: str):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        return self


types  = _NumbaTypesStub()
prange = range


# # Try enabling numba immediately
try:
    import numba as nb  # ty:ignore[unresolved-import, unused-ignore-comment]

    jit    = nb.jit
    njit   = nb.njit
    prange = nb.prange
    types  = nb.types
    NUMBA_AVAILABLE = True
except Exception:
    pass


def PkgsCheckNumba() -> None:
    """ Check optional numba support
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------
    if NUMBA_AVAILABLE:
        hopout.routine('Initializing numba (JIT compilation)...')
