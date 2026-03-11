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
import os
import traceback
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# OpenGL_AVAILABLE = True
# ==================================================================================================================================

# Try enabling OpenGL
try:
    # Make Gmsh available immediately
    import gmsh as _gmsh
    gmsh = _gmsh

except OSError as e:
    # Separately handle the case of missing library
    import pyhope.output.output as hopout

    # Check for specific errors
    if '.so' in str(e):
        print()
        # Capture and print the traceback as a string
        print(hopout.warn(traceback.format_exc(), split=False))
        hopout.error(f'Cannot import gmsh, possibly missing library "{os.path.basename(str(e).split(":")[0])}"')
    else:
        print()
        # Capture and print the traceback as a string
        print(hopout.warn(traceback.format_exc(), split=False))
        hopout.error(f'Cannot import gmsh, encountered error {e}')

except Exception as e:
    import pyhope.output.output as hopout
    print()
    # Capture and print the traceback as a string
    print(hopout.warn(traceback.format_exc(), split=False))
    hopout.error(f'Cannot import gmsh, encountered error "{e}"')
