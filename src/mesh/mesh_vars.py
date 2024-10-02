#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of UVWXYZ
#
# Copyright (c) 2022-2024 Andrea Beck
#
# UVWXYZ is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# UVWXYZ is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# UVWXYZ. If not, see <http://www.gnu.org/licenses/>.

# ==================================================================================================================================
# Mesh generation library
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================
mode   : int                                      # Mesh generation mode (1 - Internal, 2 - External (MeshIO))
mesh   : meshio._mesh.Mesh                        # MeshIO object holding the mesh
nGeo   : int                                      # Order of spline-reconstruction for curved surfaces
sortIJK: bool                                     # Flag if mesh should be I,J,K sorted

bcs    : list                                     # [list of dict] - Boundary conditions
vvs    : list                                     # [list of dict] - Periodic vectors

elems  : list                                     # [list of list] - Element nodes
sides  : list                                     # [list of list] - Side    nodes

HEXMAP : np.ndarray                               # CGNS <-> IJK ordering for high-order hexahedrons
already_curved: bool                              # Flag if mesh is already curved


class CGNS:
    regenerate_BCs: bool                          # Flag if CGNS needs BC regeneration


class ELEM:
    type = {'tetra': 4, 'pyramid': 5, 'wedge': 5, 'hexahedron': 6}


def ELEMMAP(meshioType: str) -> int:
    # Local imports ----------------------------------------
    import sys
    # ------------------------------------------------------
    # Linear or curved tetrahedron
    if 'tetra' in meshioType:
        if 'tetra' == meshioType:
            return 104
        else:
            return 204
    # Linear or curved pyramid
    elif 'pyramid' in meshioType:
        if 'pyramid' == meshioType:
            return 105
        else:
            return 205
    # Linear or curved wedge / prism
    elif 'wedge' in meshioType:
        if 'wedge' == meshioType:
            return 106
        else:
            return 206
    # Linear or curved hexahedron
    elif 'hexahedron' in meshioType:
        if 'hexahedron' == meshioType:
            return 108
        else:
            return 208
    else:
        print('Unknown element type {}'.format(meshioType))
        sys.exit(1)
