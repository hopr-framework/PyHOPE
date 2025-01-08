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
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================

def TransformMesh():
    # Local imports ----------------------------------------
    from pyhope.readintools.readintools import GetReal, GetRealArray, GetIntFromStr
    from pyhope.mesh.mesh_vars import mesh
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('TRANSFORM MESH...')
    hopout.sep()

    hopout.routine(' Performing basic transformations')
    # Get scaling factor for mesh
    meshScale = GetReal('meshScale')

    # Get translation vector for mesh
    meshTrans = GetRealArray('meshTrans')

    # Get rotation matrix for mesh
    meshRot   = GetRealArray('meshRot')
    meshRot   = np.array(meshRot).reshape(3, 3)

    # Scale mesh
    if meshScale != 1.0:
        mesh.points *= meshScale

    # Translate mesh
    if not np.array_equal(meshTrans,[0.0, 0.0, 0.0]):
        mesh.points += meshTrans

    # Rotate mesh
    if not np.array_equal(meshRot,[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]):
        mesh.points = mesh.points @ meshRot

    hopout.routine(' Performing advanced transformations')
    meshPostDeform = GetIntFromStr('MeshPostDeform')

    if meshPostDeform != 0:
        # perform actual post-deformation
        hopout.warning('Post-deformation not implemented yet!')

    hopout.sep()

    hopout.info('TRANSFORM MESH DONE!')

    return
