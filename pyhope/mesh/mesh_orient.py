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
import sys
import string
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def OrientMesh() -> None:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_common import LINMAP, faces, dir_to_nodes
    # ------------------------------------------------------

    hopout.sep()
    hopout.routine('Eliminating duplicate points')

    # Eliminate duplicate points
    mesh_vars.mesh.points, inverseIndices = np.unique(mesh_vars.mesh.points, axis=0, return_inverse=True)

    # Update the mesh
    for cell in mesh_vars.mesh.cells:
        # Map the old indices to the new ones
        # cell.data = np.vectorize(lambda idx: inverseIndices[idx])(cell.data)
        # Efficiently map all indices in one operation
        cell.data = inverseIndices[cell.data]

    hopout.sep()
    hopout.routine('Checking if surface normal vectors point outwards')

    mesh   = mesh_vars.mesh
    nElems = 0

    for elemType in mesh.cells_dict.keys():
        # Only consider three-dimensional types
        if not any(s in elemType for s in mesh_vars.ELEMTYPE.type.keys()):
            continue

        # Get the elements
        ioelems  = mesh.get_cells_type(elemType)
        baseElem = elemType.rstrip(string.digits)
        nIOElems = ioelems.shape[0]

        if isinstance(baseElem, str):
            baseElem = mesh_vars.ELEMTYPE.name[baseElem]
        mapLin = LINMAP(baseElem, order=mesh_vars.nGeo)

        # Orient the elements
        for iElem in range(nElems, nElems+nIOElems):
            ionodes  = ioelems[iElem]
            nodes    = ionodes[mapLin]
            points   = mesh_vars.mesh.points[nodes]

            # Center of element
            cElem    = points.reshape(-1, points.shape[-1]).sum(axis=0) / np.prod(nodes.shape)

            for face in faces(elemType):
                # Center of face
                fnodes   = dir_to_nodes(face, elemType, nodes)
                fpoints  = mesh_vars.mesh.points[fnodes]
                match face:
                    case 'y-' | 'x+' | 'z+':
                        fpoints = fpoints.transpose(1, 0, 2)
                cFace      = fpoints.reshape(-1, fpoints.shape[-1]).sum(axis=0) / np.prod(fnodes.shape)

                # Tangent and normal vectors
                nVecFace   = cElem - cFace
                nVecFace   = nVecFace / np.linalg.norm(nVecFace)
                vec1       = fpoints[-1, 0, :] - fpoints[0, 0, :]
                vec2       = fpoints[0, -1, :] - fpoints[0, 0, :]
                normal     = np.cross(vec1, vec2) / np.linalg.norm(np.cross(vec1, vec2))

                # Dot product and check if normal points outwards
                dotprod    = np.dot(nVecFace, normal)
                if dotprod < 0:
                    hopout.warning('Surface normals are not pointing outwards, exiting...')
                    print(hopout.warn(f'> Element {iElem+1}, Side {face}'))  # noqa: E501
                    sys.exit(1)

        # Add to nElems
        nElems += nIOElems
