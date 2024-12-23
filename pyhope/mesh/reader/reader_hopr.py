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
import os
import sys
from string import digits
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import h5py
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def NDOFperElemType(elemType: str, nGeo: int) -> int:
    """ Calculate the number of degrees of freedom for a given element type
    """
    match elemType:
        case 'triangle':
            return round((nGeo+1)*(nGeo+2)/2.)
        case 'quad':
            return round((nGeo+1)**2)
        case 'tetra':
            return round((nGeo+1)*(nGeo+2)*(nGeo+3)/6.)
        case 'pyramid':
            return round((nGeo+1)*(nGeo+2)*(2*nGeo+3)/6.)
        case 'prism':
            return round((nGeo+1)**(nGeo-1)*(nGeo+2)/2.)
        case 'hexahedron':
            return round((nGeo+1)**3)
        case _:
            raise ValueError(f'Unknown element type {elemType}')


def ReadHOPR(fnames: list, mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde, change_basis_3D
    from pyhope.mesh.mesh_common import LINTEN
    from pyhope.mesh.mesh_common import faces, face_to_cgns
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    hopout.sep()

    # Create an empty meshio object
    points   = mesh.points if len(mesh.points.shape)>1 else np.zeros((0, 3), dtype=np.float64)
    cells    = mesh.cells_dict
    cellsets = mesh.cell_sets

    nodeCoords   = mesh.points
    offsetnNodes = nodeCoords.shape[0]
    nSides       = 0

    # Vandermonde for changeBasis
    VdmEqHdf5ToEqMesh = np.array([])

    for fname in fnames:
        # Check if the file is using HDF5 format internally
        if not h5py.is_hdf5(fname):
            hopout.warning('[󰇘]/{} is not in HDF5 format, exiting...'.format(os.path.basename(fname)))
            sys.exit(1)

        with h5py.File(fname, mode='r') as f:
            # Check if file contains the Hopr version
            if 'HoprVersion' not in f.attrs:
                hopout.warning('[󰇘]/{} does not contain the Hopr version, exiting...'.format(os.path.basename(fname)))
                sys.exit(1)

            # Read the globalNodeIDs
            nodeInfo   = np.array(f['GlobalNodeIDs'])

            # Read the nodeCoords
            nodeCoords = np.array(f['NodeCoords'])

            # Read nGeo
            nGeo       = int(cast(int, f.attrs['Ngeo']))

            if nGeo == mesh_vars.nGeo:
                # only retain the unique nodes
                indices    = np.unique(nodeInfo, return_index=True)[1]
                nodeCoords = nodeCoords[indices]
                points     = np.append(points, nodeCoords, axis=0)
            else:
                # ChangeBasis on the non-unique nodes
                # > Currently only supported for hexahedrons
                filename = os.path.basename(fname)
                print(hopout.warn(f'[󰇘]/{filename} has different polynomial order than the current mesh, converting...',
                      length=999))
                print(hopout.warn(f'> NGeo [HDF5] = {nGeo}, NGeo [Mesh] = {mesh_vars.nGeo}') + '\n')

                # Compute the equidistant point set used by HOPR
                xEqHdf5 = np.zeros(nGeo+1)
                for i in range(nGeo+1):
                    xEqHdf5[i] = 2.*float(i)/float(nGeo) - 1.
                wBaryEqHdf5 = barycentric_weights(nGeo, xEqHdf5)

                # Compute the equidistant point set used by meshIO
                xEqMesh = np.zeros(mesh_vars.nGeo+1)
                for i in range(mesh_vars.nGeo+1):
                    xEqMesh[i] = 2.*float(i)/float(mesh_vars.nGeo) - 1.
                # wBaryEqMesh = barycentric_weights(mesh_vars.nGeo, xEqMesh)

                # Compute the Vandermonde matrix
                VdmEqHdf5ToEqMesh = calc_vandermonde(nGeo+1, mesh_vars.nGeo+1, wBaryEqHdf5, xEqHdf5, xEqMesh)

            # Read the elemInfo and sideInfo
            elemInfo   = np.array(f['ElemInfo'])
            sideInfo   = np.array(f['SideInfo'])
            BCNames    = [s.strip().decode('utf-8') for s in cast(h5py.Dataset, f['BCNames'])]

            # Construct the elements, meshio format
            for elem in elemInfo:
                # Correct ElemType if NGeo is changed
                elemNum  = elem[0] % 100
                elemNum += 200 if mesh_vars.nGeo > 1 else 100

                # Obtain the element type
                elemType = ELEMTYPE.inam[elemNum]
                if len(elemType) > 1:
                    elemType  = elemType[0].rstrip(digits)
                    elemType += str(NDOFperElemType(elemType, mesh_vars.nGeo))
                else:
                    elemType  = elemType[0]

                # ChangeBasis currently only supported for hexahedrons
                linMap    = LINTEN(elemNum, order=mesh_vars.nGeo)
                mapLin    = {k: v for v, k in enumerate(linMap)}

                if nGeo == mesh_vars.nGeo:
                    elemIDs   = np.arange(elem[4], elem[5])
                    elemNodes = np.zeros(len(elemIDs), dtype=np.uint64)
                    elemNodes = elemIDs[[mapLin[np.int64(i)] for i in range(len(elemIDs))]]
                    elemNodes = np.expand_dims(nodeInfo[elemNodes]-1+offsetnNodes, axis=0)
                else:
                    elemIDs   = np.arange(points.shape[0], points.shape[0]+(mesh_vars.nGeo+1)**3., dtype=np.uint64)
                    elemNodes = np.zeros(len(elemIDs), dtype=np.uint64)
                    elemNodes = elemIDs[[mapLin[np.int64(i)] for i in range(len(elemIDs))]]
                    # This needs no offset as we already accounted for the number of points in elemIDs
                    elemNodes = np.expand_dims(         elemNodes                , axis=0)

                    # This is still in tensor-product format
                    meshNodes = nodeCoords[np.arange(elem[4], elem[5])].reshape((nGeo+1, nGeo+1, nGeo+1, 3))
                    meshNodes = meshNodes.transpose(3, 0, 1, 2)
                    try:
                        meshNodes = change_basis_3D(VdmEqHdf5ToEqMesh, meshNodes)
                        meshNodes = meshNodes.transpose(1, 2, 3, 0)
                        meshNodes = meshNodes.reshape((int((mesh_vars.nGeo+1)**3.), 3))
                        points    = np.append(points, meshNodes, axis=0)
                    except UnboundLocalError:
                        raise UnboundLocalError('Something went wrong with the change basis')

                try:
                    cells[elemType] = np.append(cells[elemType], elemNodes.astype(np.uint64), axis=0)
                except KeyError:
                    cells[elemType] = elemNodes.astype(np.uint64)

                # Attach the boundary sides
                sCounter = 0
                for index in range(elem[2], elem[3]):
                    # Account for mortar sides
                    # TODO: Add mortar sides

                    # Obtain the side type
                    sideType  = sideInfo[index, 0]
                    sideBC    = sideInfo[index, 4]

                    BCName    = BCNames[sideBC-1]
                    face      = faces(elemType)[sCounter]
                    corners   = [elemNodes[0][s] for s in face_to_cgns(face, elemType)]

                    # Get the number of corners
                    nCorners  = abs(sideType % 10)
                    sideName  = 'quad' if nCorners == 4 else 'tri'
                    if mesh_vars.nGeo > 1:
                        sideName += str(NDOFperElemType(sideName, mesh_vars.nGeo))
                    sideNodes = np.expand_dims(corners, axis=0)

                    try:
                        cells[sideName] = np.append(cells[sideName], sideNodes.astype(np.uint64), axis=0)
                    except KeyError:
                        cells[sideName] = sideNodes.astype(np.uint64)

                    # Increment the side counter
                    sCounter += 1
                    nSides   += 1

                    if sideBC == 0:
                        continue

                    # Add the side to the cellset
                    # > We did not create any 0D/1D objects, so we do not need to consider any offset
                    try:
                        cellsets[BCName][1] = np.append(cellsets[BCName][1], np.array([nSides-1], dtype=np.uint64))
                    except KeyError:
                        # Pyright does not understand that Meshio expects a list with one None entry
                        cellsets[BCName]    = [None, np.array([nSides-1], dtype=np.uint64)]  # type: ignore

            # Update the offset for the next file
            offsetnNodes = points.shape[0]

    mesh   = meshio.Mesh(points    = points,    # noqa: E251
                         cells     = cells,     # noqa: E251
                         cell_sets = cellsets)  # noqa: E251

    return mesh
