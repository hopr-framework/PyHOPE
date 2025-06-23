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
import copy
import gc
import itertools
import os
import shutil
# from dataclasses import dataclass, field
from functools import cache
from string import digits
from typing import cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import h5py
import meshio
import numpy as np
from alive_progress import alive_bar
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


@cache
def NDOFperElemType(elemType: str, nGeo: int) -> int:
    """ Calculate the number of degrees of freedom for a given element type
    """
    match elemType:
        case _ if elemType.startswith('triangle'):
            return round((nGeo+1)*(nGeo+2)/2.)
        case _ if elemType.startswith('quad'):
            return round((nGeo+1)**2)
        case _ if elemType.startswith('tetra'):
            return round((nGeo+1)*(nGeo+2)*(nGeo+3)/6.)
        case _ if elemType.startswith('pyramid'):
            return round((nGeo+1)*(nGeo+2)*(2*nGeo+3)/6.)
        case _ if elemType.startswith('wedge'):
            return round((nGeo+1)**2 *(nGeo+2)/2.)
        case _ if elemType.startswith('hexahedron'):
            return round((nGeo+1)**3)
        case _:
            raise ValueError(f'Unknown element type {elemType}')


def ReadHOPR(fnames: list, mesh: meshio.Mesh) -> meshio.Mesh:
    # Standard libraries -----------------------------------
    import tempfile
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde, change_basis_3D
    from pyhope.mesh.mesh_common import LINTEN, FaceOrdering
    from pyhope.mesh.mesh_common import faces, face_to_cgns
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    hopout.sep()

    # Create an empty meshio object
    points   = mesh.points if len(mesh.points.shape)>1 else np.zeros((0, 3), dtype=np.float64)
    pointl   = cast(list, points.tolist())
    cells    = mesh.cells_dict
    cellsets = {}

    nodeCoords   = mesh.points
    offsetnNodes = nodeCoords.shape[0]
    nSides       = np.zeros(2, dtype=int)

    # Vandermonde for changeBasis
    VdmEqHdf5ToEqMesh = np.array(())
    mortarTypeToSkip  = {1: 4, 2: 2, 3: 2}

    # Instantiate ELEMTYPE
    elemTypeClass = ELEMTYPE()

    for fname in fnames:
        # Check if the file is using HDF5 format internally
        if not h5py.is_hdf5(fname):
            hopout.error('[󰇘]/{} is not in HDF5 format, exiting...'.format(os.path.basename(fname)))

        # Create a temporary directory and keep it existing until manually cleaned
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tname = tfile.name
        # Alternatively, load the file directly into tmpfs for faster access
        shutil.copyfile(fname, tname)

        with h5py.File(tname, mode='r') as f:
            # Check if file contains the Hopr version
            if 'HoprVersion' not in f.attrs:
                hopout.error('[󰇘]/{} does not contain the Hopr version, exiting...'.format(os.path.basename(fname)))

            # Read the globalNodeIDs
            nodeInfo   = np.array(f['GlobalNodeIDs'])

            # Read the nodeCoords
            nodeCoords = np.array(f['NodeCoords'])

            # Read nGeo
            nGeo       = int(cast(int, f.attrs['Ngeo']))

            # Try reading in periodic vector if it is not provided in file
            if len(mesh_vars.vvs) == 0:
                try:
                    vvs = np.array(f['VV'])
                    mesh_vars.vvs = [dict() for _ in range(vvs.shape[0])]
                    for iVV, _ in enumerate(mesh_vars.vvs):
                        mesh_vars.vvs[iVV] = dict()
                        mesh_vars.vvs[iVV]['Dir'] = vvs[iVV]
                    # Output vectors
                    print(hopout.warn('Periodicity vectors not defined in parameter file. '
                                      'Reading the vectors from given PyHOPE mesh!'))
                    hopout.sep()
                    hopout.routine('The following vectors were found:')
                    for iVV, vv in enumerate(mesh_vars.vvs):
                        hopout.printoption('vv[{}]'.format(iVV+1), '{0:}'.format(np.round(vv['Dir'], 6)), 'READ IN')
                    hopout.sep()
                # old hopr files might not contain the VV
                except KeyError:
                    pass

            if nGeo == mesh_vars.nGeo:
                # only retain the unique nodes
                indices    = np.unique(nodeInfo, return_index=True)[1]
                nodeCoords = nodeCoords[indices]
                # points     = np.append(points, nodeCoords, axis=0)
                # IMPORTANT: We need to extend the list of points, not append to it
                pointl.extend(nodeCoords.tolist())
            else:
                # ChangeBasis on the non-unique nodes
                # > Currently only supported for hexahedrons
                filename = os.path.basename(fname)
                print(hopout.warn(f'[󰇘]/{filename} has different polynomial order than the current mesh, converting...',
                      length=999))
                print(hopout.warn(f'> NGeo [HDF5] = {nGeo}, NGeo [Mesh] = {mesh_vars.nGeo}') + '\n')

                # Compute the equidistant point set used by HOPR
                xEqHdf5     = np.linspace(-1, 1, num=nGeo+1, dtype=np.float64)
                wBaryEqHdf5 = barycentric_weights(nGeo, xEqHdf5)

                # Compute the equidistant point set used by meshIO
                xEqMesh     = np.linspace(-1, 1, num=mesh_vars.nGeo+1, dtype=np.float64)
                # wBaryEqMesh = barycentric_weights(mesh_vars.nGeo, xEqMesh)

                # Compute the Vandermonde matrix
                VdmEqHdf5ToEqMesh = calc_vandermonde(nGeo+1, mesh_vars.nGeo+1, wBaryEqHdf5, xEqHdf5, xEqMesh)

            # Read the elemInfo and sideInfo
            elemInfo   = np.array(f['ElemInfo'])
            sideInfo   = np.array(f['SideInfo'])
            BCNames    = [s.strip().decode('utf-8') for s in cast(h5py.Dataset, f['BCNames'])]

            # Cache the mapping here, so we consider the mesh order
            linCache   = {}

            with alive_bar(len(elemInfo), title='│             Processing Elements', length=33) as bar:
                # Construct the elements, meshio format
                for elem in elemInfo:
                    # Correct ElemType if NGeo is changed
                    elemNum  = elem[0] % 100
                    elemNum += 200 if mesh_vars.nGeo > 1 else 100

                    # Obtain the element type
                    elemType = elemTypeClass.inam[elemNum]
                    if len(elemType) > 1:
                        elemType  = elemType[0].rstrip(digits)
                        elemType += str(NDOFperElemType(elemType, mesh_vars.nGeo))
                    else:
                        elemType  = elemType[0]

                    # ChangeBasis currently only supported for hexahedrons
                    if elemNum in linCache:
                        mapLin = linCache[elemNum]
                    else:
                        _, mapLin = LINTEN(elemNum, order=mesh_vars.nGeo)
                        mapLin    = np.array(tuple(mapLin[np.int64(i)] for i in range(len(mapLin))))
                        linCache[elemNum] = mapLin

                    if nGeo == mesh_vars.nGeo:
                        elemIDs   = np.arange(elem[4], elem[5])
                        elemNodes = elemIDs[mapLin[:len(elemIDs)]]
                        elemNodes = np.expand_dims(nodeInfo[elemNodes] - 1 + offsetnNodes, axis=0)
                    else:
                        nElemNode = (mesh_vars.nGeo+1)**3
                        # elemIDs   = np.arange(points.shape[0], points.shape[0]+nElemNode, dtype=np.uint64)
                        elemIDs   = np.arange(len(pointl), len(pointl)+nElemNode, dtype=np.uint64)
                        elemNodes = elemIDs[mapLin[:nElemNode]]
                        # This needs no offset as we already accounted for the number of points in elemIDs
                        elemNodes = np.expand_dims(         elemNodes                    , axis=0)

                        # This is still in tensor-product format
                        meshNodes = nodeCoords[np.arange(elem[4], elem[5])].reshape((nGeo+1, nGeo+1, nGeo+1, 3))
                        meshNodes = meshNodes.transpose(3, 0, 1, 2)
                        try:
                            meshNodes = change_basis_3D(VdmEqHdf5ToEqMesh, meshNodes)
                            meshNodes = meshNodes.transpose(1, 2, 3, 0)
                            meshNodes = meshNodes.reshape((int((mesh_vars.nGeo+1)**3.), 3))
                            # points    = np.append(points, meshNodes, axis=0)
                            # IMPORTANT: We need to extend the list of points, not append to it
                            pointl.extend(meshNodes.tolist())
                        except UnboundLocalError:
                            raise UnboundLocalError('Something went wrong with the change basis')

                    cells.setdefault(elemType, []).append(elemNodes.astype(np.uint64))

                    # Attach the boundary sides
                    sCounter = 0
                    sideRange = iter(range(elem[2], elem[3]))  # Create an iterator for the loop
                    for index in sideRange:
                        # Obtain the side type
                        sideType  = sideInfo[index, 0]
                        nbElemID  = sideInfo[index, 2]
                        sideBC    = sideInfo[index, 4]

                        BCName    = BCNames[sideBC-1].lower()
                        face      = faces(elemType)[sCounter]

                        # Get the number of corners
                        nCorners  = abs(sideType) % 10

                        # Assmble the side information
                        sideNum   = 0      if nCorners == 4 else 1           # noqa: E272
                        sideBase  = 'quad' if nCorners == 4 else 'triangle'  # noqa: E272
                        sideHO    = '' if mesh_vars.nGeo == 1 else str(NDOFperElemType(sideBase, mesh_vars.nGeo))
                        sideName  = sideBase + sideHO

                        # Map the face ordering from tensor-product to meshio
                        order     = FaceOrdering(sideBase, order=1)
                        corners   = elemNodes[0][face_to_cgns(face, elemType)]
                        corners   = corners.flatten()[order]
                        sideNodes = np.expand_dims(corners, axis=0)

                        # Add the side to the cells
                        cells.setdefault(sideName, []).append(sideNodes.astype(np.uint64))

                        # Increment the side counter
                        sCounter        += 1
                        nSides[sideNum] += 1

                        # Account for mortar sides
                        if nbElemID < 0:
                            # Side is a big mortar side
                            # > Skip the nVirtualSides small virtual sides
                            nVirtualSides = mortarTypeToSkip[abs(nbElemID)]
                            list(itertools.islice(sideRange, nVirtualSides))

                        if sideBC == 0:
                            continue

                        # Add the side to the cellset
                        # > CS1: We create a dictionary of the BC sides and types that we want
                        cellsets.setdefault(BCName, {}).setdefault(sideName, []).append(nSides[sideNum] - 1)

                    # Update progress bar
                    bar()

                # Update the offset for the next file
                # offsetnNodes = points.shape[0]
                offsetnNodes = len(pointl)

        # Cleanup temporary file
        if tfile is not None:
            os.unlink(tfile.name)

        hopout.sep()

    # After processing all elements, convert each list of arrays to one array
    # > Convert the list of cells to numpy arrays
    for cell_type in cells:
        cells[cell_type] = np.concatenate([a if a.ndim == 2 else a.reshape(1, -1) for a in cells[cell_type]], axis=0)

    # Convert the list of cellsets to numpy arrays
    for bc in cellsets:
        for side in cellsets[bc]:
            cellsets[bc][side] = np.array(cellsets[bc][side], dtype=np.uint64)

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    # > CS2: We create a meshio.Mesh object without cell_sets
    mesh   = meshio.Mesh(points    = points,    # noqa: E251
                         cells     = cells)     # noqa: E251

    # > CS3: We build the cell sets depending on the cells
    cell_sets  = mesh.cell_sets
    cell_types = [s for s in mesh.cells_dict.keys()]
    cell_list  = [None for _ in cell_types]

    for key, val in cellsets.items():
        for v_key, v_val in val.items():
            if key in cell_sets.keys():
                entry = cell_sets[key]
            else:
                entry = copy.copy(cell_list)

            # Find matching cell type and populate the corresponding entry
            if entry[cell_types.index(v_key)] is not None:
                entry[cell_types.index(v_key)] = np.append(cast(np.ndarray, entry[cell_types.index(v_key)]), v_val)  # type: ignore
            else:
                entry[cell_types.index(v_key)] = v_val

            # Assign the entry to the cell set
            cell_sets[key] = entry  # type: ignore

    # > CS4: We create the final meshio.Mesh object with cell_sets
    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = cells,      # noqa: E251
                         cell_sets = cell_sets)  # noqa: E251

    # Run garbage collector to release memory
    gc.collect()

    return mesh
