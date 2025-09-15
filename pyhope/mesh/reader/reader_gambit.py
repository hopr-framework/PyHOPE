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
import gc
from collections import defaultdict
from functools import cache
from typing import Any, Union, cast
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# Instantiate ELEMTYPE
elemTypeClass = mesh_vars.ELEMTYPE()
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


@cache
def gambit_faces(elemType: Union[int, str]) -> list[str]:
    """ Return a list of all sides of an element
    """
    faces_map = {  # Tetrahedron
                   # Pyramid
                   # Wedge / Prism
                   # Hexahedron
                   8: ['y-', 'x+', 'y+', 'x-', 'z-', 'z+']
                }

    if isinstance(elemType, str):
        elemType = elemTypeClass.name[elemType]

    if elemType % 100 not in faces_map:
        raise ValueError(f'Error in faces: elemType {elemType} is not supported')

    return faces_map[elemType % 100]


def ReadGambit(fnames: list, mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # import pyhope.mesh.mesh_vars as mesh_vars  # Already imported at the top
    from pyhope.common.common import lines_that_contain
    from pyhope.mesh.mesh_common import face_to_cgns
    from pyhope.mesh.mesh_common import FaceOrdering
    from pyhope.meshio.meshio_ordering import NodeOrdering
    # ------------------------------------------------------

    hopout.sep()

    # Create an empty meshio object
    points   = mesh.points if len(mesh.points.shape)>1 else np.zeros((0, 3), dtype=np.float64)
    pointl   = cast(list, points.tolist())
    cells    = mesh.cells_dict
    cellsets = defaultdict(lambda: defaultdict(list))

    nodeCoords   = mesh.points
    nSides       = np.zeros(2, dtype=int)
    elemTypes    = []

    # Initialize the node ordering
    node_ordering = NodeOrdering()

    for fnum, fname in enumerate(fnames):
        # Check if the file is using ASCII format internally
        with open(fname, 'r') as f:
            # Check if the file is in ASCII format
            try:
                # Read the file content
                content   = f.readlines()
                useBinary = not any('CONTROL INFO' in line for line in content)
            except UnicodeDecodeError:
                raise ValueError('Gambit binary files are not implemented yet')

            if not useBinary:
                # Search for the line containing the number of elements
                elemLine = lines_that_contain('NUMNP'            , content)[0] + 1
                # Read and unpack the number of elements
                npoints, nelems, _, nbcs, nDim, _ = map(int, content[elemLine].strip().split())

                # PyHOPE currently only supports 3D meshes
                if nDim != 3:
                    raise ValueError(f'Unsupported mesh dimension {nDim}. Only 3D meshes are supported.')

                # Check if the number of boundary conditions match the parameter file
                if nbcs > len(mesh_vars.bcs):
                    hopout.error(f'Number of boundary conditions in the mesh ({nbcs}) ' +
                                 f'does not match the parameter file ({len(mesh_vars.bcs)})')

                # Search for the line starting the node coordinates
                nodeLine = lines_that_contain('NODAL COORDINATES', content)[0]

                # Iterate and unpack the node coordinates
                nodeCoords = content[nodeLine+1:nodeLine+npoints+1]
                nodeCoords = np.genfromtxt(nodeCoords, dtype=np.float64, delimiter=None, usecols=(1, 2, 3))
                pointl.extend(nodeCoords)

                # Search for the line starting the element connectivity
                elemLine = lines_that_contain('ELEMENTS/CELLS', content)[0] + 1
                elemIter = iter(content[elemLine:])

                # Iterate and unpack the element connectivity
                for line in elemIter:
                    if 'ENDOFSECTION' in line:
                        break

                    tokens = line.strip().split()
                    if not tokens:
                        continue

                    try:
                        elemID, gType, nNodes, *elemNodes = tokens
                        elemID, gType, nNodes             = int(elemID), int(gType), int(nNodes)
                    except ValueError:
                        continue

                    # Map gambit element type to meshio element type
                    elemType  = node_ordering.typing_gambit_to_meshio(gType)
                    elemTypes.append(elemType)

                    # Check if the number of nodes matched the expected number
                    if nNodes != NDOFperElemType(elemType, mesh_vars.nGeo):
                        hopout.error(f'Number of element nodes ({nNodes}) does not match expectation for NGeo={mesh_vars.nGeo}')

                    # Keep extending the element connectivity until elemNodes is reached
                    while len(elemNodes) < nNodes:
                        elemNodes.extend(next(elemIter).strip().split())

                    # Convert elemNodes to a numpy array of integers
                    elemNodes = np.array(elemNodes, dtype=np.uint64)
                    elemNodes = node_ordering.ordering_gambit_to_meshio(elemType, elemNodes) - 1

                    cells.setdefault(elemType, []).append(elemNodes.astype(np.uint64))

                # Clean-up for memory safety
                del elemLine
                del elemIter

                # Check if the number of elements match the header
                if nelems != sum(len(cells[key]) for key in cells):
                    raise ValueError('Failed to obtain the correct number of elements.')

                # Search for the line starting the element groups
                grsLine  = lines_that_contain('ELEMENT GROUP', content)[0] + 1
                grsIter  = iter(content[grsLine:])
                lnum     = 0

                # Iterate and unpack the boundary conditions
                for line in grsIter:
                    lnum += 1
                    # Iterate until the number of boundary conditions is reached
                    if 'ENDOFSECTION' in line:
                        # Check if the next sections is also an element group
                        if 'ELEMENT GROUP' not in content[grsLine+lnum]:
                            break

                    tokens = line.strip().split()
                    if not tokens:
                        continue

                    try:
                        _, zoneNum, _, zoneNElems, _, zoneMat, _, zoneFlags = tokens
                        zoneNum, zoneNElems, zoneMat, zoneFlags = int(zoneNum), int(zoneNElems), int(zoneMat), int(zoneFlags)
                    except ValueError:
                        continue

                    # When merging grids with zoneID = 1, we want them to have separate IDs after the merge
                    # FIXME: GAMBIT allows own volume zone, we should read them here
                    zoneName: str = str(max(fnum+1, zoneNum))

                    # Keep extending the element connectivity until elemNodes is reached
                    _     = next(grsIter)
                    lnum += 1
                    zoneElems = []
                    while len(zoneElems) < zoneNElems:
                        lnum += 1
                        zoneElems.extend(next(grsIter).strip().split())

                    # Convert elemNodes to a numpy array of integers
                    zoneElems = np.array(zoneElems, dtype=np.uint64)

                    # Add the elem to the cellset
                    # > CS1: We create a dictionary of the zones and types that we want
                    for zoneElem in zoneElems:
                        cellsets[zoneName][elemTypes[zoneElem-1]].append(zoneElem-1)

                # Clean-up for memory safety
                del grsLine
                del grsIter

                # Search for the line starting the boundary conditions
                bcsLine  = lines_that_contain('BOUNDARY CONDITIONS', content)[0] + 1
                bcsIter  = iter(content[bcsLine:])
                lnum     = 0

                # Iterate and unpack the boundary conditions
                for line in bcsIter:
                    # Iterate until the number of boundary conditions is reached
                    lnum += 1
                    if 'ENDOFSECTION' in line:
                        # Check if the next sections is also a boundary condition
                        if bcsLine+lnum >= len(content) or 'BOUNDARY CONDITIONS' not in content[bcsLine+lnum]:
                            break

                    tokens = line.strip().split()
                    if not tokens:
                        continue

                    try:
                        bcName, bcType, bcnData, bcnVal, _ = tokens
                        bcName, bcType, bcnData, bcnVal    = bcName, int(bcType), int(bcnData), int(bcnVal)
                    except ValueError:
                        continue

                    # Ignore invalid (empty) boundary conditions
                    if bcnData <= 0:
                        continue

                    BCName = bcName.strip().lower()

                    match bcType:
                        # Nodal Data (ITYPE=0)
                        case 0:
                            raise NotImplementedError('Nodal data boundary conditions are not supported yet.')
                        # Element/Cell Data (ITYPE=1)
                        case 1:
                            # Read the next bcnData lines
                            bcnNodes = []
                            for _ in range(bcnData):
                                bcnNodes.extend(next(bcsIter).strip().split())
                                lnum += 1
                            # bcnNodes is in format [ELEM, ELEMTYPE, FACE, (VALUES)]
                            bcnNodes = np.array(bcnNodes, dtype=int).reshape(bcnData, -1)

                            # Attach the boundary sides
                            for elemID, gType, faceID in bcnNodes[:, :3]:
                                # Map gambit element type to meshio element type
                                elemType  = node_ordering.typing_gambit_to_meshio(gType)

                                # Get the face
                                elem      = cells[elemType][elemID-1]
                                face      = gambit_faces(elemType)[faceID-1]

                                # Determine the side name and number
                                nCorners  = len(face_to_cgns(face, elemType))
                                sideNum   = 0      if nCorners == 4 else 1           # noqa: E272
                                sideBase  = 'quad' if nCorners == 4 else 'triangle'  # noqa: E272
                                sideHO    = '' if mesh_vars.nGeo == 1 else str(NDOFperElemType(sideBase, mesh_vars.nGeo))
                                sideName  = sideBase + sideHO

                                # Map the face ordering from tensor-product to meshio
                                order     = FaceOrdering(sideBase, order=1)
                                corners   = elem[face_to_cgns(face, elemType)]
                                corners   = corners.flatten()[order]
                                sideNodes = np.expand_dims(corners, axis=0)

                                # Add the side to the cells
                                cells.setdefault(sideName, []).append(sideNodes.astype(np.uint64))

                                # Increment the side counter
                                nSides[sideNum] += 1

                                # Add the side to the cellset
                                # > CS1: We create a dictionary of the BC sides and types that we want
                                cellsets.setdefault(BCName, defaultdict()).setdefault(sideName, []).append(nSides[sideNum] - 1)

                # Clean-up for memory safety
                del bcsLine
                del bcsIter

    # After processing all elements, convert each list of arrays to one array
    # > Convert the list of cells to numpy arrays
    cells: dict = {cell_type: np.concatenate([a.reshape(1, -1) if a.ndim == 1 else a for a                      in cell_arrays])  # noqa: E272
                                                                                     for cell_type, cell_arrays in cells.items()}

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    # > CS2: We build the cell sets depending on the cells
    cell_sets:  dict[str, list] = mesh.cell_sets
    cell_types: list[Any      ] = list(cells.keys())
    nCellTypes: int             = len(cell_types)
    cell_tidx:  dict[Any, int ] = {ctype: idx for idx, ctype in enumerate(cell_types)}

    # Convert the dict of cellsets to numpy arrays
    for bc, bc_dict in cellsets.items():
        # Initialize entry for this BC if not exists
        if bc not in cell_sets:
            # Assign the entry to the cell set
            cell_sets[bc] = [None] * nCellTypes

        entry = cell_sets[bc]

        # Process all cell types for this BC
        for side, indices in bc_dict.items():
            BCIndices = np.fromiter(indices, dtype=np.uint64, count=len(indices))

            # Get cell type index
            type_idx = cell_tidx[side]

            # Find matching cell type and populate the corresponding entry
            if entry[type_idx] is not None:
                entry[type_idx] = np.concatenate([entry[type_idx], BCIndices])
            else:
                entry[type_idx] = BCIndices

    # > CS3: We create the final meshio.Mesh object with cell_sets
    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = cells,      # noqa: E251
                         cell_sets = cell_sets)  # noqa: E251

    # Run garbage collector to release memory
    gc.collect()

    return mesh
