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
from string import digits
from typing import cast
from typing import Final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import meshio
import numpy as np
import numpy.typing as npt
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


def MeshExtrude(mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.basis.basis_basis import barycentric_weights, legendre_gauss_nodes
    from pyhope.basis.basis_basis import calc_vandermonde, polynomial_derivative_matrix
    from pyhope.basis.basis_orient import check_orientation
    from pyhope.common.common_progress import ProgressBar
    from pyhope.common.common_template import LoadTemplate
    from pyhope.common.common_tools import temporary_assign
    from pyhope.io.io_gmsh import GMSHCELLTYPES
    from pyhope.mesh.mesh_common import NDOFperElemType
    from pyhope.mesh.mesh_vars import ELEM, nGeo
    from pyhope.mesh.topology.mesh_topology import appendBCSet
    from pyhope.readintools.readintools import GetInt, GetStr
    # ------------------------------------------------------

    # Instantiate the Gmsh cell type mapping
    gmshCellTypes = GMSHCELLTYPES()

    # Check if the mesh is already 3D
    if tuple(cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes3D):
        return mesh

    # Check if the mesh contains 2D elements to extrude
    if       tuple(cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes2D) and not mesh_vars.doExtrude:  # noqa: E271  # ruff: ignore[line-too-long]
        hopout.error('Mesh contains suitable surface cells for extrusion but MeshExtrude=F, exiting...')
    elif not tuple(cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes2D):
        hopout.error('Mesh contains no suitable surface cells for extrusion, exiting...')

    hopout.info('Extruding surface to volume mesh')

    # Read in the mesh post-deformation flag
    hopout.sep()
    extrTemplate   = GetStr('MeshExtrudeTemplate')
    extrBCIndexTop = GetInt('MeshExtrudeBCIndexTop') - 1
    extrBCIndexBot = GetInt('MeshExtrudeBCIndexBot') - 1

    # Continue with extrusion
    hopout.sep()
    hopout.routine(f'  Template: {extrTemplate}')

    # Setup the extrusion
    extrShifts = LoadTemplate(extrTemplate.strip().lower(), __file__, 'Extrusion').ExtrudeTemplate()

    # Copy original points
    points    = mesh.points
    pointl    = cast(list, points.tolist())
    elems_old = mesh.cells.copy()
    cell_sets = getattr(mesh, 'cell_sets', {})

    # Get base key to distinguish between linear and high-order elements
    ho_key    = 100 if nGeo == 1 else 200
    nPoints   = len(pointl)

    # Expected number of nodes
    faceNum   = [ int((nGeo+1)*(nGeo+2)/2), int((nGeo+1)**2) ]
    faceType  = [f'triangle{"" if nGeo == 1 else faceNum[0]}', f'quad{"" if nGeo == 1 else faceNum[1]}']

    # For zones, we need to append the expected extruded elements
    for cblock in cell_sets.values():
        # Each set_blocks is a list of arrays, one entry per cell block
        for blockID in range(len(cblock)):
            etype = elems_old[blockID].type
            if etype[:4] not in ('tria', 'quad'):
                continue

            elemNum = ho_key + (8 if etype.startswith('quad') else 6)
            # Obtain the element type
            elemType = elemTypeClass.inam[elemNum]
            if len(elemType) > 1:
                elemType  = elemType[0].rstrip(digits)
                elemDOFs  = NDOFperElemType(elemType, mesh_vars.nGeo)
                elemType += str(elemDOFs)
            else:
                elemType  = elemType[0]
                elemDOFs  = NDOFperElemType(elemType, mesh_vars.nGeo)

            if elemType not in faceType:
                faceType.append(elemType)
                faceNum .append(elemDOFs)

    # nFaces contains the existing number of [Triangle, Quad, Wedge, Hexahedron]
    nFaces    = np.zeros(len(faceType), dtype=int)

    # Prepare new cell blocks and new cell_sets
    elems_lst = {ftype: [] for ftype in faceType}
    csets_lst = {}

    # Set up the element extrusion function
    elemExtruder = {'tria'     : (extrude_pris, pris_faces ),
                    'quad'     : (extrude_hexa, hexa_faces )}
    faceMaper    = { ho_key + 6: lambda x: 0 if x == 0 else 1,
                     ho_key + 8: lambda x: 1}
    # Expected number of nodes for a triangle face
    # INFO: We reduce the new faces to first-order. Yes, this breaks direkt meshio output. But we are not using this anyways.
    #       If you want to use mesh.write() for debug purposes, comment out the BC face creation.
    # nFace = (nGeo+1)*(nGeo+2)/2
    nFace: Final[int] = 3

    # Create the element sets
    meshcells = tuple((k, v) for k, v in mesh.cell_sets_dict.items() if any(key.startswith('tria') for key in v)
                                                                     or any(key.startswith('quad') for key in v))

    # Take the correct BC index for the bottom BC
    meshcells = tuple(s for s in meshcells if s[0].lower() == mesh_vars.bcs[extrBCIndexBot].name)

    match len(meshcells):
        case 0:
            hopout.error('Could not find boundary condition for extrusion, exiting...')
        case 1:
            pass
        case _:
            hopout.error('Found more than one boundary condition for extrusion, exiting...')

    # Convert the (1D, 2D) boundary cell set into a dictionary
    csets_old = {}
    zsets_old = {}
    for cname, cblock in cell_sets.items():
        # Each set_blocks is a list of arrays, one entry per cell block
        for blockID, block in enumerate(cblock):
            etype = elems_old[blockID].type
            if etype[:4] not in ('line', 'tria', 'quad'):
                continue

            # Ignore the empty zones
            if block is None:
                continue

            # Determine how many corner nodes to keep
            nCorners = 2 if 'line' in etype else (3 if 'tria' in etype else 4)

            # Filter the zone 2D faces
            if nCorners > 2 and cname.lower() != mesh_vars.bcs[extrBCIndexBot].name:
                # Sort them as a set for membership checks
                for face in block:
                    # Slice to only include corners for the search dictionary
                    nodes = mesh.cells[blockID].data[face][:nCorners]
                    zsets_old.setdefault(frozenset(nodes), []).append(cname)

                # Do not add them to the boundary conditions
                continue

            # Sort them as a set for membership checks
            for face in block:
                # Slice to only include corners for the search dictionary
                nodes = mesh.cells[blockID].data[face][:nCorners]
                csets_old.setdefault(frozenset(nodes), []).append(cname)

    nTotalElems = sum(cdata.shape[0] for _, zdata in meshcells for cdata in cast(dict, zdata).values())
    bar = ProgressBar(value=nTotalElems, title='│             Processing Elements', length=33, threshold=1000)

    # Build an inverted index to map each node to all face keys (from csets_old) that contain it
    nodeToFace = defaultdict(set)
    for subFace in csets_old:
        for node in subFace:
            nodeToFace[node].add(subFace)

    # Build an inverted index to map each node to all zone keys (from zsets_old) that contain it
    nodeToZone = defaultdict(set)
    for subFace in zsets_old:
        for node in subFace:
            nodeToZone[node].add(subFace)

    # We need to unwrap meshcells for each zone, i.e. each 2D boundary condition
    for meshcell in meshcells:
        _    , mdict = meshcell

        # Iterate over all cell types in this BC
        for iType in range(len(mdict)):
            mtype, mcell = list(cast(dict, mdict).keys())[iType], list(cast(dict, mdict).values())[iType]
            cdata = mesh.get_cells_type(mtype)[mcell]

            # Set up the extrusion function
            extrude, faces = elemExtruder.get(mtype[:4], (None, None))
            elemNum = ho_key + (8 if cast(str, mtype).startswith('quad') else 6)
            faceMap = faceMaper.get(elemNum)

            # Consistency checks
            if faceMap is None:
                raise ValueError(f'Missing faceMap for element type {mtype}')
            if extrude is None or faces is None:
                hopout.error(f'Element type {mtype} not supported for extruding', traceback=True)

            # Obtain the element type
            elemType = elemTypeClass.inam[elemNum]
            if len(elemType) > 1:
                elemType  = str(elemType[0]).rstrip(digits)
                elemDOFs  = NDOFperElemType(elemType, mesh_vars.nGeo)
                elemType += str(elemDOFs)
            else:
                elemType  = str(elemType[0])

            # Face block: Iterate over each element
            for elem in cdata:
                # Generate new nodes
                extElems, newPoints = extrude(elem, points[elem], extrShifts, nPoints, nGeo)
                nPoints += len(newPoints)

                # Append the new point to the point list
                pointl.extend(newPoints)

                # Handle the first element, this fills the boundary conditions
                extElem = extElems[0]

                # Overwrite the element with the new indices
                # > The list is crucial here, we want to build a list of lists
                elems_lst.setdefault(elemType, []).extend([extElem])

                # Create the new faces
                subFaces   = tuple(np.array(extElem, dtype=np.int64)[face] for face in faces(nGeo))
                bcFaces    = [{} for _ in range(len(subFaces))]

                # BC: First, identify the 2D (bottom) faces
                # > We know this is the first face and 1/5 face
                botIdx , botFace = 0, subFaces[0]
                topIdx           = 1 if mtype.startswith('tria') else 5
                topFace, topName = [np.array(extElems[-1])[face] for face in faces(nGeo)][topIdx], mesh_vars.bcs[extrBCIndexTop].name  # noqa: E501

                # BC: Set the BC for the bottom face
                appendBCSet(botFace, faceMap, nFace, nFaces, nodeToFace, faceType,
                            csets_old  = csets_old      , csets_lst    = csets_lst, elems_lst  = elems_lst,  # noqa: E251, E271
                            bcFaces    = bcFaces        , bcFaceIdx    = botIdx   , bcSide     = 'bottom',   # noqa: E251, E271
                            requireDim = lambda n: n > 2, requireMatch = False    , allowMulti = False)      # noqa: E251, E271

                # ZONE: Set the ZONE for the bottom element
                appendBCSet(botFace, faceMap, nFace, nFaces, nodeToZone, faceType,
                            csets_old  = zsets_old      , csets_lst    = csets_lst, elems_lst  = elems_lst,  # noqa: E251, E271
                            bcFaces    = bcFaces        , bcFaceIdx    = botIdx   , bcSide     = 'zone',     # noqa: E251, E271
                            elemType   = elemType,                                                           # noqa: E251, E271
                            requireDim = lambda n: n > 2, requireMatch = False    , allowMulti = False)      # noqa: E251, E271

                # BC: Next, iterate over the 1D (side faces)
                for iFace, subFace in enumerate(subFaces[1::]):
                    appendBCSet(subFace, faceMap, nFace, nFaces, nodeToFace, faceType,
                                csets_old  = csets_old, csets_lst    = csets_lst, elems_lst  = elems_lst,    # noqa: E251, E271
                                bcFaces    = bcFaces  , bcFaceIdx    = iFace+1  , bcSide     = 'side',       # noqa: E251, E271
                                requireDim = 2        , requireMatch = False    , allowMulti = False)        # noqa: E251, E271

                for extElem in extElems[1:]:
                    # Overwrite the element with the new indices
                    # > The list is crucial here, we want to build a list of lists
                    elems_lst.setdefault(elemType, []).extend([extElem])

                    # Create the new faces
                    subFaces = tuple(np.array(extElem)[face] for face in faces(nGeo))
                    sidFaces = tuple((i, s) for i, s in enumerate(bcFaces) if ('side' in s and s['side'] == 'side'))
                    zonFaces = tuple((i, s) for i, s in enumerate(bcFaces) if ('side' in s and s['side'] == 'zone'))

                    for iFace, sidFace in sidFaces:
                        subFace = subFaces[iFace]
                        faceVal = faceMap(0) if len(subFace) == nFace else faceMap(1)

                        name = sidFace['name']
                        csets_lst.setdefault(name.strip(), [[] for _ in range(len(faceType))])
                        csets_lst[name][faceVal].append(nFaces[faceVal])

                        nFaces[faceVal] += 1
                        elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))

                    # Assign the new elements to the zone
                    if zonFaces:
                        faceVal = faceType.index(elemType)
                        name    = zonFaces[0][1]['name']
                        # Append the volume zones and increment
                        csets_lst[name][faceVal].append(nFaces[faceVal])
                        nFaces[faceVal] += 1

                # BC: We should have one face left, assign the bottom BC
                # > We need to hardcode this since we might have internal faces
                faceVal = faceMap(0) if len(topFace) == nFace else faceMap(1)
                csets_lst.setdefault(topName, [[] for _ in range(len(faceType))])
                csets_lst[topName][faceVal].append(nFaces[faceVal])

                nFaces[faceVal] += 1
                elems_lst[faceType[faceVal]].append(np.array(topFace, dtype=int))

                # Update the progress bar
                bar.step()

    # Close the progress bar
    bar.close()

    # Convert lists to NumPy arrays for elems_new and csets_new
    elems_new = {}
    csets_new = {}

    for key in elems_lst:  # noqa: PLC0206
        if   isinstance(elems_lst[key], list) and     elems_lst[key]:  # noqa: E271
            # Convert the list of accumulated arrays/lists into a single NumPy array
            elems_new[key] = np.array(elems_lst[key], dtype=int)
        elif isinstance(elems_lst[key], list) and not elems_lst[key]:
            # Determine the expected number of columns
            elems_new[key] = np.empty((0, faceNum[faceType.index(key)]), dtype=int)

    for key in csets_lst:  # noqa: PLC0206
        csets_new[key] = tuple(np.array(lst, dtype=int) for lst in csets_lst[key])

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = elems_new,  # noqa: E251
                         cell_sets = csets_new)  # noqa: E251

    # Temporarily assign mesh_vars.mesh
    with temporary_assign(mesh_vars, 'mesh', mesh):
        # Compute the equidistant point set used by meshIO
        xEq:       Final[npt.NDArray[np.float64]] = np.linspace(-1., 1., nGeo+1)
        wBaryEq:   Final[npt.NDArray[np.float64]] = barycentric_weights(nGeo+1, xEq)

        xGP, wGP  = legendre_gauss_nodes(nGeo+1)
        DGP:       Final[npt.NDArray[np.float64]] = polynomial_derivative_matrix(nGeo+1, xGP)
        VdmEqToGP: Final[npt.NDArray[np.float64]] = calc_vandermonde(nGeo+1, nGeo+1, wBaryEq, xEq, xGP)

        # Compute the weights
        weights:   Final[npt.NDArray[np.float64]] = np.outer(wGP, wGP)  # Shape: (N_GP+1, N_GP+1)

        # Check if the surface normal of the first cell points outwards
        for elemType in tuple(s for s in mesh.cells_dict if 'hexahedron' in s):
            # Only check the first element, other elements get covered in OrientMesh
            ionodes:   npt.NDArray = mesh.get_cells_type(elemType)[0]
            elemNames: Final[dict] = mesh_vars.ELEMTYPE.name

            # Convert elemType to HOPR integer format
            if isinstance(elemType, str):
                elemType = elemNames[elemType]

            elem = ELEM(type=elemType, nodes=ionodes, elemID=0, sides=None)
            results = check_orientation(elem, VdmEqToGP, DGP, weights)
            if results and any(not res[0] for res in results):
                hopout.error('Extruded element has inward pointing normal vector. Wrong extrusion direction?')

    # Run garbage collector to release memory
    gc.collect()

    hopout.sep()
    return mesh


@cache
def quad_meshio_to_ij(order: int) -> npt.NDArray[np.int64]:
    """Return meshio quad-node ordering mapped to tensor indices (i,j).

    The ordering is generated ring-by-ring:
    corners, edge nodes, then inner rings, matching meshio high-order quad ordering.
    """
    if order < 1:
        raise ValueError(f'Quad ordering requires order >= 1, got {order}')

    ij: list[tuple[int, int]] = []

    # Fill rings from outside to inside
    for ring in range(order // 2 + 1):
        start = ring
        end   = order - ring

        if start > end:
            break

        # Odd order: final center point
        if start == end:
            ij.append((start, start))
            break

        # Ring corners
        ij.extend(((start, start), (end, start), (end, end), (start, end)))
        # Bottom edge (left -> right), excluding corners
        ij.extend((i, start) for i in range(start + 1, end))
        # Right edge (bottom -> top), excluding corners
        ij.extend((end, j) for j in range(start + 1, end))
        # Top edge (right -> left), excluding corners
        ij.extend((i, end) for i in range(end - 1, start, -1))
        # Left edge (top -> bottom), excluding corners
        ij.extend((start, j) for j in range(end - 1, start, -1))

    expected = (order + 1) ** 2
    if len(ij) != expected:
        raise RuntimeError(f'Invalid quad ordering length for order {order}: got {len(ij)}, expected {expected}')

    return np.asarray(ij, dtype=np.int64)


@cache
def tri_meshio_to_ij(order: int) -> npt.NDArray[np.int64]:
    """Return meshio triangle-node ordering mapped to simplex indices (i,j)."""
    if order < 1:
        raise ValueError(f'Triangle ordering requires order >= 1, got {order}')

    ij: list[tuple[int, int]] = []

    # Fill triangular rings from outside to inside.
    # After one ring is emitted, the inner triangle has order reduced by 3.
    def _fill_ring(n: int, i0: int, j0: int) -> None:
        if n < 0:
            return
        if n == 0:
            ij.append((i0, j0))
            return

        # Ring corners
        ij.extend(((i0, j0), (i0+n, j0), (i0, j0+n)))
        # Edge 0->1 (excluding corners)
        ij.extend((i0+i, j0) for i in range(1, n))
        # Edge 1->2 (excluding corners)
        ij.extend((i0+n-t, j0+t) for t in range(1, n))
        # Edge 2->0 (excluding corners)
        ij.extend((i0, j0+j) for j in range(n-1, 0, -1))

        _fill_ring(n-3, i0+1, j0+1)

    _fill_ring(order, 0, 0)

    expected = round((order + 1) * (order + 2) / 2)
    if len(ij) != expected:
        raise RuntimeError(f'Invalid triangle ordering length for order {order}: got {len(ij)}, expected {expected}')

    return np.asarray(ij, dtype=np.int64)


def extrude_pris(nodes:   npt.NDArray,
                 points:  npt.NDArray,
                 shifts:  npt.NDArray,
                 nPoints: int,
                 order:   int) -> tuple[list[npt.NDArray], npt.NDArray]:

    nDOFsElem = round((order+1)**(3-1)*(order+2)/2.)
    newNodes  = [np.empty((nDOFsElem, )) for _ in range(len(shifts)-1)]

    if order == 0:
        raise ValueError(f'Extrusion not implemented for NGeo={order}')

    # Local imports ----------------------------------------
    from pyhope.mesh.mesh_common import LINMAP
    # ------------------------------------------------------

    nFaceDOFs = round((order+1)*(order+2)/2.)
    nNewDOFs  = order*nFaceDOFs
    newPoints = np.empty((nNewDOFs*(shifts.shape[0]-1), 3))

    # Generic meshio(triaN) -> meshio(wedgeM) layer mapping for any supported NGeo>=1.
    linmap = LINMAP(206, order)
    q_to_ij  = tri_meshio_to_ij(order)
    layerPos = np.empty((order+1, nFaceDOFs), dtype=np.int64)
    for q, (ii, jj) in enumerate(q_to_ij):
        for k in range(order+1):
            layerPos[k, q] = int(linmap[ii, jj, k])

    # Append the bottom layer of the first element, then stack all the other elements
    for i in range(shifts.shape[0]-1):
        offsetCurr = i*nNewDOFs
        shiftCurr  = shifts[i+1, :]
        shiftPrev  = shifts[i  , :]

        # Bottom layer
        for q in range(nFaceDOFs):
            idxBot = int(layerPos[0    , q])
            idxTop = int(layerPos[order, q])
            newNodes[i][idxBot] = nodes[q] if i == 0 else newNodes[i-1][idxTop]

        # New points for layers k=1..order
        p = 0
        for k in range(1, order+1):
            alpha  = k/order
            shiftK = (1.0-alpha)*shiftPrev + alpha*shiftCurr
            for q in range(nFaceDOFs):
                idx = nPoints + offsetCurr + p
                pos = int(layerPos[k, q])

                newNodes[i][pos] = idx
                newPoints[offsetCurr + p, :] = points[q] + shiftK
                p += 1

    return newNodes, newPoints


def extrude_hexa(nodes:   npt.NDArray,
                 points:  npt.NDArray,
                 shifts:  npt.NDArray,
                 nPoints: int,
                 order:   int) -> tuple[list[npt.NDArray], npt.NDArray]:

    nDOFsElem = (order+1)**3
    newNodes  = [np.empty((nDOFsElem, )) for _ in range(len(shifts)-1)]

    if order == 0:
        raise ValueError(f'Extrusion not implemented for NGeo={order}')

    # Local imports ----------------------------------------
    from pyhope.mesh.mesh_common import LINMAP
    # ------------------------------------------------------

    nFaceDOFs = (order+1)**2
    nNewDOFs  = order*nFaceDOFs
    newPoints = np.empty((nNewDOFs*(shifts.shape[0]-1), 3))

    # Generic meshio(quadN) -> meshio(hexaM) layer mapping for any NGeo>=1.
    linmap   = LINMAP(208, order)
    q_to_ij  = quad_meshio_to_ij(order)
    layerPos = np.empty((order+1, nFaceDOFs), dtype=np.int64)
    for q, (ii, jj) in enumerate(q_to_ij):
        for k in range(order+1):
            layerPos[k, q] = int(linmap[ii, jj, k])

    # Append the bottom layer of the first element, then stack all the other elements
    for i in range(shifts.shape[0]-1):
        offsetCurr = i*nNewDOFs
        shiftCurr  = shifts[i+1, :]
        shiftPrev  = shifts[i  , :]

        # Bottom layer
        for q in range(nFaceDOFs):
            idxBot = int(layerPos[0    , q])
            idxTop = int(layerPos[order, q])
            newNodes[i][idxBot] = nodes[q] if i == 0 else newNodes[i-1][idxTop]

        # New points for layers k=1...order
        p = 0
        for k in range(1, order+1):
            alpha  = k/order
            shiftK = (1.0-alpha)*shiftPrev + alpha*shiftCurr
            for q in range(nFaceDOFs):
                idx = nPoints + offsetCurr + p
                pos = int(layerPos[k, q])

                newNodes[i][pos] = idx
                newPoints[offsetCurr + p, :] = points[q] + shiftK
                p += 1

    return newNodes, newPoints


@cache
def pris_faces(order: int) -> tuple[npt.NDArray, ...]:
    """
    Given the 6 prism corner indices, return a tuple with the 2 triangular and 3 quadrilateral faces as arrays
    """
    match order:
        case 1:
            return (# Triangular faces  # noqa: E261
                    np.array((  0,  1,  2    ), dtype=int),
                    np.array((  3,  4,  5    ), dtype=int),
                    # Quadrilateral faces
                    np.array((  0,  1,  4,  3), dtype=int),
                    np.array((  1,  2,  5,  4), dtype=int),
                    np.array((  2,  0,  3,  5), dtype=int))
        # INFO: It would be better to return the actual high-order faces here but PyHOPE will automatically fallback to the corner
        #       nodes if the inner nodes are not available
        case _:
            return (# Triangular faces  # noqa: E261
                    np.array((  0,  1,  2    ), dtype=int),
                    np.array((  3,  4,  5    ), dtype=int),
                    # Quadrilateral faces
                    np.array((  0,  1,  4,  3), dtype=int),
                    np.array((  1,  2,  5,  4), dtype=int),
                    np.array((  2,  0,  3,  5), dtype=int))


@cache
def hexa_faces(order: int) -> tuple[npt.NDArray, ...]:
    """ Given the indices of a hexahedral element, return a tuple with the 6 faces as arrays
    """
    match order:
        case 1:
            return (np.array((  0,  1,  2,  3), dtype=int),
                    np.array((  0,  1,  5,  4), dtype=int),
                    np.array((  1,  2,  6,  5), dtype=int),
                    np.array((  2,  6,  7,  3), dtype=int),
                    np.array((  0,  4,  7,  3), dtype=int),
                    np.array((  4,  5,  6,  7), dtype=int))
        # INFO: It would be better to return the actual high-order faces here but PyHOPE will automatically fallback to the corner
        #       nodes if the inner nodes are not available
        case _:
            return (np.array((  0,  1,  2,  3), dtype=int),
                    np.array((  0,  1,  5,  4), dtype=int),
                    np.array((  1,  2,  6,  5), dtype=int),
                    np.array((  2,  6,  7,  3), dtype=int),
                    np.array((  0,  4,  7,  3), dtype=int),
                    np.array((  4,  5,  6,  7), dtype=int))
