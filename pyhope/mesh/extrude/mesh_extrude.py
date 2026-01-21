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
import importlib.util
import os
import sys
from collections import defaultdict
from functools import cache
from string import digits
from typing import cast
from typing import Optional
from types import ModuleType
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


def MeshExtrude(mesh: meshio.Mesh) -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.common.common_progress import ProgressBar
    from pyhope.config.config import prmfile
    from pyhope.io.io_gmsh import GMSHCELLTYPES
    from pyhope.mesh.mesh_common import NDOFperElemType
    from pyhope.mesh.mesh_vars import nGeo
    from pyhope.readintools.readintools import GetInt, GetReal, GetStr
    # ------------------------------------------------------

    points: np.ndarray = mesh.points

    # Instantiate the Gmsh cell type mapping
    gmshCellTypes = GMSHCELLTYPES()

    # Check if the mesh is already 3D
    if [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes3D]:
        return mesh

    # Check if the mesh contains 2D elements to extrude
    if       [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes2D] and not mesh_vars.doExtrude:  # noqa: E271
        hopout.error('Mesh contains suitable surface cells for extrusion but MeshExtrude=F, exiting...')
    elif not [cell_block for cell_block in mesh.cells if cell_block.type in gmshCellTypes.cellTypes2D]:
        hopout.error('Mesh contains no suitable surface cells for extrusion, exiting...')

    hopout.info('Extruding surface to volume mesh')

    # Read in the mesh post-deformation flag
    hopout.sep()
    meshExtrNum      = GetInt( 'MeshExtrudeElems')
    meshExtrLength   = GetReal('MeshExtrudeLength')
    meshExtrTemplate = GetStr( 'MeshExtrudeTemplate')
    meshExtrBCIndex  = GetInt( 'MeshExtrudeBCIndex')

    # Continue with extrusion
    hopout.sep()
    hopout.routine('  Template: {}'.format(meshExtrTemplate))

    # Define locations of the transformation files ( Priority: prmfile folder > CWD > templates )
    ExtrudeLocations = [
        os.path.join(os.path.dirname(prmfile), f'{meshExtrTemplate}.py'),                # Search folder of parameter file
        os.path.join(os.getcwd(), f'{meshExtrTemplate}.py'),                             # Search in CWD
        os.path.join(os.path.dirname(__file__), 'templates', f'{meshExtrTemplate}.py')   # Search in 'templates'
    ]

    # Check if the transformation file exists
    ExtrudeMod: Optional[ModuleType] = None
    for loc in ExtrudeLocations:
        if os.path.exists(loc):
            spec = importlib.util.spec_from_file_location(meshExtrTemplate, loc)
            # Skip to the next location if spec is None
            if spec is None:
                continue

            ExtrudeMod = importlib.util.module_from_spec(spec)
            sys.modules[meshExtrTemplate] = ExtrudeMod
            spec.loader.exec_module(ExtrudeMod)

            # Output filename of template
            hopout.routine('     found: {}'.format(loc))

            # Stop once the module is successfully loaded
            break

    # If the transformation file is not found, exit
    if ExtrudeMod is None:
        hopout.warning(f'Extrusion template "{meshExtrTemplate}" not found!')
        # Print all available default templates for post-deformation
        templist = []
        for file in os.listdir(os.path.join(os.path.dirname(__file__), 'templates')):
            if file.endswith('.py'):
                templist.append(f'  {file[:-3]}')
        hopout.error('Available default extrusion templates:' + ','.join(templist))

    # Setup the extrusion
    meshExtrShifts = ExtrudeMod.ExtrudeTemplate(meshExtrNum, meshExtrLength)

    # Copy original points
    pointl    = cast(list, mesh.points.tolist())
    elems_old = mesh.cells.copy()
    cell_sets = getattr(mesh, 'cell_sets', {})

    # Get base key to distinguish between linear and high-order elements
    ho_key = 100 if nGeo == 1 else 200

    nPoints  = len(pointl)
    nFaces   = np.zeros(2, dtype=int)
    match nGeo:
        case 1:
            faceType = ['triangle'  , 'quad'  ]
            faceNum  = [          3 ,       4 ]
        case 2:
            faceType = ['triangle6' , 'quad9' ]
            faceNum  = [          6 ,       9 ]
        case 3:
            faceType = ['triangle10', 'quad16']
            faceNum  = [         10 ,      16 ]
        case 4:
            faceType = ['triangle15', 'quad25']
            faceNum  = [         15 ,      25 ]
        case _:
            hopout.error('nGeo = {} not supported for element splitting'.format(nGeo))

    # Prepare new cell blocks and new cell_sets
    elems_lst = {ftype: [] for ftype in faceType}
    csets_lst = {}

    # Set up the element extrusion function
    elemExtruder = {'tria': (extrude_pris, pris_faces ),
                    'quad': (extrude_hexa, hexa_faces )}
    faceMaper = { ho_key + 4: lambda x: 0,
                  ho_key + 5: lambda x: 0 if x == 0 else 1,
                  ho_key + 6: lambda x: 0 if x == 0 else 1,
                  ho_key + 8: lambda x: 1}
    nFace   = (nGeo+1)*(nGeo+2)/2

    # Convert the (1D, 2D) boundary cell set into a dictionary
    csets_old = {}

    for cname, cblock in cell_sets.items():
        # Each set_blocks is a list of arrays, one entry per cell block
        for blockID, block in enumerate(cblock):
            if elems_old[blockID].type[:4] not in ('line', 'tria', 'quad'):
                continue

            # Ignore the empty zones
            if block is None:
                continue

            # Sort them as a set for membership checks
            for face in block:
                # nodes = mesh.cells_dict[elems_old[blockID].type][face]
                nodes = mesh.cells[blockID].data[face]
                csets_old.setdefault(frozenset(nodes), []).append(cname)

    # Create the element sets
    meshcells = tuple((k, v) for k, v in mesh.cell_sets_dict.items() if any(key.startswith('tria') for key in v.keys())
                                                                     or any(key.startswith('quad') for key in v.keys()))

    match len(meshcells):
        case 0:
            hopout.error('Could not found boundary condition for extrusion, exiting...')
            # If meshcells is empty, we fake it assign it to Zone1
            # meshcells = tuple(('Zone1', {k: np.array([i for i in range(len(v))])}) for k, v in mesh.cells_dict.items()
            #                                                                                 if (k.startswith('tria')
            #                                                                                 or  k.startswith('quad')))  # noqa: E271, E501
        case 1:
            pass
        case _:
            hopout.error('Found more than one boundary condition for extrusion, exiting...')

    nTotalElems = sum(cdata.shape[0] for _, zdata in meshcells for _, cdata in cast(dict, zdata).items())

    bar = ProgressBar(value=nTotalElems, title='│             Processing Elements', length=33, threshold=1000)

    # Build an inverted index to map each node to all face keys (from csets_old) that contain it
    nodeToFace = defaultdict(set)
    for subFace in csets_old:
        for node in subFace:
            nodeToFace[node].add(subFace)

    # We need to unwrap meshcells
    for iElem, meshcell in enumerate(meshcells):
        _    , mdict = meshcell

        for iType in range(len(mdict)):
            mtype, mcell = list(cast(dict, mdict).keys())[iType], list(cast(dict, mdict).values())[iType]

            extrude, faces = elemExtruder.get(mtype[:4], (None, None))
            elemNum = ho_key + (8 if cast(str, mtype).startswith('quad') else 6)
            faceMap = faceMaper.get(elemNum, None)

            # Sanity check
            if faceMap is None:
                raise ValueError('Missing faceMap for element type {}'.format(mtype))

            cdata = mesh.get_cells_type(mtype)[mcell]

            if extrude is None or faces is None:
                hopout.error('Element type {} not supported for extruding'.format(mtype), traceback=True)

            # Obtain the element type
            elemType = elemTypeClass.inam[elemNum]
            if len(elemType) > 1:
                elemType  = elemType[0].rstrip(digits)
                elemDOFs  = NDOFperElemType(elemType, mesh_vars.nGeo)
                elemType += str(elemDOFs)
            else:
                elemType  = elemType[0]
                elemDOFs  = NDOFperElemType(elemType, mesh_vars.nGeo)

            # Face block: Iterate over each element
            for elem in cdata:
                # Generate new nodes
                extElems, newPoints = extrude(elem, points[elem], meshExtrShifts, nPoints, nGeo)
                nPoints += len(newPoints)

                # Append the new point to the point list
                pointl.extend(newPoints)

                # Handle the first element, this fills the boundary conditions
                extElem = extElems[0]

                # Overwrite the element with the new indices
                # > The list is crucial here, we want to build a list of lists
                elems_lst.setdefault(elemType, []).extend([extElem])

                # Create the new faces
                subFaces   = [np.array(extElem)[face] for face in faces(nGeo)]
                bcFaces    = [{} for s in range(len(subFaces))]
                bottomName = mesh_vars.bcs[meshExtrBCIndex-1].name

                # BC: First, find the 2D (bottom) face
                # > Iterate over a copy, so elements can be removed
                for iFace, subFace in enumerate(subFaces):
                    faceVal = faceMap(0) if len(subFace) == nFace else faceMap(1)
                    faceSet = frozenset(subFace)

                    # Get candidate cset keys using the nodes in the face
                    candidate_sets = [nodeToFace[node] for node in faceSet if node in nodeToFace]
                    # Filter only 2D set
                    candidate_sets = [filtered for s in candidate_sets if (filtered := {fs for fs in s if len(fs) > 2})]
                    if not candidate_sets:
                        continue

                    common_candidates = set.intersection(*candidate_sets)
                    for candidate in common_candidates:
                        # Check if the subFace is indeed a subset of the candidate from csets_old
                        if faceSet.issubset(candidate):
                            # Use the associated boundary name
                            names = csets_old[candidate]

                            if len(names) > 1:
                                hopout.error(f'Matched more than one BC [{names}] during extrusion, exiting...', traceback=True)

                            # Update csets_lst for each name in the list.
                            (name,) = names
                            csets_lst.setdefault(name.strip(), [[], []])
                            csets_lst[name][faceVal].append(nFaces[faceVal])

                            # Store the 2D face and the (unique) name
                            bcFaces[iFace] = {'name': name.strip(),
                                              'side': 'bottom'
                                             }

                            nFaces[faceVal] += 1
                            elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))

                # BC: Next, iterate over the 1D (side faces)
                for iFace, subFace in enumerate(subFaces):
                    faceVal = faceMap(0) if len(subFace) == nFace else faceMap(1)
                    faceSet = frozenset(subFace)

                    # Get candidate cset keys using the nodes in the face
                    candidate_sets = [nodeToFace[node] for node in faceSet if node in nodeToFace]
                    # Filter only 1D set
                    candidate_sets = [filtered for s in candidate_sets if (filtered := {fs for fs in s if len(fs) == 2})]
                    if not candidate_sets:
                        continue

                    common_candidates = set.intersection(*candidate_sets)
                    for candidate in common_candidates:
                        # Check if the subFace is indeed a subset of the candidate from csets_old
                        if candidate.issubset(faceSet):
                            # Use the associated boundary name
                            names = csets_old[candidate]

                            if len(names) > 1:
                                hopout.error(f'Matched more than one BC [{names}] during extrusion, exiting...', traceback=True)

                            # Update csets_lst for each name in the list.
                            (name,) = names
                            csets_lst.setdefault(name.strip(), [[], []])
                            csets_lst[name][faceVal].append(nFaces[faceVal])

                            # Store the 1D faces
                            bcFaces[iFace] = {'name': name.strip(),
                                              'side': 'side'
                                             }

                            nFaces[faceVal] += 1
                            elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))

                for extElem in extElems[1:]:
                    # Overwrite the element with the new indices
                    # > The list is crucial here, we want to build a list of lists
                    elems_lst.setdefault(elemType, []).extend([extElem])

                    # Create the new faces
                    subFaces = [np.array(extElem)[face] for face in faces(nGeo)]
                    sidFaces = [(i, s) for i, s in enumerate(bcFaces) if ('side' in s.keys() and s['side'] == 'side')]

                    for iFace, sidFace in sidFaces:
                        subFace = subFaces[iFace]
                        faceVal = faceMap(0) if len(subFace) == nFace else faceMap(1)

                        name = sidFace['name']
                        csets_lst.setdefault(name.strip(), [[], []])
                        csets_lst[name][faceVal].append(nFaces[faceVal])

                        nFaces[faceVal] += 1
                        elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))

                # BC: We should have one face left, assign the bottom BC
                # > We need to hardcode this since we might have internal faces
                topIndex = 1 if mtype.startswith('tria') else 5
                subFaces = [np.array(extElems[-1])[face] for face in faces(nGeo)]
                subFace  = subFaces[topIndex]
                faceVal    = faceMap(0) if len(subFace) == nFace else faceMap(1)
                csets_lst.setdefault(bottomName, [[], []])
                csets_lst[bottomName][faceVal].append(nFaces[faceVal])

                nFaces[faceVal] += 1
                elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))

                # topFaces = [(i, s) for i, s in enumerate(bcFaces) if len(s) == 0]
                # match len(topFaces):
                #     case 1:
                #         (topFace,) = topFaces
                #         subFaces   = [np.array(extElems[-1])[face] for face in faces(nGeo)]
                #         subFace    = subFaces[topFace[0]]
                #         faceVal    = faceMap(0) if len(subFace) == nFace else faceMap(1)
                #         csets_lst.setdefault(bottomName, [[], []])
                #         csets_lst[name][faceVal].append(nFaces[faceVal])
                #
                #         nFaces[faceVal] += 1
                #         elems_lst[faceType[faceVal]].append(np.array(subFace, dtype=int))
                #     case _:
                #         hopout.error(f'Matched more than one BC [{names}] during extrusion, exiting...', traceback=True)

            # Update the progress bar
            bar.step()

    # Close the progress bar
    bar.close()

    # Convert lists to NumPy arrays for elems_new and csets_new
    elems_new = {}
    csets_new = {}

    for key in elems_lst:
        if   isinstance(elems_lst[key], list) and     elems_lst[key]:  # noqa: E271
            # Convert the list of accumulated arrays/lists into a single NumPy array
            elems_new[key] = np.array(elems_lst[key], dtype=int)
        elif isinstance(elems_lst[key], list) and not elems_lst[key]:
            # Determine the expected number of columns
            elems_new[key] = np.empty((0, faceNum[faceType.index(key)]), dtype=int)

    for key in csets_lst:
        csets_new[key] = tuple(np.array(lst, dtype=int) for lst in csets_lst[key])

    # Convert points_list back to a NumPy array
    points = np.array(pointl)

    mesh   = meshio.Mesh(points    = points,     # noqa: E251
                         cells     = elems_new,  # noqa: E251
                         cell_sets = csets_new)  # noqa: E251

    hopout.sep()
    return mesh


def extrude_pris(nodes:   np.ndarray,
                 points:  np.ndarray,
                 shifts:  np.ndarray,
                 nPoints: int,
                 order:   int) -> tuple[list, ...]:

    newPoints   = []
    newNodes    = [[] for s in range(len(shifts)-1)]
    newNodes[0] = nodes.squeeze().tolist()

    match order:
        case 1:
            # Append the bottom layer the first element
            newNodes[ 0].extend(np.add(np.arange(3, dtype=np.int64), nPoints).tolist())
            newPoints.extend((points + shifts[1, :]).squeeze().tolist())

            # Stack all the other elements
            for i in range(1, shifts.shape[0]-1):
                newNodes[i]  = np.add(np.arange(3, dtype=np.int64), nPoints+(i-1)*3).tolist()
                newNodes[i] += np.add(np.arange(3, dtype=np.int64), nPoints+(i  )*3).tolist()
                newPoints.extend((points + shifts[i+1, :]).squeeze().tolist())

        # FIXME: Implement the other orders

    return newNodes, newPoints


def extrude_hexa(nodes:   np.ndarray,
                 points:  np.ndarray,
                 shifts:  np.ndarray,
                 nPoints: int,
                 order:   int) -> tuple[list, ...]:

    newPoints   = []
    newNodes    = [[] for s in range(len(shifts)-1)]
    newNodes[0] = nodes.squeeze().tolist()

    match order:
        case 1:
            # Append the bottom layer the first element
            newNodes[ 0].extend(np.add(np.arange(4, dtype=np.int64), nPoints).tolist())
            newPoints.extend((points + shifts[1, :]).squeeze().tolist())

            # Stack all the other elements
            for i in range(1, shifts.shape[0]-1):
                newNodes[i]  = np.add(np.arange(4, dtype=np.int64), nPoints+(i-1)*4).tolist()
                newNodes[i] += np.add(np.arange(4, dtype=np.int64), nPoints+(i  )*4).tolist()
                newPoints.extend((points + shifts[i+1, :]).squeeze().tolist())

        # FIXME: Implement the other orders

    return newNodes, newPoints


@cache
def pris_faces(order: int) -> tuple[np.ndarray, ...]:
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
def hexa_faces(order: int) -> tuple[np.ndarray, ...]:
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
