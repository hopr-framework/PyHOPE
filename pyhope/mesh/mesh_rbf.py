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
import os
import resource
from string import digits
from typing import Any, Final, cast
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


def ReadSurfMesh(meshFile: str) -> meshio.Mesh:
    """ Read a subdivided surface mesh
    """
    # Third-party libraries --------------------------------
    import gmsh
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.io.io_vars import debugvisu
    import pyhope.output.output as hopout
    from pyhope.common.common_vars import np_mtp
    from pyhope.meshio.meshio_convert import gmsh_to_meshio
    # ------------------------------------------------------

    # Setup stacksize
    resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    gmsh.initialize()

    # Setup multiprocessing
    numThreads = np_mtp if np_mtp > 0 else 1
    gmsh.option.setNumber('General.NumThreads',   numThreads)
    gmsh.option.setNumber('Geometry.OCCParallel', 1 if np_mtp > 0 else 0)

    # Setup mesh factory
    # gmsh.option.setString('SetFactory', 'OpenCascade')

    # Setup debug visualization
    if not debugvisu:
        # Hide the GMSH debug output
        gmsh.option.setNumber('General.Terminal', 0)

    # Get file extension
    _, ext = os.path.splitext(meshFile)

    gmsh.option.setNumber('Mesh.RecombineAll', 1)
    # gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 0)

    # If not GMSH format convert
    if ext == '.cgns':
        # Setup GMSH to import required data
        # gmsh.option.setNumber('Mesh.SaveAll', 1)
        gmsh.option.setNumber('Mesh.CgnsImportIgnoreBC', 0)
        gmsh.option.setNumber('Mesh.CgnsImportIgnoreSolution', 1)

    # Set the element order
    # > Technically, this is only required in generate_mesh but let's be precise here
    gmsh.model.mesh.setOrder(mesh_vars.nGeo)
    gmsh.merge(meshFile)

    # Explicitly load the OpenCASCADE kernel
    gmsh.model.occ.synchronize()

    # gmsh.model.geo.synchronize()
    gmsh.model.occ.synchronize()

    # Optimize the high-order mesh
    # gmsh.model.mesh.optimize(method='Relocate3D', force=True)
    # gmsh.model.occ.synchronize()

    # Reclassify the nodes to ensure correct node ordering
    gmsh.model.mesh.reclassifyNodes()
    gmsh.model.occ.synchronize()

    # Consistency check if the mesh contains surface elements
    gmshTypes = gmsh.model.mesh.getElementTypes()
    gmshElems = np.asarray([(elemName, order) for type                          in gmshTypes                                     # noqa: E272
                                               for elemName, dim, order, _, _, _ in [gmsh.model.mesh.getElementProperties(type)]  # noqa: E272
                              if dim == 2])
    if not np.any(gmshElems):
        hopout.error('Generated mesh does not contain surface elements, exiting...')

    # Consistency check if the mesh elements have the correct order
    gmshIssue  = np.asarray([(elemName, order) for type                          in gmshTypes                                     # noqa: E272
                                               for elemName, dim, order, _, _, _ in [gmsh.model.mesh.getElementProperties(type)]  # noqa: E272
                              if dim == 2 and order != 1])

    if gmshIssue.size > 0:
        for elem in gmshIssue:
            print(hopout.warn(f'Wrong Gmsh order {elem[1]} for element {elem[0].replace(" ", "")}'))
        elemOrders = set([int(elem[1]) for elem in gmshIssue])
        hopout.error(f'Gmsh element order(s) {elemOrders} does not match requested mesh order {set([mesh_vars.nGeo])}')

    # Convert Gmsh object to meshio object
    mesh = gmsh_to_meshio(gmsh)

    # Finally done with GMSH, finalize
    gmsh.finalize()

    # Convert BC names to lower case
    mesh.cell_sets = {k.lower(): v for k, v in mesh.cell_sets.items()}

    # Run garbage collector to release memory
    gc.collect()

    return mesh


def ApplyRBFCurving(mesh: meshio.Mesh, splitMesh: meshio.Mesh) -> meshio.Mesh:

    # TODO: STILL NEED THE ACTUAL MESH CURVING
    return mesh


def MeshCurveRBF(mesh: meshio.Mesh) -> meshio.Mesh:
    """ Curve a linear mesh during postprocessing
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.basis.basis_basis import barycentric_weights, calc_vandermonde, change_basis_3D
    from pyhope.mesh.mesh_common import LINTEN, NDOFperElemType
    from pyhope.mesh.mesh_vars import ELEMTYPE
    from pyhope.readintools.readintools import CountOption, GetStr
    # ------------------------------------------------------

    # Linear meshes need not be curved
    if mesh_vars.nGeo <= 1:
        return mesh

    if CountOption('SplitElemFile') <= 0:
        return mesh

    hopout.routine('Curving with radial-basis functions')
    hopout.sep()

    SplitElemFile = GetStr('SplitElemFile')

    # Leave if no curving is required
    if SplitElemFile is None:
        return mesh

    # Load the previous mesh
    points   = mesh.points
    pointl   = cast(list, points.tolist())
    cells    = mesh.cells_dict
    cellsets = mesh.cell_sets_dict
    elemNames: Final[dict] = mesh_vars.ELEMTYPE.name
    elemKeys : Final       = mesh_vars.ELEMTYPE.type.keys()

    # Instantiate ELEMTYPE
    elemTypeClass = ELEMTYPE()

    # Compute the equidistant point set used by HOPR
    xEqHdf5     = np.linspace(-1, 1, num=2, dtype=np.float64)
    wBaryEqHdf5 = barycentric_weights(1, xEqHdf5)

    # Compute the equidistant point set used by meshIO
    xEqMesh     = np.linspace(-1, 1, num=mesh_vars.nGeo+1, dtype=np.float64)

    # Compute the Vandermonde matrix
    VdmEqHdf5ToEqMesh = calc_vandermonde(2, mesh_vars.nGeo+1, wBaryEqHdf5, xEqHdf5, xEqMesh)

    # Pre-compute LINTEN mappings for all element types
    # > Cache the mapping here, so we consider the mesh order
    linCache  = {}
    mapCache  = {}
    elemTypes = tuple([s + 200 for s in (4, 5, 6, 8)])
    for elemType in elemTypes:
        try:
            # Forward mapping
            _, mapLin = LINTEN(elemType, order=mesh_vars.nGeo)
            mapLin    = np.array(tuple(mapLin[np.int64(i)] for i in range(len(mapLin))))
            linCache[elemType] = mapLin
            # Reverse mapping
            _, linMap = LINTEN(elemType, order=1)
            linMap    = np.array(tuple(linMap[np.int64(i)] for i in range(len(linMap))))
            mapCache[elemType] = linMap
        # Only hexahedrons supported for specific nGeo
        except ValueError:
            pass

    # Convert the mesh to the new (higher) nGeo
    nElems = 0
    for elemType in mesh.cells_dict.keys():
        # Only consider three-dimensional types
        if not any(s in elemType for s in elemKeys):
            continue

        # Load one cell type at a time
        ioelems  = mesh.get_cells_type(elemType)
        nIOElems = ioelems.shape[0]

        if isinstance(elemType, str):
            elemType = elemNames[elemType]

        # Assert that the previous elemType is linear
        if elemType >= 200:
            hopout.error('RBF curving only available with linear base mesh!')

        # Obtain the previous element type
        elemNum  = elemType
        elemType = elemTypeClass.inam[elemNum]
        if len(elemType) > 1:
            elemType  = elemType[0].rstrip(digits)
            elemDOFs  = NDOFperElemType(elemType, mesh_vars.nGeo)
            elemType += str(elemDOFs)
        else:
            elemType  = elemType[0]
            elemDOFs  = NDOFperElemType(elemType, mesh_vars.nGeo)

        # Obtain the new element type
        elemNum  = elemNum % 100 + 200
        meshType = elemTypeClass.inam[elemNum]
        if len(meshType) > 1:
            meshType  = meshType[0].rstrip(digits)
            meshDOFs  = NDOFperElemType(meshType, mesh_vars.nGeo)
            meshType += str(meshDOFs)
        else:
            meshType  = meshType[0]
            meshDOFs  = NDOFperElemType(meshType, mesh_vars.nGeo)

        # ChangeBasis currently only supported for hexahedrons
        mapLin = linCache[elemNum]
        linMap = mapCache[elemNum]

        # Iterate over all elements
        for iElem in range(nElems, nElems + nIOElems):
            nElemNode = NDOFperElemType(meshType, mesh_vars.nGeo)
            elemIDs   = np.arange(len(pointl), len(pointl)+nElemNode, dtype=np.uint64)
            elemNodes = elemIDs[mapLin[:nElemNode]]
            # This needs no offset as we already accounted for the number of points in elemIDs
            elemNodes = np.expand_dims(elemNodes, axis=0)

            # This is still in tensor-product format
            meshNodes = points[ioelems[iElem - nElems]][linMap].reshape((2, 2, 2, 3)).transpose(3, 0, 1, 2)
            try:
                meshNodes = change_basis_3D(VdmEqHdf5ToEqMesh, meshNodes)
                meshNodes = meshNodes.transpose(1, 2, 3, 0)
                meshNodes = meshNodes.reshape((meshDOFs), 3)
                # IMPORTANT: We need to extend the list of points, not append to it
                pointl.extend(meshNodes.tolist())
            except UnboundLocalError:
                raise UnboundLocalError('Something went wrong with the change basis')

            # Add the new element
            cells.setdefault(meshType, []).append(elemNodes.astype(np.uint64))

            # TODO:
            # When merging grids with zoneID = 1, we want them to have separate IDs after the merge
            # zoneName: str = str(max(fnum+1, elem[1]))

            # Add the elem to the cellset
            # > CS1: We create a dictionary of the zones and types that we want
            # cellsets[zoneName][elemType].append(len(cells[elemType]) - 1)

        # Destroy the old dictionary type
        cells.pop(elemType)

        # Add to nElems
        nElems += nIOElems

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

    # Load the subdivided surface mesh
    splitMesh = ReadSurfMesh(SplitElemFile)

    # Apply the actual curving
    mesh      = ApplyRBFCurving(mesh, splitMesh)

    hopout.sep()

    return mesh
