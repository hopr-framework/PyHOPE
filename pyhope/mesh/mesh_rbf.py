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


def MeshCurveRBF(mesh: meshio.Mesh) -> meshio.Mesh:
    """ Curve a linear mesh during postprocessing
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.readintools.readintools import CountOption, GetStr
    # ------------------------------------------------------

    # Linear meshes need not be curved
    if mesh_vars.nGeo <= 1:
        return mesh

    if CountOption('SplitElemFile') <= 0:
        return mesh

    SplitElemFile = GetStr('SplitElemFile')
    # Leave if no curving is required
    if SplitElemFile is None:
        return mesh

    splitMesh = ReadSurfMesh(SplitElemFile)
    splitMesh = splitMesh

    # TODO: STILL NEED THE ACTUAL MESH CURVING
    # TODO: STILL NEED TO MANUALLY CHANGE THE MESH ORDER

    return mesh
