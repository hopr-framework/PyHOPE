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
import os
import sys
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import gmsh
import h5py
import pygmsh
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def MeshCGNS():
    # Local imports ----------------------------------------
    import src.mesh.mesh_vars as mesh_vars
    import src.output.output as hopout
    from src.common.common_vars import Common
    from.src.io.io_cgns import ElemTypes
    from src.io.io_vars import debugvisu
    from src.readintools.readintools import CountOption, GetIntArray, GetRealArray, GetStr
    # ------------------------------------------------------

    gmsh.initialize()
    if not debugvisu:
        # Hide the GMSH debug output
        gmsh.option.setNumber('General.Terminal', 0)

    hopout.sep()

    # Set default value
    nBCs_CGNS = 0

    hopout.sep()
    hopout.routine('Setting boundary conditions')
    hopout.sep()
    nBCs = CountOption('BoundaryName')
    mesh_vars.bcs = [dict() for _ in range(nBCs)]
    bcs = mesh_vars.bcs

    # Check if the number of BCs matches
    # if nBCs_CGNS is not nBCs:
    #     hopout.warning(    'Different number of boundary conditions between CGNS and parameter file\n'
    #                    ' !! Possibly see upstream issue, https://gitlab.onelab.info/gmsh/gmsh/-/issues/2727\n'
    #                    ' !! {} is now exiting ...'.format(Common.program))
    #     sys.exit()

    for iBC, bc in enumerate(bcs):
        bcs[iBC]['Name'] = GetStr('BoundaryName', number=iBC)
        bcs[iBC]['BCID'] = iBC + 1
        bcs[iBC]['Type'] = GetIntArray('BoundaryType', number=iBC)

    nVVs = CountOption('vv')
    mesh_vars.vvs = [dict() for _ in range(nVVs)]
    vvs = mesh_vars.vvs
    for iVV, vv in enumerate(vvs):
        vvs[iVV] = dict()
        vvs[iVV]['Dir'] = GetRealArray('vv', number=iVV)
    if len(vvs) > 0:
        hopout.sep()

    gmsh_needs_bcs = False

    fnames = CountOption('filename')
    for iName in range(fnames):
        fname = GetStr('filename')
        fname = os.path.join(os.getcwd(), fname)

        # get file extension
        _, ext = os.path.splitext(fname)

        # if not GMSH format convert
        if ext != '.msh':
            # Setup GMSH to import required data
            # gmsh.option.setNumber('Mesh.SaveAll', 1)
            gmsh.option.setNumber('Mesh.RecombineAll', 1)
            gmsh.option.setNumber('Mesh.CgnsImportIgnoreBC', 0)
            gmsh.option.setNumber('Mesh.CgnsImportIgnoreSolution', 1)

            gmsh.merge(fname)

        # read in boundary conditions from cgns file as gmsh is not capable of reading vertex based boundaries
        # TODO: ACTUALLY NOT NEEDED FOR ANSA CGNS
        # if ext == '.cgns':
        #     with h5py.File(fname, mode='r') as f:
        #         domain = f['Base']
        #         for iZone, zone in enumerate(domain.keys()):
        #             # skip the data zone
        #             if zone.strip() == 'data':
        #                 continue

        gmsh.model.geo.synchronize()
        # gmsh.model.occ.synchronize()

        entities  = gmsh.model.getEntities()
        nBCs_CGNS = len([s for s in entities if s[0] == 2])

        # Check if GMSH read all BCs
        # > This will only work if the CGNS file identifies elementary entities by CGNS "families" and by "BC" structures
        # > Possibly see upstream issue, https://gitlab.onelab.info/gmsh/gmsh/-/issues/2727\n'
        if nBCs_CGNS is nBCs:
            for entDim, entTag in entities:
                # Surfaces are dim-1
                if entDim == 3:
                    continue

                entName = gmsh.model.get_entity_name(dim=entDim, tag=entTag)
                gmsh.model.addPhysicalGroup(entDim, [entTag], name=entName)
        else:
            gmsh_needs_bcs = True

        gmsh.model.geo.synchronize()

    # PyGMSH returns a meshio.mesh datatype
    mesh = pygmsh.geo.Geometry().generate_mesh(order=mesh_vars.nGeo)

    if debugvisu:
        gmsh.fltk.run()

    # Finally done with GMSH, finalize
    gmsh.finalize()

    return mesh
