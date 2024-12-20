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


def MeshExternal() -> meshio.Mesh:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_vars import BC
    from pyhope.mesh.reader.reader_gmsh import compatibleGMSH, ReadGMSH
    from pyhope.mesh.reader.reader_hopr import ReadHOPR
    from pyhope.readintools.readintools import CountOption, GetIntArray, GetRealArray, GetStr
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('LOADING EXTERNAL MESH')

    hopout.sep()
    hopout.routine('Setting boundary conditions')
    hopout.sep()

    # Load the boundary conditions
    nBCs = CountOption('BoundaryName')
    mesh_vars.bcs = [BC() for _ in range(nBCs)]
    bcs = mesh_vars.bcs

    for iBC, bc in enumerate(bcs):
        # bc.update(name = GetStr(     'BoundaryName', number=iBC),  # noqa: E251
        #           bcid = iBC + 1,                                  # noqa: E251
        #           type = GetIntArray('BoundaryType', number=iBC))  # noqa: E251
        bc.name = GetStr(     'BoundaryName', number=iBC)  # noqa: E251
        bc.bcid = iBC + 1                                  # noqa: E251
        bc.type = GetIntArray('BoundaryType', number=iBC)  # noqa: E251

    nVVs = CountOption('vv')
    mesh_vars.vvs = [dict() for _ in range(nVVs)]
    vvs = mesh_vars.vvs
    if len(vvs) > 0:
        hopout.sep()
    for iVV, _ in enumerate(vvs):
        vvs[iVV] = dict()
        vvs[iVV]['Dir'] = GetRealArray('vv', number=iVV)

    # Load the mesh(es)
    mesh   = meshio.Mesh(np.array([]), dict())
    fnames = [GetStr('Filename', number=i) for i in range(CountOption('Filename'))]
    for fname in fnames:
        fname = os.path.join(os.getcwd(), fname)

        # check if the file exists
        if not os.path.isfile(os.path.join(os.getcwd(), fname)):
            hopout.warning('File [󰇘]/{} does not exist'.format(os.path.basename(fname)))
            sys.exit(1)

    if not all(compatibleGMSH(fname) for fname in fnames):
        if any(compatibleGMSH(fname) for fname in fnames):
            hopout.warning('Mixed file formats detected, this is untested and may not work')
            # sys.exit(1)

    # Gmsh has to come first as we cannot extend the mesh
    fgmsh = [s for s in fnames if compatibleGMSH(s)]
    if len(fgmsh) > 0:
        mesh = ReadGMSH(fgmsh)
    fnames = list(filter(lambda x: not compatibleGMSH(x), fnames))

    # HOPR meshes can extend the Gmsh mesh
    fhopr  = [s for s in fnames if s.endswith('.h5')]
    if len(fhopr) > 0:
        mesh = ReadHOPR(fhopr, mesh)
    fnames = list(filter(lambda x: x not in fhopr, fnames))

    # If there are still files left, we have an unknown format
    if len(fnames) > 0:
        hopout.warning('Unknown file format {}, exiting...'.format(fnames))
        sys.exit(1)

    hopout.info('LOADING EXTERNAL MESH DONE!')
    hopout.separator()

    return mesh
