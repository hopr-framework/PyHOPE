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
# import sys
from typing import cast
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
    from pyhope.config.config import prmfile
    from pyhope.mesh.mesh_vars import BC
    from pyhope.mesh.reader.reader_gmsh import compatibleGMSH, ReadGMSH, BCCGNS
    from pyhope.mesh.reader.reader_gambit import ReadGambit
    from pyhope.mesh.reader.reader_hopr import ReadHOPR
    from pyhope.readintools.readintools import CountOption, GetIntArray, GetRealArray, GetStr
    # ------------------------------------------------------

    hopout.sep()
    hopout.info('LOADING EXTERNAL MESH...')

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
        bc.name = GetStr(     'BoundaryName', number=iBC).lower()    # noqa: E251
        bc.bcid = iBC + 1                                            # noqa: E251
        bc.type = GetIntArray('BoundaryType', number=iBC)            # noqa: E251

    nVVs = CountOption('vv')
    mesh_vars.vvs = [dict() for _ in range(nVVs)]
    vvs = mesh_vars.vvs
    if len(vvs) > 0:
        hopout.sep()
    for iVV, _ in enumerate(vvs):
        vvs[iVV] = dict()
        vvs[iVV]['Dir'] = GetRealArray('vv', number=iVV)

    # Load the mesh(es)
    mesh   = meshio.Mesh(np.array(()), dict())
    fnames = [GetStr('Filename', number=i) for i in range(CountOption('Filename'))]

    # Check whether mesh file exists in the current directory or in the same directory
    for iFile, fname in enumerate(fnames):
        if os.path.isfile(os.path.abspath(fname)):
            fnames[iFile] = os.path.abspath(fname)
        elif os.path.isfile(os.path.join(os.path.dirname(prmfile), fname)):
            fnames[iFile] = os.path.abspath(os.path.join(os.path.dirname(prmfile), fname))
            print(hopout.warn('Mesh not found in the CWD, but found in the prmfile directory.'))
        else:
            hopout.error('Mesh file [󰇘]/{} does not exist'.format(os.path.basename(fname)))

    if not all(compatibleGMSH(fname) for fname in fnames):
        if any(compatibleGMSH(fname) for fname in fnames):
            hopout.warning('Mixed file formats detected, this is untested and may not work')
            # sys.exit(1)

    # Gmsh has to come first as we cannot extend the mesh
    fgmsh = [s for s in fnames if compatibleGMSH(s)]
    if len(fgmsh) > 0:
        mesh = ReadGMSH(fgmsh)
    fnames = list(filter(lambda x: not compatibleGMSH(x), fnames))

    # Gambit meshes can extend the Gmsh mesh
    fgambit = [s for s in fnames if s.endswith('.neu')]
    if len(fgambit) > 0:
        mesh = ReadGambit(fgambit, mesh)
    fnames = list(filter(lambda x: x not in fgambit, fnames))

    # HOPR meshes can extend the Gmsh mesh
    fhopr  = [s for s in fnames if s.endswith('.h5')]
    if len(fhopr) > 0:
        mesh = ReadHOPR(fhopr, mesh)
    fnames = list(filter(lambda x: x not in fhopr, fnames))

    # If there are still files left, we have an unknown format
    if len(fnames) > 0:
        hopout.error('Unknown file format {}, exiting...'.format(fnames))

    # Regenerate the boundary conditions
    if mesh_vars.CGNS.regenerate_BCs:
        mesh = BCCGNS(mesh, fgmsh)

    # Check if mesh has any boundary conditions
    if len(bcs) == 0:
        hopout.error('No boundary conditions defined in the parameter file.')

    # Reconstruct periodicity vectors from mesh
    hasPeriodic = np.any([cast(np.ndarray, bcs[s].type)[0] == 1 for s in range(nBCs)])
    if len(mesh_vars.vvs) == 0 and hasPeriodic:
        print(hopout.warn('Periodicity vectors neither defined in parameter file nor '
                          'in the given mesh file. Reconstructing the vectors from BCs!'))
        # Get max number of periodic alphas
        mesh_vars.vvs = [dict() for _ in range(int(np.max([np.abs(cast(np.ndarray, bc.type)[3]) for bc in bcs])))]
        vvs = recontruct_periodicity(mesh)
        hopout.routine('The following vectors were recovered:')
        for iVV, vv in enumerate(vvs):
            hopout.printoption('vv[{}]'.format(iVV+1), '{0:}'.format(np.round(vv['Dir'], 6)), 'RECOVER')
        hopout.sep()

    hopout.info('LOADING EXTERNAL MESH DONE!')
    hopout.sep()

    return mesh


def recontruct_periodicity(mesh: meshio.Mesh) -> list:
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    bcs = mesh_vars.bcs
    vvs = mesh_vars.vvs

    for iVV, vv in enumerate(vvs):

        # Identify positive and negative periodic boundaries
        boundaries = {1: None, -1: None}
        for bc in [s for s in bcs if abs(s.type[3]) == iVV + 1]:
            sign = np.sign(bc.type[3])
            if boundaries[sign] is not None:
                hopout.error("Multiple periodic boundaries found for the same direction. Exiting...")
            boundaries[sign] = bc.name

        # Compute mean coordinates for both boundaries as a tuple
        mean_coords = tuple(
            np.mean(mesh.points[
                np.array(sorted({
                    node for iBlock, _ in enumerate(mesh.cells)
                    if (mesh.cell_sets[bc] and mesh.cell_sets[bc][iBlock] is not None)
                    for node in mesh.cells[iBlock].data[mesh.cell_sets[bc][iBlock]].flatten()
                }))
            ], axis=0) if bc else None
            for bc in (boundaries[1], boundaries[-1])
        )

        # Store the periodicity vector if both mean coordinates exist
        if mean_coords[0] is not None and mean_coords[1] is not None:
            vv.update({"Dir": mean_coords[1] - mean_coords[0]})
        else:
            vv.update({"Dir": np.array([0.0, 0.0, 0.0])})

    return vvs
