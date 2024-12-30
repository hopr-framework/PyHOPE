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
import sys
import traceback
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def DefineMesh() -> None:
    """ Define general options for mesh generation / readin
    """
    # Local imports ----------------------------------------
    from pyhope.readintools.readintools import CreateInt, CreateIntArray, CreateRealArray, CreateSection, CreateStr, CreateLogical
    from pyhope.readintools.readintools import CreateIntFromString, CreateIntOption
    from pyhope.mesh.mesh_vars import ELEMTYPE
    # ------------------------------------------------------

    CreateSection('Mesh')
    CreateInt(      'Mode',                               help='Mesh generation mode (1 - Internal, 2 - External [MeshIO])')
    # Internal mesh generator
    CreateInt(      'nZones',                             help='Number of mesh zones')
    CreateRealArray('Corner',         24,  multiple=True, help='Corner node positions: (/ x_1,y_1,z_1,, x_2,y_2,z_2,, ' +
                                                                                        '... ,, x_8,y_8,z_8/)')  # noqa: E127
    CreateIntArray( 'nElems',          3,  multiple=True, help='Number of elements in each direction')
    CreateIntFromString('ElemType'      ,  multiple=True, default='hexahedron', help='Element type')
    for key, val in ELEMTYPE.name.items():
        CreateIntOption('ElemType', number=val, name=key)
    CreateStr(      'BoundaryName',        multiple=True, help='Name of domain boundary')
    CreateIntArray( 'BoundaryType',    4,  multiple=True, help='(/ Type, curveIndex, State, alpha /)')
    CreateIntArray( 'BCIndex',         6,  multiple=True, help='Index of BC for each boundary face')
    # Gmsh
    CreateLogical(  'EliminateNearDuplicates', default=True, help='Enables elimination of near duplicate points')
    # Periodicity
    CreateRealArray('vv',              3,  multiple=True, help='Vector for periodic BC')
    CreateLogical(  'doPeriodicCorrect',   default=True,  help='Enables periodic correction')
    # External mesh readin through GMSH
    CreateStr(      'Filename',            multiple=True, help='Name of external mesh file')
    CreateLogical(  'MeshIsAlreadyCurved', default=False, help='Enables mesh agglomeration')
    # Common settings
    CreateInt(      'NGeo'         ,       default=1,     help='Order of spline-reconstruction for curved surfaces')
    CreateInt(      'BoundaryOrder',       default=2,     help='Order of spline-reconstruction for curved surfaces (legacy)')
    CreateLogical(  'doSortIJK',           default=False, help='Sort the mesh elements along the I,J,K directions')
    CreateLogical(  'CheckElemJacobians',  default=True,  help='Check the Jacobian and scaled Jacobian for each element')
    CreateLogical(  'CheckWatertightness', default=True,  help='Check if the mesh is watertight')
    CreateLogical(  'CheckSurfaceNormals', default=True,  help='Check if the surface normals point outwards')
    # Mortars
    CreateLogical(  'doMortars',           default=True,  help='Enables mortars')


def InitMesh() -> None:
    """ Readin general option for mesh generation / readin
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import GetInt, CountOption
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('INIT MESH...')

    mesh_vars.mode = GetInt('Mode')

    NGeo     = GetInt('NGeo')          if CountOption('NGeo')          else None  # noqa: E272
    BCOrder  = GetInt('BoundaryOrder') if CountOption('BoundaryOrder') else None  # noqa: E272

    if not NGeo and not BCOrder:
        mesh_vars.nGeo = 1
    elif NGeo and BCOrder and NGeo != BCOrder - 1:
        hopout.warning('NGeo / BoundaryOrder must be equal to NGeo + 1!')
        sys.exit(1)
    else:
        if NGeo is not None:
            mesh_vars.nGeo = NGeo
        elif BCOrder is not None:
            mesh_vars.nGeo = BCOrder - 1

        if mesh_vars.nGeo < 1:
            hopout.warning('Effective boundary order < 1. Try increasing the NGeo / BoundaryOrder parameter!')
            sys.exit(1)

    hopout.info('INIT MESH DONE!')


def GenerateMesh() -> None:
    """ Generate the mesh
        Mode 1 - Use internal mesh generator
        Mode 2 - Readin external mesh through GMSH
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.mesh_builtin import MeshCartesian
    from pyhope.mesh.mesh_external import MeshExternal
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('GENERATE MESH...')

    match mesh_vars.mode:
        case 1:  # Internal Cartesian Mesh
            mesh = MeshCartesian()
        case 3:  # External mesh
            mesh = MeshExternal()
        case _:  # Default
            hopout.warning('Unknown mesh mode {}, exiting...'.format(mesh_vars.mode))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

    mesh_vars.mesh = mesh

    # Final count
    nElems = 0
    for cellType in mesh.cells:
        if any(s in cellType.type for s in mesh_vars.ELEMTYPE.type.keys()):
            nElems += mesh.get_cells_type(cellType.type).shape[0]
    hopout.sep()
    hopout.routine('Generated mesh with {} cells'.format(nElems))
    hopout.sep()

    hopout.info('GENERATE MESH DONE!')


def RegenerateMesh() -> None:
    """ Finish missing mesh information such as BCs
    """
    # Local imports ----------------------------------------
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.mesh.reader.reader_gmsh import BCCGNS
    # ------------------------------------------------------

    match mesh_vars.mode:
        case 1:  # Internal Cartesian Mesh
            mesh = mesh_vars.mesh
        case 3:  # External CGNS mesh
            if mesh_vars.CGNS.regenerate_BCs:
                hopout.separator()
                hopout.info('REGENERATE MESH...')
                mesh = BCCGNS()
                hopout.info('REGENERATE MESH DONE!')
            else:
                mesh = mesh_vars.mesh
        case _:  # Default
            hopout.warning('Unknown mesh mode {}, exiting...'.format(mesh_vars.mode))
            traceback.print_stack(file=sys.stdout)
            sys.exit(1)

    mesh_vars.mesh = mesh
