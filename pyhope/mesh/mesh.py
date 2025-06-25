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
    from pyhope.readintools.readintools import CreateInt, CreateIntArray, CreateRealArray, CreateSection, CreateStr
    from pyhope.readintools.readintools import CreateLogical, CreateReal
    from pyhope.readintools.readintools import CreateIntFromString, CreateIntOption
    from pyhope.mesh.mesh_vars import ELEMTYPE, MeshMode
    # ------------------------------------------------------

    CreateSection('Mesh')
    CreateIntFromString('Mode',                            help='Mesh generation mode (1 - Internal, 3 - External [MeshIO])')
    CreateIntOption(    'Mode', number=MeshMode.MODE_INT,  name='Internal')
    CreateIntOption(    'Mode', number=MeshMode.MODE_EXT,  name='External')
    # Internal mesh generator
    CreateInt(      'nZones',                              help='Number of mesh zones')
    CreateRealArray('Corner',         24,   multiple=True, help='Corner node positions: (/ x_1,y_1,z_1,, x_2,y_2,z_2,, ' +
                                                                                         '... ,, x_8,y_8,z_8/)')
    CreateRealArray('X0',              3,   multiple=True, help='Origin of a zone. Equivalent to a corner node.')
    CreateRealArray('DX',              3,   multiple=True, help='Extension of the zone in each spatial direction ' +
                                                                 'starting from the origin X0 corner node')
    CreateIntArray( 'nElems',          3,   multiple=True, help='Number of elements in each direction')
    CreateIntFromString('ElemType'      ,   multiple=True, default='hexahedron', help='Element type')
    for key, val in ELEMTYPE.name.items():
        # Only consider uncurved element types
        if val > 200:
            continue
        CreateIntOption('ElemType', number=val, name=key)
    # Gmsh
    CreateLogical(  'EliminateNearDuplicates', default=True, help='Enables elimination of near duplicate points')
    # External mesh readin through GMSH
    CreateStr(      'Filename',             multiple=True, help='Name of external mesh file')
    CreateLogical(  'MeshIsAlreadyCurved',  default=False, help='Enables mesh agglomeration')
    # Common settings
    CreateInt(      'NGeo'         ,        default=1,     help='Order of spline-reconstruction for curved surfaces')
    CreateInt(      'BoundaryOrder',        default=2,     help='Order of spline-reconstruction for curved surfaces (legacy)')
    # Periodicity
    CreateRealArray('vv',              3,   multiple=True, help='Vector for periodic BC')
    CreateLogical(  'doPeriodicCorrect',    default=False, help='Enables periodic correction')
    # Connections
    CreateLogical(  'doSortIJK',            default=False, help='Sort the mesh elements along the I,J,K directions')
    CreateLogical(  'doSplitToHex',         default=False, help='Split simplex elements into hexahedral elements')
    # Mortars
    CreateLogical(  'doMortars',            default=True,  help='Enables mortars')
    # Boundaries
    CreateSection('Boundaries')
    CreateStr(      'BoundaryName',         multiple=True, help='Name of domain boundary')
    CreateIntArray( 'BoundaryType',    4,   multiple=True, help='(/ Type, curveIndex, State, alpha /)')
    CreateIntArray( 'BCIndex',         6,   multiple=True, help='Index of BC for each boundary face')
    # Checking
    CreateSection('Mesh Checks')
    CreateLogical(  'CheckElemJacobians',   default=True,  help='Check the Jacobian and scaled Jacobian for each element')
    CreateLogical(  'CheckConnectivity'  ,  default=True,  help='Check if the side connectivity, including correct flip')
    CreateLogical(  'CheckWatertightness',  default=True,  help='Check if the mesh is watertight')
    CreateLogical(  'CheckSurfaceNormals',  default=True,  help='Check if the surface normals point outwards')
    # Transformation
    CreateSection('Transformation')
    CreateReal(      'meshScale',           default=1.0,                              help='Scale the mesh')
    CreateRealArray( 'meshTrans', nReals=3, default='(/0.,0.,0./)',                   help='Translate the mesh')
    CreateRealArray( 'meshRot',   nReals=9, default='(/1.,0.,0.,0.,1.,0.,0.,0.,1./)', help='Rotate the mesh around rotation center')
    CreateRealArray( 'meshRotCenter', nReals=3, default='(/0.,0.,0./)',               help='Rotate the mesh around rotation center')
    CreateStr(       'MeshPostDeform',   default='none',                              help='Mesh post-transformation template')
    # Stretching
    CreateSection('Stretching')
    CreateIntArray( 'StretchType',      3,   default='(/0,0,0/)', multiple=True,      help='Stretching type for individual '
                                                                                             'zone per spatial direction.')
    CreateRealArray( 'Factor',          3,   multiple=True, help='Stretching factor of zone for geometric stretching for '
                                                                                                 'each spatial direction.')
    CreateRealArray( 'l0',              3,   multiple=True, help='Smallest desired element in zone per spatial direction.')
    CreateRealArray( 'DXmaxToDXmin',    3,   multiple=True, help='Ratio between the smallest and largest element per spatial '
                                                                                                               'direction')
    # Edge connectivity
    CreateSection('Finite Element Method (FEM) Connectivity')
    CreateLogical(  'doFEMConnect',         default=False, help='Generate finite element method (FEM) connectivity')


def InitMesh() -> None:
    """ Readin general option for mesh generation / readin
    """
    # Local imports ----------------------------------------
    import pyhope.io.io_vars as io_vars
    import pyhope.mesh.mesh_vars as mesh_vars
    import pyhope.output.output as hopout
    from pyhope.readintools.readintools import GetInt, GetIntFromStr, CountOption
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('INIT MESH...')

    mesh_vars.mode = GetIntFromStr('Mode')

    NGeo     = GetInt('NGeo')          if CountOption('NGeo')          else None  # noqa: E272
    BCOrder  = GetInt('BoundaryOrder') if CountOption('BoundaryOrder') else None  # noqa: E272

    if not NGeo and not BCOrder:
        mesh_vars.nGeo = 1
    elif NGeo and BCOrder and NGeo != BCOrder - 1:
        hopout.error('NGeo / BoundaryOrder must be equal to NGeo + 1!')
    else:
        if NGeo is not None:
            mesh_vars.nGeo = NGeo
        elif BCOrder is not None:
            mesh_vars.nGeo = BCOrder - 1

        if mesh_vars.nGeo < 1:
            hopout.error('Effective boundary order < 1. Try increasing the NGeo / BoundaryOrder parameter!')

    # Check if the requested output format can supported the requested polynomial order
    match io_vars.outputformat:
        case io_vars.MeshFormat.FORMAT_VTK:
            if mesh_vars.nGeo > 2:
                hopout.error('Output format VTK does not support polynomial order > 2!')

    # hopout.info('INIT MESH DONE!')


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
    from pyhope.mesh.mesh_vars import MeshMode
    from pyhope.mesh.topology.mesh_splittohex import MeshSplitToHex
    from pyhope.mesh.topology.mesh_topology import MeshChangeElemType
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('GENERATE MESH...')

    match mesh_vars.mode:
        case MeshMode.MODE_INT:  # Internal Cartesian Mesh
            mesh = MeshCartesian()
        case MeshMode.MODE_EXT:  # External mesh
            mesh = MeshExternal()
        case _:  # Default
            hopout.error('Unknown mesh mode {}, exiting...'.format(mesh_vars.mode), traceback=True)

    # Split hexahedral elements if requested
    mesh = MeshChangeElemType(mesh)
    # Split simplex elements if requested
    mesh = MeshSplitToHex(mesh)
    mesh_vars.mesh = mesh

    # Final count
    nElems = 0
    for cellType in mesh.cells:
        if any(s in cellType.type for s in mesh_vars.ELEMTYPE.type.keys()):
            nElems += mesh.get_cells_type(cellType.type).shape[0]

    hopout.routine('Generated mesh with {} cells'.format(nElems))
    # hopout.sep()
    # hopout.info('GENERATE MESH DONE!')
    hopout.separator()
