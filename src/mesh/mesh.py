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


def DefineMesh():
    """ Define general options for mesh generation / readin
    """
    # Local imports ----------------------------------------
    from src.readintools.readintools import CreateInt, CreateIntArray, CreateRealArray, CreateSection, CreateStr
    # ------------------------------------------------------

    CreateSection('Mesh')
    CreateInt(      'Mode',                           help='Mesh generation mode (1 - Internal, 2 - External [MeshIO])')
    CreateInt(      'BoundaryOrder',   default=2,     help='Order of spline-reconstruction for curved surfaces')
    CreateInt(      'nZones',                         help='Number of mesh zones')
    CreateRealArray('Corner',      24, multiple=True, help='Corner node positions: (/ x_1,y_1,z_1,, x_2,y_2,z_2,, ... ,, x_8,y_8,z_8/)')
    CreateIntArray( 'nElems',       3, multiple=True, help='Number of elements in each direction')
    CreateStr(      'BoundaryName',    multiple=True, help='Name of domain boundary')
    CreateIntArray( 'BoundaryType', 4, multiple=True, help='(/ Type, curveIndex, State, alpha /)')
    CreateIntArray( 'BCIndex',      6, multiple=True, help='Index of BC for each boundary face')
    CreateRealArray('vv',           3, multiple=True, help='Vector for periodic BC')
    CreateStr(      'filename',        multiple=True, help='Name of external mesh file')


def InitMesh():
    """ Readin general option for mesh generation / readin
    """
    # Local imports ----------------------------------------
    import src.mesh.mesh_vars as mesh_vars
    import src.output.output as hopout
    from src.readintools.readintools import GetInt
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('INIT MESH...')

    mesh_vars.mode = GetInt('Mode')
    mesh_vars.nGeo = GetInt('BoundaryOrder') - 1
    if mesh_vars.nGeo < 1:
        hopout.warning('Effective boundary order < 1. Try increasing the BoundaryOrder parameter!')
        sys.exit()

    hopout.info('INIT MESH DONE!')


def GenerateMesh():
    """ Generate the mesh
        Mode 1 - Use internal mesh generation
        Mode 2 - Readin external mesh through MeshIO
    """
    # Local imports ----------------------------------------
    import src.mesh.mesh_vars as mesh_vars
    import src.output.output as hopout
    from src.mesh.mesh_builtin import MeshCartesian
    from src.mesh.mesh_external import MeshCGNS
    # ------------------------------------------------------

    hopout.separator()
    hopout.info('GENERATE MESH...')

    match mesh_vars.mode:
        case 1:  # Internal Cartesian Mesh
            mesh = MeshCartesian()
        case 3:  # External CGNS mesh
            mesh = MeshCGNS()
        case _:  # Default
            hopout.warning('Unknown mesh mode {}, exiting...'.format(mesh_vars.mode))
            traceback.print_stack(file=sys.stdout)
            sys.exit()

    mesh_vars.mesh = mesh

    hopout.info('GENERATE MESH DONE!')
