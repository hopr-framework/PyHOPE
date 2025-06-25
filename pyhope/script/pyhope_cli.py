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
import multiprocessing
import os
import subprocess
import sys
import time
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def main() -> None:
    """ Main routine of PyHOPE
    """
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.output.output as hopout
    from pyhope.common.common import DefineCommon, InitCommon
    from pyhope.common.common_vars import Common
    from pyhope.basis.basis_connect import CheckConnect
    from pyhope.basis.basis_jacobian import CheckJacobians
    from pyhope.basis.basis_watertight import CheckWatertight
    from pyhope.io.io import IO, DefineIO, InitIO
    from pyhope.mesh.connect.connect import ConnectMesh
    from pyhope.mesh.mesh import DefineMesh, InitMesh, GenerateMesh
    from pyhope.mesh.mesh_duplicates import EliminateDuplicates
    from pyhope.mesh.mesh_orient import OrientMesh
    from pyhope.mesh.mesh_sides import GenerateSides
    from pyhope.mesh.mesh_sort import SortMesh
    from pyhope.mesh.fem.fem import FEMConnect
    from pyhope.mesh.transform.mesh_transform import TransformMesh
    from pyhope.readintools.commandline import CommandLine
    from pyhope.readintools.readintools import DefineConfig, ReadConfig
    # ------------------------------------------------------

    # Always spawn with "fork" method to inherit the address space of the parent process
    try:
        multiprocessing.set_start_method('fork')
    # Safely set the multiprocessing start method
    except RuntimeError as e:
        if 'context has already been set' not in str(e):
            raise

    tStart  = time.time()
    process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'],
                               shell=False,
                               cwd=os.path.dirname(os.path.realpath(__file__)),
                               stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)

    common  = Common()
    program = common.program
    version = common.version
    commit = process.communicate()[0].strip().decode('ascii')
    if process.returncode != 0:
        commit = None

    with DefineConfig() as dc:
        config.prms = dc
        DefineCommon()
        DefineIO()
        DefineMesh()

    # Parse the command line arguments
    with CommandLine(sys.argv, program, version, commit) as command:
        args = command[0]
        argv = command[1]

    # Exit with version if requested
    if args.version:
        print(f'{program} version {version}' + (f' [commit {commit}]' if commit else ''))
        sys.exit(0)

    # Check if there are unrecognized arguments
    if len(argv) >= 1:
        print('{} expects exactly one parameter or HDF5-mesh file! Exiting ...'
              .format(program))
        sys.exit()

    with ReadConfig(args.input) as rc:
        config.params = rc

    # Print banner
    hopout.header(program, version, commit)

    # Read-in required parameters
    InitCommon()
    InitIO()
    InitMesh()

    # Generate the actual mesh
    GenerateMesh()

    # Optimize the Gmsh mesh
    hopout.routine('BUILD DATA STRUCTURE...')
    hopout.sep()

    EliminateDuplicates()
    OrientMesh()

    # Build our data structures
    GenerateSides()
    SortMesh()
    ConnectMesh()
    TransformMesh()

    # Generate edge/vertex connectivity
    FEMConnect()

    # Perform the mesh checks
    CheckConnect()
    CheckWatertight()
    CheckJacobians()

    # Output the mesh
    IO()

    tEnd = time.time()
    hopout.end(program, tEnd - tStart)


if __name__ == '__main__':
    main()
