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
# import os
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
    """ Main routine of UVWXYZ
    """
    # Local imports ----------------------------------------
    import src.config.config as config
    import src.output.output as hopout
    from src.common.common import DefineCommon, InitCommon
    from src.common.common_vars import Common
    from src.basis.basis_jacobian import CheckJacobians
    from src.io.io import IO, DefineIO, InitIO
    from src.mesh.mesh import DefineMesh, InitMesh, GenerateMesh, RegenerateMesh
    from src.mesh.mesh_connect import ConnectMesh
    from src.mesh.mesh_sides import GenerateSides
    from src.mesh.mesh_sort import SortMesh
    from src.readintools.commandline import CommandLine
    from src.readintools.readintools import DefineConfig, ReadConfig
    # ------------------------------------------------------

    tStart  = time.time()
    process = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], shell=False, stdout=subprocess.PIPE)

    program = Common.program
    version = Common.version
    commit  = process.communicate()[0].strip().decode('ascii')

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
        print('{} version {} [commit {}]'.format(program, version, commit))
        sys.exit(0)

    # Check if there are unrecognized arguments
    if len(argv) >= 1:
        print('{} expects exactly one parameter file! Exiting ...'
              .format(program))
        sys.exit()

    with ReadConfig(args.parameter) as rc:
        config.params = rc

    # Print banner
    hopout.header(program, version, commit)

    # Read-in required parameters
    InitCommon()
    InitIO()
    InitMesh()

    # Generate the actual mesh
    GenerateMesh()
    SortMesh()
    GenerateSides()
    RegenerateMesh()
    ConnectMesh()

    # Check the mapping
    CheckJacobians()

    # Output the mesh
    IO()

    tEnd = time.time()
    hopout.end(tEnd - tStart)


if __name__ == '__main__':
    main()
