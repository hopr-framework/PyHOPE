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
from contextlib import contextmanager
from contextlib import redirect_stdout, redirect_stderr, ExitStack
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import h5py
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


class MeshContainer:
    """ A simple container to hold the mesh result
    """
    def __init__(self, mesh, nGeo, bcs, elems, sides) -> None:
        self.mesh  = mesh
        self.nGeo  = nGeo
        self.bcs   = bcs
        self.elems = elems
        self.sides = sides


@contextmanager
def Mesh(*args, stdout=False, stderr=True):
    """ Mesh context manager to generate a mesh from a given file

        Args:
            *args: The mesh file path(s) to be processed
            stdout (bool): If False, standard output is suppressed
            stderr (bool): If False, standard error  is suppressed

        Yields:
            Mesh: An object containing the generated mesh and its properties
    """
    # Local imports ----------------------------------------
    import pyhope.config.config as config
    import pyhope.mesh.mesh_vars as mesh_vars
    from pyhope.common.common import DefineCommon, InitCommon
    from pyhope.io.io import DefineIO, InitIO
    from pyhope.mesh.connect.connect import ConnectMesh
    from pyhope.mesh.mesh import DefineMesh, InitMesh, GenerateMesh
    from pyhope.mesh.mesh_sides import GenerateSides
    from pyhope.readintools.readintools import DefineConfig, ReadConfig
    # ------------------------------------------------------

    try:
        # Check if the arguments provided are valid mesh files
        if not args:
            raise ValueError('No mesh file provided.')

        for arg in args:
            # Check if the argument is a valid file path
            if not os.path.isfile(arg):
                raise FileNotFoundError(f'Mesh file not found: {arg}')

            # Check if the argument is a valid HDF5 file
            if not h5py.is_hdf5(arg):
                raise ValueError(f'Mesh file not a valid HDF5 file: {arg}')

        # Suppress output to standard output
        with ExitStack() as stack:
            with open(os.devnull, 'w') as null:
                if not stdout:
                    stack.enter_context(redirect_stdout(null))
                if not stderr:
                    stack.enter_context(redirect_stderr(null))

                # Perform the reduced PyHOPE initialization
                with DefineConfig() as dc:
                    config.prms = dc
                    DefineCommon()
                    DefineIO()
                    DefineMesh()

                with ReadConfig(args[0]) as rc:
                    config.params = rc

                # Read-in required parameters
                InitCommon()
                InitIO()
                InitMesh()

                # Generate the actual mesh
                GenerateMesh()

                # Build our data structures
                GenerateSides()
                ConnectMesh()

        # Export mesh variables
        mesh = mesh_vars.mesh

        nGeo  = mesh_vars.nGeo
        bcs   = mesh_vars.bcs

        elems = mesh_vars.elems
        sides = mesh_vars.sides

        # yield {
        #     'mesh':  mesh,
        #     'nGeo':  nGeo,
        #     'bcs':   bcs,
        #     'elems': elems,
        #     'sides': sides,
        # }

        yield MeshContainer(mesh, nGeo, bcs, elems, sides)

    finally:
        # Cleanup resources after exiting the context
        mesh_vars.mesh  = None
        mesh_vars.nGeo  = None
        mesh_vars.bcs   = None
        mesh_vars.elems = None
        mesh_vars.sides = None
