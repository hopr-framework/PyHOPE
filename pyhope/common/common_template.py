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
import importlib.util
import os
import sys
from typing import Optional
from types import ModuleType
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
import pyhope.mesh.mesh_vars as mesh_vars
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# Instantiate ELEMTYPE
elemTypeClass = mesh_vars.ELEMTYPE()
# ==================================================================================================================================


def LoadTemplate(template: str,
                 origin:   str,
                 reason:   str):
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.config.config import prmfile
    # ------------------------------------------------------
    # Define locations of the template files
    # > Priority: prmfile folder > CWD > templates
    templateLocations = [
        os.path.join(os.path.dirname(prmfile), f'{template}.py'),              # Search folder of parameter file
        os.path.join(os.getcwd(), f'{template}.py'),                           # Search in CWD
        os.path.join(os.path.dirname(origin), 'templates', f'{template}.py')   # Search in 'templates'
    ]

    # Check if the template file exists
    templateModule: Optional[ModuleType] = None
    for loc in templateLocations:
        if os.path.exists(loc):
            spec = importlib.util.spec_from_file_location(template, loc)
            # Skip to the next location if spec is None
            if spec is None:
                continue

            templateModule = importlib.util.module_from_spec(spec)
            sys.modules[template] = templateModule
            spec.loader.exec_module(templateModule)

            # Output filename of template
            hopout.routine(f'     found: {loc}')

            # Stop once the module is successfully loaded
            break

    # If the template file is not found, exit
    if templateModule is None:
        hopout.warning(f'{reason} template "{template}" not found!')
        # Print all available default templates for post-deformation
        templist = [f'  {file[:-3]}' for file in os.listdir(os.path.join(os.path.dirname(origin), 'templates')) if file.endswith('.py')]
        hopout.error('Available default extrusion templates:' + ','.join(templist))
    return templateModule
