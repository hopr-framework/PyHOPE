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
import argparse
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def Check(args: argparse.Namespace
        ) -> None:
    """ Check routine of PyHOPE
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    from pyhope.check.check_health import CheckHealth
    from pyhope.check.check_install import CheckInstall
    from pyhope.common.common_vars import Common
    from pyhope.gmsh.gmsh_install import PkgsCheckGmsh
    # ------------------------------------------------------

    # Print banner
    common  = Common()
    hopout.header(common.program, common.version, common.commit)

    # Check if we are using the NRG Gmsh version and install it if not
    PkgsCheckGmsh()

    # Attempt to extract the directory
    directory = None
    if   isinstance(args.verify        , str):  # noqa: E271
        directory = args.verify
    elif isinstance(args.verify_install, str):
        directory = args.verify_install

    # Return all checks and return
    if args.verify:
        CheckHealth()
        hopout.info('')
        CheckInstall(directory)

    # Run only health check
    elif args.verify_health:
        CheckHealth()

    # Run only install check
    elif args.verify_install:
        CheckInstall(directory)
