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
from typing import Optional, Final, final
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
from alive_progress import alive_bar
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


# Number of elements to display in the progress bar
barElems: Final[int] = int(1.E5)


@final
class ProgressBar:
    """ Provide a progress bar outside of the context manager
    """
    def __init__(self, title: Optional[str], value: int, length: int = 33, threshold: int = barElems) -> None:
        # Local imports ----------------------------------------
        from pyhope.common.common import IsInteractive
        # ------------------------------------------------------
        if value <= threshold or not IsInteractive():
            self.bar = None
            return None

        self._cm   : Final         = alive_bar(title=title, total=value, length=length)
        self._title: Optional[str] = title
        self._len  : int           = length

        # Initialize the progress bar
        self.bar = self._cm.__enter__()

    def step(self, steps: int = 1) -> None:
        if self.bar is not None:
            self.bar(steps)

    def title(self, title: str) -> None:
        if self.bar is not None:
            self.bar.title(title)

    def close(self) -> None:
        if self.bar is not None:
            _ = self._cm.__exit__(None, None, None)
