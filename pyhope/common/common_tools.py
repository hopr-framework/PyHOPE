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
import time
# from sortedcontainers import SortedDict
from collections import defaultdict
from typing import Final, Tuple
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def time_function(func, *args, **kwargs) -> float:
    """ A helper function to measure the execution time of an arbitrary function.

    Parameters:
    func (callable): The function to be timed.
    *args: Positional arguments to pass to the function.
    **kwargs: Keyword arguments to pass to the function.

    Returns:
    The return value of the function being timed.
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------
    tStart: Final[float] = time.time()
    result               = func(*args, **kwargs)
    tEnd:   Final[float] = time.time()
    tFunc:  Final[float] = tEnd - tStart
    hopout.info(  hopout.Colors.BANNERA + f'Function {func.__name__} required {tFunc:.6f} seconds to complete.'
                + hopout.Colors.END)

    return result


def allocate_or_resize( dict: dict, key: str, shape: Tuple[int, int]) -> Tuple[dict, int]:
    """ Allocate or resize a numpy array in a dictionary.
    """
    offset = 0
    if key not in dict:
        dict[key] = np.ndarray(shape, dtype=np.uint)
    else:
        offset = dict[key].shape[0]
        new_len = offset + shape[0]
        dict[key] = np.resize(dict[key],  (new_len, shape[1]))

    return dict, offset


class IndexedLists:
    def __init__(self) -> None:
        # PERF: The unsorted dict is faster for adding and removing elements
        # Create a SortedDict to keep the data sorted by index
        # self.data = SortedDict()

        # Use a plain dict for faster key operations
        self.data = {}
        # Inverse mapping: for each value, store the set of keys (indices) that contain it
        self._inverse = defaultdict(set)

    def add(self, index: int, values) -> None:
        """ Add a sublist at a specific integer index
        """
        value_set = set(values)  # Use set for fast removals
        self.data[index] = value_set
        for v in value_set:
            self._inverse[v].add(index)

    def remove_index(self, indices) -> None:
        """ Remove the sublist at idx and remove the integer idx from all remaining sublists
        """
        if isinstance(indices, int):
            # Convert to a set for fast operations
            indices = {indices}
        else:
            # Convert list to set for O(1) lookups
            indices = set(indices)

        # Create a set to hold all affected keys
        affected_keys = set()
        # Remove keys directly and collect affected keys from inverse mapping
        for idx in indices:
            if idx in self.data:
                # Remove the entire key; also clean up the inverse mapping
                for v in self.data[idx]:
                    self._inverse[v].discard(idx)
                del self.data[idx]
            affected_keys |= self._inverse.pop(idx, set())

        # Now update each affected key only once while checking for emptiness
        for key in affected_keys:
            if key in self.data:
                self.data[key].difference_update(indices)

    def __getitem__(self, index) -> list[int]:
        """ Retrieve a sublist by index
        """
        return list(self.data[index])  # Convert set back to list when accessed

    def __repr__(self) -> str:
        return repr({k: list(v) for k, v in self.data.items()})  # Convert to list for cleaner output
