#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2024 Numerics Research Group, University of Stuttgart, Prof. Andrea Beck
# Copyright (c) 2022 Gabriel Altay (Original Version)
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
from typing import List, Literal, Iterable, Union, overload
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import numpy.typing as npt
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def HilbertCurveNumpy() -> None:
    """ Monkey-patch for hilbertcurve.HilbertCurve:
        - Adds a NumPy-vectorized _distances_from_points_numpy(self, points)
        - Wraps distances_from_points to use the vectorized path for ndarray inputs

        Assumptions:
        - points is (M, n) with integer coordinates in [0, 2**p - 1]
        - p * n <= 63 so distances fit into uint64

        The patch is idempotent: calling apply_hilbert_numpy_patch() multiple times is harmless
    """
    # If the package isn't available, silently skip patching
    try:
        from hilbertcurve.hilbertcurve import HilbertCurve
    except Exception:
        return None

    # Avoid re-patching
    if getattr(HilbertCurve, '_numpy_patch_applied', False):
        return None

    # Typing helpers
    @overload
    def _distances_from_points_numpy(self, points: npt.NDArray, match_type: Literal[False] = False) -> List: ...
    @overload
    def _distances_from_points_numpy(self, points: List       , match_type: Literal[True])          -> List: ...
    @overload
    def _distances_from_points_numpy(self, points: npt.NDArray, match_type: Literal[True])          -> npt.NDArray: ...
    # Function
    def _distances_from_points_numpy(self,
                                     points    : Union[List, np.ndarray],
                                     match_type: bool = False) -> Union[np.ndarray, list]:
        """ Batch implementation for distances_from_points in numpy
        """
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] != self.n:
            raise ValueError(f'points must be (M, {self.n})')

        # Work in uint64 for per-coordinate logic (coordinates fit in 64-bit)
        # > Copy to not mutate caller memory
        upts = pts.astype(np.uint64, copy=True)

        # Range checks
        max_x = np.uint64(self.max_x)
        if (upts > max_x).any():
            raise ValueError('point coordinates out of range [0, 2**p - 1]')

        M     = upts.shape[0]
        n     = self.n
        pbits = self.p

        # Inverse undo excess work
        q = np.uint64(1) << np.uint64(pbits - 1)  # m
        while q > 1:
            pmask = q - 1
            for i in range(n):
                mask = (upts[:, i] & q) != 0
                if mask.any():
                    upts[mask, 0] ^= pmask
                if (~mask).any():
                    t = (upts[~mask, 0] ^ upts[~mask, i]) & pmask
                    upts[~mask, 0] ^= t
                    upts[~mask, i] ^= t
            q >>= 1

        # Gray encode
        for i in range(1, n):
            upts[:, i] ^= upts[:, i - 1]

        tmask = np.zeros(M, dtype=np.uint64)
        q = np.uint64(1) << np.uint64(pbits - 1)
        while q > 1:
            mask = (upts[:, n - 1] & q) != 0
            if mask.any():
                tmask[mask] ^= (q - 1)
            q >>= 1
        upts ^= tmask[:, None]

        # Interleave bit-planes into a big (potentially >64-bit) integer per row
        # > Build as Python ints to avoid overflow; keep it efficient by
        # >   - computing the per-plane "combine" (0..(2^n-1)) vectorized,
        # >   - updating the big integer with one vectorized object op per plane
        # > Using dtype=object leverages Python big-int shifts/ors elementwise
        h = np.zeros(M, dtype=object)

        # Precompute weights across dims for each plane: (1 << (n-1-i))
        dim_weights = np.array([1 << (n - 1 - i) for i in range(n)], dtype=np.uint64)

        for bit in range(pbits - 1, -1, -1):
            # bits_plane: (M, n) of 0/1 for current bit
            bits_plane = (upts >> np.uint64(bit)) & np.uint64(1)
            # Combine per row into [0 .. 2^n-1] by weighted sum across dims
            combine = (bits_plane * dim_weights).sum(axis=1).astype(np.uint64)
            # Update big-int index: shift by n, then OR the small combine
            # > Cast combine to object just for the OR
            h = (h << n) | combine.astype(object)

        # Result type parity
        if match_type and isinstance(points, np.ndarray):
            return np.array(h, dtype=points.dtype, copy=False)

        return list(map(int, h))

    # Attach the vectorized helper
    HilbertCurve._distances_from_points_numpy = _distances_from_points_numpy  # type: ignore[attr-defined]

    # Wrap the public API to prefer the NumPy path
    _orig_dfp = HilbertCurve.distances_from_points

    def _dfp_patched(self, points: Iterable[Iterable[int]], match_type: bool = False):
        """ Wrapper for the monkey-patched numby implementation
        """
        # Prefer NumPy path when possible
        if isinstance(points, np.ndarray):
            try:
                distances = self._distances_from_points_numpy(points, match_type=match_type)
            except Exception as e:
                raise RuntimeError(f'HilbertCurve.distances_from_points_numpy encountered an unexpected error: {e}')
                # Fallback to original behavior on any unexpected issue
                # distances = _orig_dfp(self, points, match_type=match_type)
        else:
            distances = _orig_dfp(self, points, match_type=match_type)

        return distances

    HilbertCurve.distances_from_points = _dfp_patched
    HilbertCurve._numpy_patch_applied  = True  # type: ignore[attr-defined]
