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
# Stub generation helper
# ==================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------
# Standard libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import argparse
import h5py
import numpy as np
import toml  # ty: ignore [unresolved-import]
from typing import Union
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


def collectStats(file_path: str) -> dict:
    stats = {}
    with h5py.File(file_path, 'r') as f:

        def visit_func(name: str, obj: Union[np.ndarray, h5py.Dataset]) -> None:
            if name not in ('ElemInfo', 'GlobalNodeIDs', 'NodeCoords', 'SideInfo'):
                return

            if isinstance(obj, h5py.Dataset):
                try:
                    data = obj[()]
                    if np.issubdtype(data.dtype, np.number):
                        minv    = float(np.min(data))
                        maxv    = float(np.max(data))
                        meanv   = float(np.mean(data))
                        stddevv = float(np.std(data))
                        print(f'Dataset: {name}, min: {minv}, max: {maxv}, mean: {meanv}, stddev: {stddevv}')
                        stats[name] = {
                            'min': minv,
                            'max': maxv,
                            'mean': meanv,
                            'stddev': stddevv
                        }
                    else:
                        print(f'Dataset: {name} skipped (non-numeric data)')
                except Exception as e:
                    print(f'Dataset: {name} - error reading data: {e}')
        f.visititems(visit_func)
    return stats


def writeTOML(stats: dict, output_file: str = 'analyze.toml') -> None:
    # TOML tables can't have slashes, so replace with dot or underscore
    toml_compatible_stats = {name.replace('/', '.') : values for name, values in stats.items()}
    with open(output_file, 'w') as f:
        toml.dump(toml_compatible_stats, f)
    print(f'Stats written to {output_file}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Collect statistics for HDF5 HOPR mesh file and output to TOML')
    parser.add_argument('file_path',      type    = str,                      # noqa: E251
                                          help    = 'Path to the HDF5 file')  # noqa: E251
    parser.add_argument('--output', '-o', type    = str,                      # noqa: E251
                                          default = 'analyze.toml',           # noqa: E251
                                          help    = 'Output TOML file name')  # noqa: E251
    args = parser.parse_args()

    print(f'Reading mesh {args.file_path}')
    stats = collectStats(args.file_path)
    writeTOML(stats, args.output)


if __name__ == '__main__':
    main()
