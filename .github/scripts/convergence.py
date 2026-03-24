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
# Continuous Integration/Continuous Deployment
# ==================================================================================================================================

# Script for FLEXI convergence checks
import sys

# Input file
convfile = "convergence.log"


# Set tolerance bounds: (N+1)+-0.35
def compute_expected_range(N):
    return N + 0.65, N + 1.35


# Process the results
def process_results(degree, data):
    min_eoc, max_eoc = compute_expected_range(degree)
    print(f"\n=== Checking EOCs for Polynomial Degree N={degree} (Expected: [{min_eoc}, {max_eoc}]) ===")
    results = []
    for idx, (grid, eoc) in enumerate(data[1:]):
        passed = min_eoc <= eoc <= max_eoc
        results.append((grid, min_eoc, max_eoc, eoc, "PASS" if passed else ("(PASS)" if idx < len(data) - 2 else "FAIL")))
    draw_table(results)
    return all(r[4] == "PASS" for r in results[-1:])


# Draw the results as a table
def draw_table(results):
    print("\n┌───────┬─────────┬─────────┬─────────┬────────┐")
    print("│ Mesh  │   min   │   max   │  result │ Passed │")
    print("├───────┼─────────┼─────────┼─────────┼────────┤")
    for grid, min_eoc, max_eoc, result, passed in results:
        pass_col = ( "\033[92m PASS \033[0m" if passed == "PASS" else ("\033[93m(PASS)\033[0m" if passed == "(PASS)" else "\033[91m FAIL \033[0m"))  # noqa: E501
        print(f"│ {grid:^5} │ {min_eoc:^7.2f} │ {max_eoc:^7.2f} │ {result:^7.2f} │ {pass_col:^6} │")
    print("└───────┴─────────┴─────────┴─────────┴────────┘")


# Parse the log file
try:
    with open(convfile, "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    sys.exit(f"Error: File '{convfile}' not found.")

# Process the file
success, current_degree, current_data = True, None, []

for line in lines:
    line = line.strip()
    # Match polynomial degree header
    if line.startswith("=== Running Convergence Test") and "N=" in line:
        # Process the previous degree upon encountering a new header
        if current_degree is not None:
            success &= process_results(current_degree, current_data)
        try:
            current_degree = int(line.split("N=")[1].split()[0])
        except ValueError:
            current_degree = None
        current_data = []
    # Match grid data rows
    elif current_degree is not None and "║" in line:
        try:
            grid, eoc = int(line.split("║")[0].strip()), float(line.split("║")[2].strip())
            current_data.append((grid, eoc))
        except ValueError:
            pass

# Final processing for the last degree with no subsequent header
if current_degree is not None:
    success &= process_results(current_degree, current_data)

# Final status
if not success:
    print("Some EOCs are out of range.", flush=True)
    sys.exit(1)
else:
    sys.exit(0)
