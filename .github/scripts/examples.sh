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

# Script for PyHOPE examples
set -euo pipefail

# Store the base directory
basedir=$(pwd)

# Define colors for PASS/FAIL output
green="\033[32m"
red="\033[31m"
reset="\033[0m"

# Initialize an array for storing results (directory and pass/fail status)
declare -a results

# Store all directories with "parameter.ini" files in an array (bash-only)
mapfile -d '' paramfiles < <(find tutorials -type f -name "parameter.ini" -print0 | sort -z)
echo "Found "${#paramfiles[@]}" parameter.ini files..."

# Calculate max width
max_dir_length=0
for paramfile in "${paramfiles[@]}"; do
  paramdir=$(dirname "$paramfile")
  paramlen=${#paramdir}

  if [ "$paramlen" -gt "$max_dir_length" ]; then
    max_dir_length=$paramlen
  fi
done
box_width=$((max_dir_length + 10))  # Add padding for aesthetics
col_width=$((max_dir_length + 2))   # Add padding for the table

# Iterate over all directories, run tests, and collect results
echo "Running PyHOPE with coverage for each parameter.ini file..."

for paramfile in "${paramfiles[@]}"; do
  paramdir=$(dirname  "$paramfile")
  paramstr=$(basename "$paramdir" )
  paramini=$(basename "$paramfile")

  # Print a message for the running task
  echo ""
  printf "┌─%s─┐\n" "$(printf '─%.0s' $(seq 1 $((box_width))))"
  printf "│ Running %-$(($box_width-8))s │\n" "$paramdir"
  printf "└─%s─┘\n" "$(printf '─%.0s' $(seq 1 $((box_width))))"

  # Change directory to where the parameter.ini file is located, suppress output
  pushd "$paramdir" > /dev/null

  # Run the script under coverage and capture the result
  if coverage run --data-file="$basedir/.coverage.$paramstr" --source="$basedir" -m pyhope "$paramini"; then
    results+=("$paramdir: PASS")
    echo -e "${green}✔ PASS${reset}: $paramini"
  else
    results+=("$paramdir: FAIL")
    echo -e "${red}✖ FAIL${reset}: $paramini"
  fi

  # Return to the previous directory, suppress output
  popd > /dev/null
done

# Combine coverage reports from all example directories
# echo "Combining coverage reports..."
# coverage combine --keep

# Generate the coverage report in XML format
# echo "Generating coverage report..."
# coverage xml
# coverage report --skip-empty --precision=2

# Output the final sorted report as a UTF-8 box-drawing table
echo ""
printf "┌─%s─┬────────┐\n" "$(printf '─%.0s' $(seq 1 $((max_dir_length))))"
printf "│ %-$(($col_width-2))s │ %-5s │\n" "Example Directory" "Result"
printf "├─%s─┼────────┤\n" "$(printf '─%.0s' $(seq 1 $((max_dir_length))))"
for result in "${results[@]}"; do
  paramdir=$(echo "$result" | cut -d ':' -f 1)
  paramres=$(echo "$result" | cut -d ':' -f 2 | tr -d '[:space:]')

  if [[ "$paramres" == "PASS" ]]; then
    printf "│ %-$(($col_width-2))s │ ${green}%-6s${reset} │\n" "$paramdir" "$paramres"
  else
    printf "│ %-$(($col_width-2))s │ ${red}%-6s${reset} │\n" "$paramdir" "$paramres"
  fi
done
printf "└─%s─┴────────┘\n" "$(printf '─%.0s' $(seq 1 $((max_dir_length))))"

# Fail the job if any examples failed
if [[ "${results[*]}" == *"FAIL"* ]]; then
  echo "Some examples failed.";
  exit 1;
else
  echo "All examples passed.";
fi

# Repeat the coverage report for Gitlab CI/CD to pick up
# coverage report --skip-empty --precision=2 | grep TOTAL
