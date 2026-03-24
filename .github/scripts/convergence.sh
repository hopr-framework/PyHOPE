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
artifact_file="convergence_artifacts.log"
rm $artifact_file || true
touch $artifact_file

directories=("5-02-convtest_flipped" "5-03-convtest_mortar") # List of directories to process
for dir in "${directories[@]}"; do
  echo "=== Processing directory: $dir ==="
  rm -rf convergence_test || true
  mkdir convergence_test
  cd convergence_test
  cp "../tutorials/$dir/parameter_template_pyhope.ini" .
  cp "../tutorials/$dir/parameter_flexi.ini" .
  # Generate Meshes
  meshres=("001" "002" "004" "008")
  template_file="parameter_template_pyhope.ini"
  for res in "${meshres[@]}"; do
    # Calculate nElem2 as half the integer res of nElem
    nElem2=$((10#$res / 2)) # Interpret value as a base-10 integer
    # Create a temporary parameter file
    temp_file=$(mktemp)
    # Replace placeholders in the template and save to the temporary file
    sed -e "s/<nElem>/$res/g" -e "s/<nElem2>/$nElem2/g" "$template_file" > "$temp_file"
    # Build the mesh
    pyhope "$temp_file"
    # Clean up the temporary file
    rm "$temp_file"
  done
  # Call convergence script
  for N in {1..4}; do
    echo ""
    echo "=== Running Convergence Test N=$N ===" | tee -a ../convergence.log
    # Restrict multiprocessing
    procs=4
    if [ "$N" -eq 1 ]; then procs=2; fi
    # Run FLEXI
    python3 /flexi/tools/convergence_test/convergence_grid --N $N --dim 3 --procs "$procs" /flexi/bin/flexi parameter_flexi.ini | tee -a ../convergence.log
  done
  cd ..
  # Merge convergence.log for artifacts
  echo "===== Processing Folder: $dir =====" >> "$artifact_file"
  cat "convergence.log" >> "$artifact_file"
  echo "" >> "$artifact_file"
  # Analyze convergence behavior using a simple python script
  python3 ./convergence.py

  # Clean-up the log file
  rm -f convergence.log
done
