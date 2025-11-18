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
import glob
import os
import shutil
import subprocess
import sys
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
import argparse
from typing import List, Optional, Mapping, Tuple
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================

# ANSI colors
GREEN = '\033[32m'
RED   = '\033[31m'
CYAN  = '\033[36m'
RESET = '\033[0m'


def printHeader(title: str, width: int) -> None:
    ''' Print a box header for a section.
    '''
    print('')
    print('┌─' + '─' * width + '─┐')
    print('│ ' + f'{title:<{width}}' + ' │')
    print('└─' + '─' * width + '─┘')


def findCoverageFiles(base_dir   : str,
                      pattern    : str,
                      output_name: str) -> List[str]:
    ''' Find coverage data files in base_dir that match pattern, excluding the combined output
    '''
    candidates    = sorted(f for f in glob.glob(os.path.join(base_dir, pattern)) if os.path.isfile(f))
    combined_path = os.path.join(base_dir, output_name)
    combined_abs  = os.path.abspath(combined_path)
    input_datafiles = [f for f in candidates if os.path.abspath(f) != combined_abs]
    return input_datafiles


def findParameterFiles(base_dir  : str,
                       param_name: str) -> List[str]:
    ''' Recursively find all parameter.ini files under tutorials_root
    '''
    matches: List[str] = []
    for base_dir, _dirs, files in os.walk(base_dir):
        if param_name in files:
            matches.append(os.path.join(base_dir, param_name))
    matches.sort()
    return matches


def runCmd(cmd: List[str],
           cwd: str,
           env: Optional[Mapping[str, str]] = None) -> None:
    ''' Run a command, streaming output to the console. Raise on failure.
    '''
    try:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
    except subprocess.CalledProcessError:
        print(f'{RED}Command failed:{RESET} {" ".join(cmd)}')
        raise


def runExamples(base_dir  : str,
                exam_dir  : str,
                param_name: str) -> Tuple[List[Tuple[str, str]], int]:
    ''' Run all examples found under tutorials_root with coverage collection
    '''
    paramfiles = findParameterFiles(exam_dir, param_name)
    print(f'Found {len(paramfiles)} {param_name} files...')

    # Calculate widths
    max_dir_length = 0
    paramdirs      = [os.path.dirname(p) for p in paramfiles]
    for d in paramdirs:
        if len(d) > max_dir_length:
            max_dir_length = len(d)
    box_width = max_dir_length + 10  # Add padding for aesthetics
    col_width = max_dir_length + 2   # Add padding for the table

    print('Running PyHOPE with coverage for each parameter.ini file...')

    results: List[Tuple[str, str]] = []
    failures                        = 0

    for paramfile in paramfiles:
        paramdir = os.path.dirname( paramfile)
        paramstr = os.path.basename(paramdir)
        paramini = os.path.basename(paramfile)

        printHeader(f'Running {paramdir}', box_width)

        # Run coverage in the example directory, writing a unique data-file at the repo root
        datafile = os.path.join(base_dir, f'.coverage.{paramstr}')
        cmd       = [
            'coverage', 'run',
            f'--data-file={datafile}',
            f'--source={base_dir}',
            '-m', 'pyhope', paramini,
        ]
        proc = subprocess.run(cmd, cwd=paramdir)
        if proc.returncode == 0:
            results.append((paramdir, 'PASS'))
            print(f'{GREEN}✔ PASS{RESET}: {paramini}')
        else:
            results.append((paramdir, 'FAIL'))
            print(f'{RED}✖ FAIL{RESET}: {paramini}')
            failures += 1

    # Output the final sorted report as a UTF-8 box-drawing table
    if results:
        print('')
        print('┌─' + '─' * max_dir_length + '─┬────────┐')
        print('│ ' + f'{"Example Directory":<{col_width-2}}' + ' │ ' + f'{"Result":<5}' + ' │')
        print('├─' + '─' * max_dir_length + '─┼────────┤')
        for paramdir, status in results:
            if status == 'PASS':
                print('│ ' + f'{paramdir:<{col_width-2}}' + ' │ ' + f'{GREEN}{status:<6}{RESET}' + ' │')
            else:
                print('│ ' + f'{paramdir:<{col_width-2}}' + ' │ ' + f'{RED}{status:<6}{RESET}' + ' │')
        print('└─' + '─' * max_dir_length + '─┴────────┘')

    # Return both results and failure count
    return results, failures


def runVerify(base_dir: str) -> int:
    ''' Run the internal health check with coverage and return the process return code
    '''
    print('Running internal health check with coverage...')
    data_file = os.path.join(base_dir, '.coverage.verify')
    cmd       = ['coverage', 'run', f'--data-file={data_file}', '-m', 'pyhope', '--verify']
    proc      = subprocess.run(cmd, cwd=base_dir)
    if proc.returncode == 0:
        print(f'{GREEN}✔ PASS{RESET}: pyhope --verify')
    else:
        print(f'{RED}✖ FAIL{RESET}: pyhope --verify')
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description='Combine multiple coverage data files into a single .coverage file, and optionally generate them by running tutorials and the internal health check.')  # noqa: E501
    # Generation controls
    parser.add_argument('--run-examples',  action  = 'store_true',                                                                            # noqa: E251
                                        help      = 'Run all tutorials (parameter.ini) with coverage and create .coverage.<example> files.',)  # noqa: E251, E501
    parser.add_argument('--run-verify',    action  = 'store_true',                                                                            # noqa: E251
                                        help      = 'Run "pyhope --verify" with coverage and create .coverage.verify.',                     )  # noqa: E251, E501
    parser.add_argument('--tutorials-root', default = 'tutorials',                                                                            # noqa: E251
                                        help      = 'Root directory to search for tutorials (default: "tutorials").',                       )  # noqa: E251, E501
    parser.add_argument('--param-name',    default = 'parameter.ini',                                                                         # noqa: E251
                                        help      = 'Name of the parameter file to look for (default: "parameter.ini").',                   )  # noqa: E251, E501
    # Combine/report controls
    parser.add_argument('--base-dir',      default = '.',                                                                                     # noqa: E251
                                        help      = 'Base directory where coverage data files are located (default: current directory).',   )  # noqa: E251, E501
    parser.add_argument('--pattern',       default = '.coverage.*',                                                                           # noqa: E251
                                        help      = 'Glob pattern for input coverage files (default: ".coverage.*").',                      )  # noqa: E251, E501
    parser.add_argument('--output',        default = '.coverage',                                                                             # noqa: E251
                                        help      = 'Name of the combined coverage file to produce (default: ".coverage").',                )  # noqa: E251, E501
    parser.add_argument('--keep',          default = True,                                                                                    # noqa: E251
                                        action    = 'store_true',                                                                            # noqa: E251
                                        help      = 'Keep original data files after combining (default: True).',                            )  # noqa: E251, E501
    parser.add_argument('--no-keep',       dest    = 'keep',                                                                                  # noqa: E251
                                        action    = 'store_false',                                                                           # noqa: E251
                                        help      = 'Remove original data files after combining.',                                          )  # noqa: E251, E501
    parser.add_argument('--xml',           action   = 'store_true',                                                                           # noqa: E251
                                        help      = 'Also generate "coverage.xml" after combining.',                                        )  # noqa: E251, E501
    parser.add_argument('--report',        action   = 'store_true',                                                                           # noqa: E251
                                        help      = 'Also print a text coverage report after combining (skip-empty, precision=2).',         )  # noqa: E251, E501
    parser.add_argument('--strict',        action   = 'store_true',                                                                           # noqa: E251
                                        help      = 'Exit with a non-zero status if no coverage data files are found (combine step only).',)  # noqa: E251, E501

    args      = parser.parse_args()
    base_dir  = os.path.abspath(args.base_dir)
    out_name  = args.output
    out_path  = os.path.join(base_dir, out_name)

    # Ensure coverage CLI is available
    if shutil.which('coverage') is None:
        print(f'{RED}Error:{RESET} The "coverage" CLI was not found. Install with: pip install coverage')
        return 2

    # Optional generation: examples/tutorials
    any_failures = 0
    if args.run_examples:
        print(f'{CYAN}Searching for coverage data files to generate from tutorials...{RESET}')
        _results, failures = runExamples(base_dir=base_dir, exam_dir=args.tutorials_root, param_name=args.param_name)
        if failures:
            any_failures += failures
            print('Some examples failed.')

    # Optional generation: verify
    verify_rc = 0
    if args.run_verify:
        verify_rc = runVerify(base_dir=base_dir)

    # Combine coverage reports from all data files
    print('Combining coverage reports...')
    files = findCoverageFiles(base_dir, args.pattern, args.output)
    if not files:
        msg = f'No coverage data files found matching pattern "{args.pattern}" in {base_dir}.'
        if args.strict:
            print(f'{RED}{msg}{RESET}')
            # Preserve failures precedence if any occurred earlier
            return 1 if any_failures == 0 else 1
        print(msg)
        # Still return failures if generation failed
        if any_failures or verify_rc != 0:
            return 1
        return 0

    # Pretty-print discovered input files
    print(f'Found {len(files)} coverage data file(s):')
    for f in files:
        print(f'  - {f}')

    # Prepare environment so coverage writes/reads the desired combined file
    env                 = os.environ.copy()
    env['COVERAGE_FILE'] = out_path

    # Combine (equivalent to: coverage combine --keep [files...])
    cmd = ['coverage', 'combine']
    if args.keep:
        cmd.append('--keep')
    cmd.extend(files)
    runCmd(cmd, cwd=base_dir, env=env)

    if os.path.exists(out_path):
        print(f'{GREEN}✔ Combined coverage written to:{RESET} {out_path}')
    else:
        print(f'{RED}Warning:{RESET} Expected combined file not found at {out_path}.')
        print('The "coverage combine" command may have used a different data file name via COVERAGE_FILE or config.')

    # Optionally generate reports (mirrors CI job steps)
    if args.xml:
        print('Generating coverage XML...')
        runCmd(['coverage', 'xml'], cwd=base_dir, env=env)
        print(f'{GREEN}✔ Generated coverage.xml{RESET}')

    if args.report:
        print('Generating coverage text report...')
        runCmd(['coverage', 'report', '--skip-empty', '--precision=2'], cwd=base_dir, env=env)

    # Fail the job if any examples or verify failed
    if any_failures or verify_rc != 0:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
