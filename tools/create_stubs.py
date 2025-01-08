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
import importlib
import os
import re
import subprocess
import sys
from collections import deque
from dataclasses import dataclass
from shutil import which
from time import time
from typing import Final, Optional
# ----------------------------------------------------------------------------------------------------------------------------------
# Third-party libraries
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local imports
# ----------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Local definitions
# ----------------------------------------------------------------------------------------------------------------------------------
# ==================================================================================================================================


@dataclass
class Colors:
    # Define colors used throughout this framework
    #
    # Attributes:
    #     WARN    (str): Defines color for warnings.
    #     END     (str): Defines end of color in string.
    BANNERA: Final[str] = '\033[93m'
    BANNERB: Final[str] = '\033[94m'
    WARN:    Final[str] = '\033[91m'
    END:     Final[str] = '\033[0m'


def find_executable():
    """ Attempt to find the stub generation executable
        > Primary location:   PATH
        > Secondary location: Mason
    """
    for exe in ['pyright', 'basedpyright']:
        path = which(exe)
        if path:
            return path

    mason_bin_dir = os.path.expanduser('~/.local/share/nvim/mason/bin')
    for exe in ['pyright', 'basedpyright']:
        path = os.path.join(mason_bin_dir, exe)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    raise FileNotFoundError('Neither "pyright" nor "basedpyright" found in PATH or Mason directory.')


def find_git_root():
    """ Attempt to find the git root
    """
    try:
        result = subprocess.run(['git', 'rev-parse', '--show-toplevel'],
                                capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        raise FileNotFoundError('Not a Git repository or unable to determine Git root.')


def parse_dependencies(toml_file):
    """ Parse the pyproject.toml file to obtain the dependencies
    """
    with open(toml_file, 'r') as f:
        content = f.read()

    dependencies_match = re.search(r'dependencies\s*=\s*\[([^\]]+)\]', content, re.DOTALL)
    if not dependencies_match:
        return []

    raw_dependencies = dependencies_match.group(1).strip()
    dependencies = [
        re.split(r'[<=>~]', dep.strip().strip("'\""))[0]
        for dep in raw_dependencies.split(',')
        if dep.strip()
    ]

    return dependencies


def fetch_import_name(package: str) -> Optional[str]:
    """ Attempt to fetch the import name of a package
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if line.startswith('Name:'):
                package_name = line.split(':', 1)[1].strip()
                break
        else:
            raise ValueError('Name not found in pip show output.')
    except subprocess.CalledProcessError:
        print(f'Warning: Unable to fetch package metadata for {package}.', file=sys.stderr)
        return None

    # Check both `package_name` and `package_name.replace('-', '_')`
    for name_variant in (package_name, package_name.replace('-', '_')):
        try:
            _ = importlib.import_module(name_variant)
            return name_variant
        except ImportError:
            continue

    print(f'Warning: Unable to determine import name for {package_name}.', file=sys.stderr)
    return None


def fetch_transitive_dependencies(package):
    """ Attempt to fetch transitive dependencies of the direct dependencies
    """
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', package],
                                capture_output=True, text=True, check=True)
        dependencies = []
        for line in result.stdout.splitlines():
            if line.startswith('Requires:'):
                raw_requires = line.split(':', 1)[1].strip()
                if raw_requires:
                    dependencies = [dep.strip() for dep in raw_requires.split(',')]
                break
        return dependencies
    except subprocess.CalledProcessError:
        print(f'Warning: Unable to fetch dependencies for {package}.', file=sys.stderr)
        return []


def collect_all_dependencies(initial_dependencies):
    """ Gather all collected dependencies
    """
    visited          = set()
    to_process       = deque(initial_dependencies)
    all_dependencies = set()

    while to_process:
        current = to_process.popleft()
        if current not in visited:
            visited.add(current)
            all_dependencies.add(current)

            new_dependencies = fetch_transitive_dependencies(current)
            for dep in new_dependencies:
                if dep not in visited:
                    to_process.append(dep)

    return all_dependencies


def run_pyright_on_dependencies(pyright_path, dependencies):
    """ Run (based)pyright to generate the stubs
    """
    for dep in dependencies:
        import_name = fetch_import_name(dep)

        if import_name is None:
            continue

        if import_name == 'meshio':
            print(f'Skipping {import_name} due to known issue.')
            continue

        print(f'Creating stubs for {import_name} using {pyright_path}...')
        try:
            _ = subprocess.run([pyright_path, '--createstub', import_name], check=True)
        except subprocess.CalledProcessError as e:
            print(f'Error while processing {dep} ({import_name}): {e}', file=sys.stderr)


def main():
    """ Main routine
    """
    print(Colors.BANNERA + '┏' + '━'*(79-1))
    # print(Colors.BANNERA + '┃')
    print(Colors.BANNERA + '┃' + ' P y H O P E — Python High-Order Preprocessing Environment')
    # print(Colors.BANNERA + '┃' + ' {}'.format(string))
    print(f'{Colors.BANNERA}┃{Colors.END} > Stub generation')
    print(Colors.BANNERA + '┡' + '━'*(79-1) + Colors.END)

    tStart = time()

    git_root       = find_git_root()
    print('│ ' + f'Git root:       {git_root}')

    toml_file    = os.path.join(git_root, 'pyproject.toml')
    if not os.path.exists(toml_file):
        raise FileNotFoundError(f'{toml_file} not found in Git root.')
    print('│ ' + f'TOML file:      {toml_file}')

    pyright_path   = find_executable()
    pyright_config = os.path.join(git_root, 'pyrightconfig.json')
    print('│ ' + f'Pyright path:   {pyright_path}')
    if not os.path.exists(pyright_config):
        print('Warning: No pyrightconfig.json found in the Git root.')
        print('> Using default configuration.')
        print('| Stubs will be saved to ./typings')
        print('| Pyright may display errors')
    else:
        print('│ ' + f'Pyright config: {pyright_config}')
    print('└' + '─'*(79-1))

    initial_dependencies = parse_dependencies(toml_file)
    if not initial_dependencies:
        print('No dependencies found.')
        return

    print('Resolving transitive dependencies...')
    all_dependencies = collect_all_dependencies(initial_dependencies)
    print(f'Resolved dependencies: {', '.join(all_dependencies)}')

    run_pyright_on_dependencies(pyright_path, all_dependencies)

    tEnd = time()
    print('┏' + '━'*(79-1))
    print('┃ {} completed in [{:.2f} sec]'.format('Stub generation', tEnd - tStart))
    print('┗' + '━'*(79-1))


if __name__ == '__main__':
    main()
