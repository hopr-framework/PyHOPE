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
import json
import urllib.request
from packaging.requirements import Requirement
from packaging.version import Version
from typing import Optional, Union
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


def PyPIVersion(package: str, timeout: int = 10) -> Optional[Version]:
    """ Query PyPI JSON API for the latest version

        Returns:
            Optional[str]: Version string
    """
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            data = json.load(resp)
            info = data.get('info', {})
            return Version(info.get('version'))
    except Exception:  # pragma: no cover
        return None


def _ParseVersion(text: str) -> Optional[Version]:
    """ Find a semver-like version in arbitrary text
    """
    # Local imports ----------------------------------------
    from re import search
    from packaging.version import Version
    # ------------------------------------------------------
    if not text:
        return None

    # 1) Prefer to capture numeric version before a "-git" marker
    #    Example: "4.14.0-git-8425b99f0" -> capture "4.14.0"
    m = search(r"\bv?(\d+(?:\.\d+){0,3})(?=-git\b)", text)
    if m:
        try:
            return Version(m.group(1))
        except Exception:  # pragma: no cover
            return None

    # 2) Fallback: capture semver-like token including -rc / +build etc.
    #    Example: "v1.2.3-rc1", "1.2.3+build.1"
    m = search(r"\bv?(\d+(?:\.\d+){0,3}(?:[-_a-zA-Z0-9+.]+)?)\b", text)
    if m:
        try:
            return Version(m.group(1))
        except Exception:  # pragma: no cover
            return None

    return None


def GmshVersion() -> tuple[Union[Version, bool, None], Union[str, None]]:
    """ Query the local system for the Gmsh version

        Returns:
            Optional[str]: Version string
    """
    # Local imports ----------------------------------------
    import re
    import shutil
    import subprocess
    # ------------------------------------------------------
    path = shutil.which('gmsh')
    ver: Union[Version, bool, None] = None
    pac: Union[str,           None] = None  # noqa: E272

    if path:
        try:
            p   = subprocess.run([path, '--info'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            raw = (p.stdout or '') + "\n" + (p.stderr or '')
            # Parse version from the Version line if present
            v   = _ParseVersion(raw)
            if v:
                ver = v
            # Parse Packaged by: <who>
            m   = re.search(r'Packaged by\s*:\s*(.+)', raw, flags=re.IGNORECASE)
            if m:
                pac = str(m.group(1).strip())
        except Exception:  # pragma: no cover
            pass

    # If no version found yet, try generic flags
    ver = DependencyVersion('gmsh')

    # Finalize fields and return
    if not ver:  # pragma: no cover
        ver = None
    if not pac:  # pragma: no cover
        pac = None
    return ver, pac


def DependencyVersion(program: str) -> Optional[Version | bool]:
    """ Query the local system for the version of a dependency

        Returns:
            Optional[str]: Version string
    """
    # Local imports ----------------------------------------
    import shutil
    import subprocess
    # ------------------------------------------------------
    path = shutil.which(program)

    # Try common version flags. Some executables print version to stdout, others to stderr
    if path:
        for flag in ('--version', '-V', 'version', '-v'):
            try:
                p = subprocess.run([path, flag], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
            except Exception:
                continue

            out = (p.stdout or '') + "\n" + (p.stderr or '')
            ver = _ParseVersion(out)
            if ver:
                return ver

        # Found dependency but failed to parse version
        return True
    return None


def DependencyHealth(program: str,
                     version: Union[Version, bool, None],
                     status:  Optional[str] = None,
                     info:    Optional[str] = None) -> None:
    """ Print the dependency health
    """
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------

    if version:
        if isinstance(version, Version):
            status = status if status is not None else hopout.Symbols.OK
            hopout.info(f'{status} {program} found [v{version}]'      + ('' if info is None else info))
        else:  # pragma: no cover
            status = status if status is not None else hopout.Symbols.WARN
            hopout.info(f'{status} {program} found [unknown version]' + ('' if info is None else info))
    else:
        status = status if status is not None else hopout.Symbols.WARN
        hopout.info(f'{status} {program} not installed' + ('' if info is None else info))


def _PackageInstalledVersion(package: str) -> Optional["Version"]:
    """ Query the local system for the version of a Python package

        Returns:
            Optional[str]: Version string
    """
    # Local imports ----------------------------------------
    import importlib
    from importlib import metadata as importlib_metadata
    from packaging.version import Version
    # ------------------------------------------------------
    # Try distribution metadata first (package name as on PyPI)
    try:
        v = importlib_metadata.version(package)
        return Version(v)
    except Exception:
        pass

    # Fallback: try importing top-level module and reading __version__
    try:
        mod = importlib.import_module(package)
        v = getattr(mod, '__version__', None) or getattr(mod, 'version', None) or getattr(mod, 'release', None)
        if v:
            try:
                return Version(str(v))
            except Exception:  # pragma: no cover
                return None
    except Exception:  # pragma: no cover
        pass

    return None


def PackageHealth(   pkg:    str,
                     version: Union[Version, None],
                     pypiver: Union[Version, None]  # noqa: E272
                 ) -> None:
    # Local imports ----------------------------------------
    import pyhope.output.output as hopout
    # ------------------------------------------------------
    if pypiver and version:
        try:
            if version >= pypiver:
                hopout.info(f'{hopout.Symbols.OK  } {pkg} [v{version}] is up-to-date')
            else:
                hopout.info(f'{hopout.Symbols.WARN} {pkg} [v{version}] is outdated (PyPI: v{pypiver})')
        except Exception:  # pragma: no cover
            hopout.info(f'{hopout.Symbols.WARN} {pkg} [v{version}] is installed (PyPI: v{pypiver}) -- unable to compare reliably')
    elif version:  # pragma: no cover
        hopout.info(f'{hopout.Symbols.WARN} {pkg} [v{version}] is installed (PyPI info unavailable)')
    else:  # pragma: no cover
        hopout.info(f'{hopout.Symbols.ERR} {pkg} [v{version}] is not installed')


def CheckHealth() -> None:
    """ Internal health check
    """
    # Local imports ----------------------------------------
    from pyhope.common.common_vars import Common
    import pyhope.output.output as hopout
    from importlib import metadata as importlib_metadata
    # ------------------------------------------------------

    common  = Common()
    program = common.program
    symbols = hopout.Symbols

    hopout.small_banner('System')

    pypiver = PyPIVersion('pyhope')
    PackageHealth(common.program, Version(common.version), pypiver)

    hopout.info('')
    hopout.small_banner('Packages')

    # Try to read the installed distribution metadata
    try:
        dist = importlib_metadata.distribution(program)
        pkgs = dist.requires or []
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        # Fallback: try metadata() api to read Requires-Dist directly
        try:
            meta = importlib_metadata.metadata(program)
            # metadata.get_all returns a sequence of Requires-Dist entries if present
            pkgs = list(meta.get_all('Requires-Dist') or [])
        except Exception:
            pkgs = []

    if not pkgs:  # pragma: no cover
        print(hopout.warn(f'Could not discover declared Python requirements for {program}.'))
        print(hopout.warn( 'If you are running from a source checkout, install the package first (pip install -e .) to ' +
                           'enable requirement checks.'))

    for pack in pkgs:
        # packaging.Requirement will handle extras and environment markers
        try:
            req = Requirement(pack)
        except Exception:  # pragma: no cover
            # If parsing fails, fall back to a simple split at first space or semicolon
            raw_name = pack.split()[0].split(';')[0]
            name = raw_name.split('(')[0].strip()
            req  = None
            pkg  = name
        else:  # pragma: no cover
            pkg  = req.name

        pkgver  = _PackageInstalledVersion(pkg)
        pypiver = PyPIVersion(pkg)
        PackageHealth(pkg, pkgver, pypiver)

    hopout.info('')
    hopout.small_banner('Dependencies')

    # For Gmsh, also check the packager
    gmshv, gmshp = GmshVersion()
    if gmshp is not None and gmshp.strip() == 'NRG':
        gmshi = ' (packaged by NRG)'
        gmshs = symbols.OK
    else:  # pragma: no cover
        gmshi = ' (packaged externally)'
        gmshs = symbols.WARN

    DependencyHealth('Gmsh'    , version=gmshv,           status=gmshs, info=gmshi)
    # Warn if we know that Gmsh uses an outdated CGNS
    if gmshp is not None and gmshp.strip() != 'NRG':  # pragma: no cover
        print(hopout.warn('Detected Gmsh package uses an outdated CGNS (v3.4). ' +
                          'For compatibility, replace with the updated NRG version'))
    DependencyHealth('ParaView', version=DependencyVersion('paraview'), info=' (optional)')
