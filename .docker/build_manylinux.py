#!/opt/python/cp310-cp310/bin/python
# -*- coding: utf-8 -*-
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This file is part of PyHOPE
#
# Copyright (c) 2022-2024 Andrea Beck
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
import os
import sys
import build
import errno
import multiprocessing
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from typing import Optional
# ==================================================================================================================================

# Output some runtime information
print('Python version:', sys.version)


def print_header(title: str) -> None:
    """ Prints a nicely formatted header with a title
    """
    header_line = '=' * 80
    print('\n{}\n{:^80}\n{}'.format(header_line, title, header_line))


def print_step(step_description: str) -> None:
    """ Prints a step description in a consistent format
    """
    print('➤ {}'.format(step_description))


# Helper function for downloads
def download(url: str, dir: str) -> str:
    file = os.path.join(dir, os.path.basename(url))
    print_step('Downloading: {} to {}'.format(url, file))
    urllib.request.urlretrieve(url, file)
    return file


# Helper function to extract .tar.gz files
def extract(file: str, dir: str) -> None:
    print_step('Extracting: {} to {}'.format(file, dir))
    if file.endswith('.tar.gz') or file.endswith('.tgz'):
        mode = 'r:gz'
    elif file.endswith('.tar.xz'):
        mode = 'r:xz'
    # elif file.endswith('.zip'):
    #     subprocess.call(['unzip', '-j', file])
    #     return None
    else:
        raise ValueError('Unsupported archive format: {}'.format(file))

    with tarfile.open(file, mode) as tar:
        tar.extractall(path=dir)


# Helper function to configure and build software
def configure(configure_cmd: list, cwd: Optional[str] = None, env: Optional[dict] = None) -> None:
    cwd = cwd or os.getcwd()

    # Run the configure script ...
    configure_path = os.path.join(cwd, 'configure')
    if os.path.isfile(configure_path):
        print_step('Configuring with: {}'.format(configure_cmd))
        subprocess.run(configure_cmd, check=True, cwd=cwd, env=env, stderr=sys.stderr, stdout=sys.stdout)

    # ... or the cmake script
    elif configure_cmd[0] == 'cmake':
        print_step('Configuring with: {}'.format(configure_cmd))
        subprocess.run(configure_cmd, check=True, cwd=cwd, env=env, stderr=sys.stderr, stdout=sys.stdout)

    # ... or the meson script
    elif configure_cmd[0] == 'meson':
        print_step('Configuring with: {}'.format(configure_cmd))
        subprocess.run(configure_cmd, check=True, cwd=cwd, env=env, stderr=sys.stderr, stdout=sys.stdout)


def compile(install_cmd: Optional[list] = None, ncores: int = 1, cwd: Optional[str] = None, env: Optional[dict] = None) -> None:
    # Build the software
    print_step('Building with {} cores'.format(ncores))
    subprocess.run(['make', f'-j{ncores}'], check=True, cwd=cwd, env=env, stderr=sys.stderr, stdout=sys.stdout)

    # Run the install command if provided
    if install_cmd:
        print_step('Installing...')
        subprocess.run(install_cmd, check=True, cwd=cwd, stderr=sys.stderr, stdout=sys.stdout)

# Get systemc architecture
arch = platform.machine()

# Define paths
WORK_DIR     = '/io'
BUILD_DIR    = os.path.join(WORK_DIR, 'build')
INSTALL_DIR  = os.path.join(WORK_DIR, 'gmsh_install')
Doxygen_DIR  = os.path.join(WORK_DIR, 'doxygen')

# HDF5
HDF5_DIR     = os.path.join(WORK_DIR, 'hdf5')
HDF5_VERSION = '1.14.5'

# TK
TK_DIR       = os.path.join(WORK_DIR, 'tk')
TK_VERSION = '8.6.15'

# TCL
TCL_DIR      = os.path.join(WORK_DIR, 'tcl')
TCL_VERSION = '8.6.14'

# CGNS
CGNS_DIR     = os.path.join(WORK_DIR, 'CGNS')
CGNS_VERSION = 'v4.4.0'

# OpenCASCADE
OCC_DIR      = os.path.join(WORK_DIR, 'occt')
OCC_VERSION  = 'V7_8_1'

# FreeType
FREETYPE_VERSION = '2.13.3'
FREETYPE_DIR = os.path.join(WORK_DIR, 'freetype')

# libXFT
LIBXFT_DIR     = os.path.join(WORK_DIR, 'libxft')
LIBXFT_VERSION = '2.3.8'

# libPNG
LIBPNG_DIR     = os.path.join(WORK_DIR, 'libpng')
LIBPNG_VERSION = '1.6.44'

# libJPEG
LIBJPEG_TURBO_DIR     = os.path.join(WORK_DIR, 'libjpeg-turbo')
LIBJPEG_TURBO_VERSION = '3.0.4'

# FLTK
FLTK_DIR     = os.path.join(WORK_DIR, 'fltk')
FLTK_VERSION = '1.3.9'
# FLTK_VERSION = '1.4.0'  # Currently broken, produces segfault

# gperf
GPERF_VERSION = "3.1"

# Fontconfig
FONTCONFIG_VERSION = '2.14.2'
FONTCONFIG_DIR = os.path.join(WORK_DIR, 'fontconfig')

# LIBXFT
LIBXFT_VERSION = '2.3.8'

# GLU
# GLU_DIR      = os.path.join(WORK_DIR, 'glu')
# GLU_VERSION  = '9.0.3'

# Gmsh
GMSH_VERSION = '4.13.1'
GMSH_STRING  = 'gmsh_{}'.format(GMSH_VERSION.replace('.', '_'))
GMSH_DIR     = os.path.join(WORK_DIR, 'gmsh')

# Gitlab "python-gmsh" access
LIB_GITLAB   = 'gitlab.iag.uni-stuttgart.de'
# LIB_PROJECT  = 'libs/python-gmsh'
LIB_PROJECT  = '797'


# Determine the number of available cores and leave 2 for other tasks
total_cores  = multiprocessing.cpu_count()
build_cores  = max(total_cores - 2, 1)


# ------------------------------------------------------------------------
# Clean any previous build artifacts
# ------------------------------------------------------------------------
def clean():
    print_header('CLEANING UP PREVIOUS BUILD ARTIFACTS...')

    shutil.rmtree(BUILD_DIR,         ignore_errors=True)
    shutil.rmtree(INSTALL_DIR,       ignore_errors=True)
    # All libraries
    shutil.rmtree(HDF5_DIR,          ignore_errors=True)
    shutil.rmtree(TK_DIR,            ignore_errors=True)
    shutil.rmtree(TCL_DIR,           ignore_errors=True)
    shutil.rmtree(CGNS_DIR,          ignore_errors=True)
    shutil.rmtree(OCC_DIR,           ignore_errors=True)
    shutil.rmtree(FREETYPE_DIR,      ignore_errors=True)
    shutil.rmtree(LIBJPEG_TURBO_DIR, ignore_errors=True)
    shutil.rmtree(LIBPNG_DIR,        ignore_errors=True)
    shutil.rmtree(LIBXFT_DIR,        ignore_errors=True)
    shutil.rmtree(FLTK_DIR,          ignore_errors=True)
    shutil.rmtree(FONTCONFIG_DIR,    ignore_errors=True)
    shutil.rmtree(GMSH_DIR,          ignore_errors=True)
    # Python dist
    DIST_DIR = os.path.join(WORK_DIR, 'dist')
    EGG_DIR  = os.path.join(WORK_DIR, 'gmsh.egg-info')
    shutil.rmtree(DIST_DIR,          ignore_errors=True)
    shutil.rmtree(EGG_DIR,           ignore_errors=True)
    # Python wheel
    try:
        os.remove(os.path.join(WORK_DIR, 'dist', f'gmsh-{GMSH_VERSION}-py3-none-any.whl'))
    except OSError:
        pass

    os.makedirs(BUILD_DIR,   exist_ok=True)
    os.makedirs(INSTALL_DIR, exist_ok=True)


# ------------------------------------------------------------------------
# Build HDF5 (Static)
# ------------------------------------------------------------------------
def build_hdf5() -> None:
    print_header('BUILDING HDF5...')
    HDF5_VERSION_UNDERSCORE = HDF5_VERSION.replace('.', '_')
    HDF5_MAJOR_UNDERSCORE   = HDF5_VERSION_UNDERSCORE.rsplit('_', 1)[0]

    os.makedirs(BUILD_DIR, exist_ok=True)

    url  = ( f'https://support.hdfgroup.org/releases/hdf5/v{HDF5_MAJOR_UNDERSCORE}/'
             f'v{HDF5_VERSION_UNDERSCORE}/downloads/hdf5-{HDF5_VERSION}.tar.gz' )
    file = download(url, BUILD_DIR)
    extract(file, BUILD_DIR)

    hdf5_src_dir = os.path.join(BUILD_DIR, f'hdf5-{HDF5_VERSION}')
    conf_cmd = [ './configure',
                '--prefix={}'.format(HDF5_DIR),
                '--enable-static',
                # '--enable-shared'
                '--disable-shared'
                ]
    conf_env = os.environ.copy()
    conf_env['CFLAGS'  ] = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'

    build_cmd = ['make', 'install']
    if os.path.exists(HDF5_DIR):
        shutil.rmtree(HDF5_DIR)

    configure(conf_cmd,              env=conf_env, cwd=hdf5_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=hdf5_src_dir)


# ------------------------------------------------------------------------
# Build Tcl (Static)
# ------------------------------------------------------------------------
def build_tcl() -> None:
    print_header('BUILDING TCL...')

    url  = f'https://downloads.sourceforge.net/sourceforge/tcl/tcl{TCL_VERSION}-src.tar.gz'
    file = download(url, BUILD_DIR)
    extract(file, BUILD_DIR)

    tcl_src_dir = os.path.join(BUILD_DIR, f'tcl{TCL_VERSION}', 'unix')
    conf_cmd = ['./configure',
                '--prefix={}'.format(TCL_DIR),
                '--enable-static',
                # '--disable-shared'
                '--enable-shared'
                ]
    conf_env = os.environ.copy()
    conf_env['CFLAGS']   = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'

    build_cmd = ['make', 'install']

    if os.path.exists(TCL_DIR):
        shutil.rmtree(TCL_DIR)

    configure(conf_cmd,              env=conf_env, cwd=tcl_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=tcl_src_dir)


# ------------------------------------------------------------------------
# Build Tk (Static)
# ------------------------------------------------------------------------
def build_tk() -> None:
    print_header('BUILDING TK...')

    url  = f'https://downloads.sourceforge.net/sourceforge/tcl/tk{TK_VERSION}-src.tar.gz'
    file = download(url, BUILD_DIR)
    extract(file, BUILD_DIR)

    tk_src_dir = os.path.join(BUILD_DIR, f'tk{TK_VERSION}', 'unix')
    conf_cmd = ['./configure',
                '--prefix={}'.format(TK_DIR),
                '--with-tcl={}/lib'.format(TCL_DIR),
                '--enable-static',
                # '--disable-shared'
                '--enable-shared'
                ]
    conf_env = os.environ.copy()
    conf_env['CFLAGS']   = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'

    build_cmd = ['make', 'install']

    if os.path.exists(TK_DIR):
        shutil.rmtree(TK_DIR)

    configure(conf_cmd,              env=conf_env, cwd=tk_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=tk_src_dir)


# ------------------------------------------------------------------------
# Build FreeTyoe
# ------------------------------------------------------------------------
def build_freetype():
    print_header('BUILDING FREETYPE2')

    freetype_src_dir = os.path.join(BUILD_DIR, f'freetype-{FREETYPE_VERSION}')

    url = f'https://download.savannah.gnu.org/releases/freetype/freetype-{FREETYPE_VERSION}.tar.gz'
    file = download(url, BUILD_DIR)
    extract(file, BUILD_DIR)

    conf_cmd  = ['./configure',
                 '--prefix={}'.format(FREETYPE_DIR),
                 '--enable-static',
                 '--disable-shared',
                 # '--enable-shared',
                 ]
    conf_env = os.environ.copy()
    conf_env['CFLAGS']   = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'

    build_cmd = ['make', 'install']

    if os.path.exists(FREETYPE_DIR):
        shutil.rmtree(FREETYPE_DIR)

    configure(conf_cmd,              env=conf_env, cwd=freetype_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=freetype_src_dir)

    print_step('FreeType2 installed at: {}'.format(FREETYPE_DIR))


# ------------------------------------------------------------------------
# Build libPNG
# ------------------------------------------------------------------------
def build_libpng():
    print_header('BUILDING LIBPNG')

    libpng_src_dir = os.path.join(BUILD_DIR, f'libpng-{LIBPNG_VERSION}')
    if os.path.exists(libpng_src_dir):
        shutil.rmtree(libpng_src_dir)

    url = f'https://download.sourceforge.net/libpng/libpng-{LIBPNG_VERSION}.tar.xz'
    file = download(url, BUILD_DIR)
    extract(file, BUILD_DIR)

    conf_cmd  = ['./configure',
                 '--prefix={}'.format(LIBPNG_DIR),
                 '--disable-static',
                 # '--enable-static',
                 # '--disable-shared',
                 '--enable-shared',
                 ]
    conf_env = os.environ.copy()
    conf_env['CFLAGS']   = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'

    build_cmd = ['make', 'install']

    if os.path.exists(LIBPNG_DIR):
        shutil.rmtree(LIBPNG_DIR)

    configure(conf_cmd,              env=conf_env, cwd=libpng_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=libpng_src_dir)

    print_step('libPNG installed at: {}'.format(LIBPNG_DIR))


# ------------------------------------------------------------------------
# Build libPNG
# ------------------------------------------------------------------------
def build_libjpeg():
    print_header('BUILDING LIBJPEG(-TURBO)')

    libjpeg_src_dir = os.path.join(BUILD_DIR, f'libjpeg-turbo-{LIBJPEG_TURBO_VERSION}')
    if os.path.exists(libjpeg_src_dir):
        shutil.rmtree(libjpeg_src_dir)

    url = f'https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/{LIBJPEG_TURBO_VERSION}/libjpeg-turbo-{LIBJPEG_TURBO_VERSION}.tar.gz'
    file = download(url, BUILD_DIR)
    extract(file, BUILD_DIR)

    # conf_cmd  = ['./configure',
    #              '--prefix={}'.format(LIBJPEG_TURBO_DIR),
    #              '--disable-static',
    #              # '--enable-static',
    #              # '--disable-shared',
    #              '--enable-shared',
    #              ]
    conf_cmd  = [ 'cmake',
                 '-G', 'Unix Makefiles',
                 '-DENABLE_STATIC=ON',
                 '-DENABLE_SHARED=OFF',
                 # '-DENABLE_STATIC=OFF',
                 # '-DENABLE_SHARED=ON',
                 '-DWITH_JPEG8=ON',
                 '-DCMAKE_INSTALL_PREFIX={}'.format(LIBJPEG_TURBO_DIR),
                 '-DCMAKE_POSITION_INDEPENDENT_CODE=1',
                 '.'
                ]
    conf_env = os.environ.copy()
    conf_env['CFLAGS']   = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'

    build_cmd = ['make', 'install']

    if os.path.exists(LIBJPEG_TURBO_DIR):
        shutil.rmtree(LIBJPEG_TURBO_DIR)

    configure(conf_cmd,              env=conf_env, cwd=libjpeg_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=libjpeg_src_dir)

    print_step('libJPEG(-turbo) installed at: {}'.format(LIBJPEG_TURBO_DIR))


# ------------------------------------------------------------------------
# Build libxft
# ------------------------------------------------------------------------
def build_libxft():
    print_header('BUILDING LIBXFT...')

    libxft_src_dir = os.path.join(BUILD_DIR, f'libXft-{LIBXFT_VERSION}')
    if os.path.exists(libxft_src_dir):
        shutil.rmtree(libxft_src_dir)

    # Download and extract libxft
    url = f'https://xorg.freedesktop.org/releases/individual/lib/libXft-{LIBXFT_VERSION}.tar.xz'
    file = download(url, BUILD_DIR)
    extract(file, BUILD_DIR)

    # Configure and build libxft with freetype dependency
    conf_cmd = [
        './configure',
        '--prefix={}'.format(LIBXFT_DIR),
        '--disable-static',
        '--enable-shared',
        # '--enable-static',
        '--with-freetype-config={}/bin/freetype-config'.format(FREETYPE_DIR),  # Link freetype
        '--with-fontconfig'  # Fontconfig is commonly used with Xft for font handling
    ]
    conf_env  = os.environ.copy()
    conf_env['FREETYPE_LIBS'  ] = os.path.join(FREETYPE_DIR, 'lib')
    conf_env['PKG_CONFIG_PATH'] = '{}{}{}'.format(os.path.join(FREETYPE_DIR,   'lib', 'pkgconfig'), os.pathsep,
                                                  os.path.join(FONTCONFIG_DIR, 'lib', 'pkgconfig'))

    build_cmd = ['make', 'install']

    if os.path.exists(LIBXFT_DIR):
        shutil.rmtree(LIBXFT_DIR)

    configure(conf_cmd,              env=conf_env, cwd=libxft_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=libxft_src_dir)


# ------------------------------------------------------------------------
# Build Fontconfig (Static)
# ------------------------------------------------------------------------
def build_fontconfig():
    print_header('Building Fontconfig...')
    fontconfig_url = f'https://www.freedesktop.org/software/fontconfig/release/fontconfig-{FONTCONFIG_VERSION}.tar.gz'

    # Download and extract Fontconfig
    fontconfig_file = download(fontconfig_url, BUILD_DIR)
    extract(fontconfig_file, BUILD_DIR)

    fontconfig_src_dir = os.path.join(BUILD_DIR, f'fontconfig-{FONTCONFIG_VERSION}')
    conf_cmd = [
        './configure',
        f'--prefix={FONTCONFIG_DIR}',
        '--enable-static',
        '--enable-shared'
        # '--disable-shared'
    ]

    conf_env = os.environ.copy()
    conf_env['CFLAGS'  ] = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'
    conf_env['PKG_CONFIG_PATH'] = '{}'.format(os.path.join(FREETYPE_DIR, 'lib'  , 'pkgconfig'))
    conf_env['PATH'    ] = '{}{}{}'.format(   os.path.join(INSTALL_DIR , 'gperf', 'bin'), os.pathsep, conf_env['PATH'])

    build_cmd = ['make', 'install']

    if os.path.exists(FONTCONFIG_DIR):
        shutil.rmtree(FONTCONFIG_DIR)

    # Configure, build, and install Fontconfig
    configure(conf_cmd,              env=conf_env, cwd=fontconfig_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=fontconfig_src_dir)


# ------------------------------------------------------------------------
# Build FLTK
# ------------------------------------------------------------------------
def build_fltk():
    print_header('BUILDING FLTK')

    # Add system libjpeg-turbo-devel to PATH
    # lfs = 'yes'
    # lib = f'libjpeg-turbo-devel-1.2.90-8.el7.{arch}'
    # subprocess.run([f'curl https://{LIB_GITLAB}/api/v4/projects/{LIB_PROJECT}/repository/files/{lib}.tar.gz/raw?lfs={lfs} --output {lib}.tar.gz'],  # noqa: E501
    #                check=True,
    #                cwd=BUILD_DIR,
    #                shell=True)
    # extract(os.path.join(BUILD_DIR, f'{lib}.tar.gz'), BUILD_DIR)
    #
    # # Move files into common directory
    # shutil.copytree(os.path.join(BUILD_DIR, lib, 'usr', 'include/')  , '/io/build/include'  , dirs_exist_ok=True)
    #
    # # Add system libpng-devel to PATH
    # lib = f'libpng-devel-1.5.13-8.el7.{arch}'
    # subprocess.run([f'curl https://{LIB_GITLAB}/api/v4/projects/{LIB_PROJECT}/repository/files/{lib}.tar.gz/raw?lfs={lfs} --output {lib}.tar.gz'],  # noqa: E501
    #                check=True,
    #                cwd=BUILD_DIR,
    #                shell=True)
    # extract(os.path.join(BUILD_DIR, f'{lib}.tar.gz'), BUILD_DIR)
    #
    # # Move files into common directory
    # shutil.copytree(os.path.join(BUILD_DIR, lib, 'usr', 'include/')  , '/io/build/include'  , dirs_exist_ok=True)
    #
    # os.environ['CPPFLAGS'] = '-I{}'.format( os.path.join(BUILD_DIR, 'include'))
    # subprocess.check_call(['yum', 'install', '-y', 'libjpeg-turbo-devel', 'libpng-devel'])

    fltk_src_dir = os.path.join(BUILD_DIR, f'fltk-{FLTK_VERSION}')
    if os.path.exists(fltk_src_dir):
        shutil.rmtree(fltk_src_dir)

    url  = f'https://www.fltk.org/pub/fltk/{FLTK_VERSION}/fltk-{FLTK_VERSION}-source.tar.gz'
    # url  = 'https://www.fltk.org/pub/fltk/snapshots/fltk-1.4.x-20241011-013e939c.tar.gz'
    file = download(url, BUILD_DIR)
    extract(file, BUILD_DIR)

    # if os.path.isdir(f'{BUILD_DIR}/fltk-1.4.0'):
    #     shutil.rmtree(f'{BUILD_DIR}/fltk-1.4.0')
    # subprocess.run(['mv', f'{BUILD_DIR}/fltk-1.4.x-20241011-013e939c', f'{BUILD_DIR}/fltk-1.4.0'], check=True)

    conf_cmd = [
        './configure',
        '--prefix={}'.format(FLTK_DIR),
        # '--includedir={}'.format(os.path.join(BUILD_DIR, 'include')),
        # '--enable-static',
        '--disable-shared',
        # '--enable-shared',
        # '--disable-xft',
        # '--disable-xfixes'
        # '--disable-fluid',
        '--enable-localpng=no',
        '--enable-localjpeg=no',
        '--enable-localzlib=no',
    ]
    conf_env = os.environ.copy()
    conf_env['CFLAGS']   = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'

    conf_env['CPPFLAGS'       ] = '-I{}'   .format(os.path.join(LIBPNG_DIR       , 'include'))
    conf_env['CPPFLAGS'       ] = '-I{} {}'.format(os.path.join(LIBXFT_DIR       , 'include'), conf_env['CPPFLAGS'])
    conf_env['CPPFLAGS'       ] = '-I{} {}'.format(os.path.join(FONTCONFIG_DIR   , 'include'), conf_env['CPPFLAGS'])
    conf_env['CPPFLAGS'       ] = '-I{} {}'.format(os.path.join(LIBJPEG_TURBO_DIR, 'include'), conf_env['CPPFLAGS'])
    conf_env['LDFLAGS'        ] = '-L{}'   .format(os.path.join(LIBPNG_DIR       , 'lib'    ))
    conf_env['LDFLAGS'        ] = '-L{} {}'.format(os.path.join(LIBXFT_DIR       , 'lib'    ), conf_env['LDFLAGS' ])
    conf_env['LDFLAGS'        ] = '-L{} {}'.format(os.path.join(FONTCONFIG_DIR   , 'lib'    ), conf_env['LDFLAGS' ])
    conf_env['LDFLAGS'        ] = '-L{} {}'.format(os.path.join(LIBJPEG_TURBO_DIR, 'lib64'  ), conf_env['LDFLAGS' ])
    if 'LD_LIBRARY_PATH' in conf_env:
        conf_env['LD_LIBRARY_PATH'] = '{}{}{}'.format(os.path.join(LIBPNG_DIR,        'lib'  ), os.pathsep,
                                                      conf_env['LD_LIBRARY_PATH'])
    else:
        conf_env['LD_LIBRARY_PATH'] = '{}'    .format(os.path.join(LIBPNG_DIR,        'lib'  ))
    conf_env['LD_LIBRARY_PATH'] = '{}{}{}'    .format(os.path.join(LIBXFT_DIR,        'lib'  ), os.pathsep,
                                                      conf_env['LD_LIBRARY_PATH'])
    conf_env['LD_LIBRARY_PATH'] = '{}{}{}'    .format(os.path.join(LIBJPEG_TURBO_DIR, 'lib64'), os.pathsep,
                                                      conf_env['LD_LIBRARY_PATH'])
    conf_env['LD_LIBRARY_PATH'] = '{}{}{}'    .format(os.path.join(FONTCONFIG_DIR,    'lib'), os.pathsep,
                                                      conf_env['LD_LIBRARY_PATH'])

    conf_env['FREETYPE_LIBS'  ] = os.path.join(FREETYPE_DIR, 'lib')
    conf_env['LIBPNG_PATH'    ] = '{}'.format(LIBPNG_DIR)
    conf_env['PKG_CONFIG_PATH'] = '{}{}{}'.format(os.path.join(FREETYPE_DIR,      'lib',   'pkgconfig'), os.pathsep,
                                                  os.path.join(FONTCONFIG_DIR,    'lib',   'pkgconfig'))
    conf_env['PKG_CONFIG_PATH'] = '{}{}{}'.format(os.path.join(LIBPNG_DIR,        'lib',   'pkgconfig'), os.pathsep,
                                                  conf_env['PKG_CONFIG_PATH'])
    conf_env['PKG_CONFIG_PATH'] = '{}{}{}'.format(os.path.join(LIBXFT_DIR,        'lib',   'pkgconfig'), os.pathsep,
                                                  conf_env['PKG_CONFIG_PATH'])
    conf_env['PKG_CONFIG_PATH'] = '{}{}{}'.format(os.path.join(LIBJPEG_TURBO_DIR, 'lib64', 'pkgconfig'), os.pathsep,
                                                  conf_env['PKG_CONFIG_PATH'])

    build_cmd = ['make', 'install']

    if os.path.exists(FLTK_DIR):
        shutil.rmtree(FLTK_DIR)

    configure(conf_cmd,              env=conf_env, cwd=fltk_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=fltk_src_dir)

    print_step('FLTK installed at: {}'.format(FLTK_DIR))


# ------------------------------------------------------------------------
# Ensure gperf is installed
# ------------------------------------------------------------------------
def build_gperf():
    print_header('Building gperf...')
    gperf_url = f"https://ftp.gnu.org/pub/gnu/gperf/gperf-{GPERF_VERSION}.tar.gz"

    gperf_file = download(gperf_url, BUILD_DIR)
    extract(gperf_file, BUILD_DIR)

    gperf_src_dir = os.path.join(BUILD_DIR, f'gperf-{GPERF_VERSION}')
    conf_cmd = [
        './configure',
        f'--prefix={INSTALL_DIR}/gperf',
        '--enable-static',
        # '--enable-shared',
        '--disable-shared',
    ]

    conf_env = os.environ.copy()
    conf_env['CFLAGS']   = '-fPIC'
    conf_env['CXXFLAGS'] = '-fPIC'

    build_cmd = ['make', 'install']

    if os.path.exists(os.path.join(INSTALL_DIR, 'gperf')):
        shutil.rmtree(os.path.join(INSTALL_DIR, 'gperf'))

    # Configure, build, and install Fontconfig
    configure(conf_cmd,              env=conf_env, cwd=gperf_src_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=gperf_src_dir)


# ------------------------------------------------------------------------
# Build CGNS (Static)
# ------------------------------------------------------------------------
def build_cgns() -> None:
    print_header('BUILDING CGNS...')

    cgns_dir = os.path.join(BUILD_DIR, 'cgns')
    if os.path.exists(cgns_dir):
        shutil.rmtree(cgns_dir)

    git_url = 'https://github.com/CGNS/CGNS.git'
    subprocess.run(['git', 'clone'   , git_url     , cgns_dir], check=True)
    subprocess.run(['git', 'checkout', CGNS_VERSION          ], check=True, cwd=cgns_dir)

    conf_env = os.environ.copy()
    conf_env['CFLAGS']    = '-fPIC'
    conf_env['CXXFLAGS']  = '-fPIC'
    if 'PATH' in conf_env:
        conf_env['PATH']  = '{}{}{}'.format(os.path.join(HDF5_DIR), os.pathsep, conf_env['PATH'])
    else:
        conf_env['PATH']  = '{}'.format(   os.path.join(HDF5_DIR))
    conf_env['HDF5_DIR']  = '{}'.format(   os.path.join(HDF5_DIR))
    conf_env['HDF5_ROOT'] = '{}'.format(   os.path.join(HDF5_DIR))
    conf_cmd = [
        'cmake', cgns_dir,
        '-DCMAKE_INSTALL_PREFIX={}/install'.format(CGNS_DIR),
        '-DCMAKE_BUILD_TYPE=Release',
        '-DHDF5_DIR={}'.format(HDF5_DIR),
        '-DHDF5_ROOT={}'.format(HDF5_DIR),
        '-DHDF5_INCLUDE_DIR={}/include'.format(HDF5_DIR),
        '-DHDF5_LIBRARY={}/lib/libhdf5.a'.format(HDF5_DIR),
        '-DCGNS_ENABLE_HDF5=ON',
        '-DCGNS_BUILD_SHARED=OFF',
    ]

    build_cmd = ['make', 'install']

    cgns_build_dir = os.path.join(CGNS_DIR, 'build')
    # Only needed if not performing "clean" step
    if os.path.exists(CGNS_DIR):
        shutil.rmtree(CGNS_DIR)
    if os.path.exists(cgns_build_dir):
        shutil.rmtree(cgns_build_dir)
    os.makedirs(cgns_build_dir, exist_ok=True)

    configure(conf_cmd,              env=conf_env, cwd=cgns_build_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=cgns_build_dir)


# ------------------------------------------------------------------------
# Build OpenCASCADE (Static)
# ------------------------------------------------------------------------
def build_occt() -> None:
    print_header('BUILDING OPENCASCADE...')

    occ_dir = os.path.join(BUILD_DIR, 'occ')
    if os.path.exists(occ_dir):
        shutil.rmtree(occ_dir)

    git_url = 'https://git.dev.opencascade.org/repos/occt.git'
    subprocess.run(['git', 'clone'   , git_url    , occ_dir ], check=True)
    subprocess.run(['git', 'checkout', OCC_VERSION          ], check=True, cwd=occ_dir)

    conf_env = os.environ.copy()
    conf_env['CFLAGS']   = '-pthread -lrt'
    conf_env['CXXFLAGS'] = '-pthread -lrt'
    conf_cmd = [
        'cmake', occ_dir,
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_INSTALL_PREFIX={}/install'.format(OCC_DIR),
        '-DBUILD_LIBRARY_TYPE=Static',
        '-DBUILD_MODULE_DETools=OFF',
        '-DBUILD_MODULE_Draw=OFF',
        '-DBUILD_MODULE_Visualization=OFF',
        '-D3RDPARTY_TCL_INCLUDE_DIR={}/include'.format(TCL_DIR),
        '-D3RDPARTY_TK_INCLUDE_DIR={}/include'.format(TK_DIR),
        '-DUSE_FREETYPE=OFF',
        '-DUSE_TK=OFF',
    ]

    build_cmd = ['make', 'install']

    occ_build_dir = os.path.join(OCC_DIR, 'build')
    # Only needed if not performing "clean" step
    if os.path.exists(occ_build_dir):
        shutil.rmtree(occ_build_dir)
    os.makedirs(occ_build_dir, exist_ok=True)

    configure(conf_cmd,              env=conf_env, cwd=occ_build_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=occ_build_dir)


# ------------------------------------------------------------------------
# Build GLU (Static)
# ------------------------------------------------------------------------
# def build_glu():
#     print_header('Building GLU...')
#
#     glu_src_dir = os.path.join(BUILD_DIR, f'glu-{GLU_VERSION}')
#     glu_bld_dir = os.path.join(BUILD_DIR, f'glu-{GLU_VERSION}', 'build')
#     os.makedirs(glu_bld_dir, exist_ok=True)
#
#     subprocess.check_call(['yum', 'install', '-y', 'libglvnd-devel'])
#
#     # GLU requires the meson build system
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', '-t',
#                            os.path.join(BUILD_DIR, 'meson', 'meson')])
#     # commit  = '47b9353c331318a13eb050887eacfd61eb075746285f9baf7ef7de6ae235'
#     # version = '1.5.2'
#     # url  = f'https://files.pythonhosted.org/packages/55/a6/{commit}/meson-{version}-py3-none-any.whl'
#     # file = download(url, BUILD_DIR)
#     # zfil = os.path.join(BUILD_DIR, f'meson-{version}-py3-none-any.zip')
#     # shutil.move(file, zfil)
#     # # extract(zfil, BUILD_DIR)
#     # subprocess.call(['unzip', '-j', '-d', 'meson', zfil], cwd=os.path.join(BUILD_DIR))
#
#     # Download and extract libglu
#     url = f'https://mesa.freedesktop.org/archive/glu/glu-{GLU_VERSION}.tar.xz'
#     file = download(url, BUILD_DIR)
#     extract(file, BUILD_DIR)
#
#     # Configure and build glu with freetype dependency
#     setup_cmd = [
#             'meson',
#             'setup',
#             'build',
#     ]
#     conf_cmd = [
#             'meson',
#             'configure',
#             'build',
#             # '--buildtype=release',
#     ]
#     conf_env  = os.environ.copy()
#     # conf_env['FREETYPE_LIBS'  ] = os.path.join(FREETYPE_DIR, 'lib')
#     # conf_env['PKG_CONFIG_PATH'] = '{}:{}'.format(os.path.join(FREETYPE_DIR,   'lib', 'pkgconfig'),
#     #                                              os.path.join(FONTCONFIG_DIR, 'lib', 'pkgconfig'))
#
#     # Add meson to path
#     conf_env['PATH']       = '{}{}{}'.format(os.path.join(BUILD_DIR, 'meson', 'bin'), conf_env['PATH'])
#     try:
#         conf_env['PYTHONPATH'] = '{}{}{}'.format(os.path.join(BUILD_DIR, 'meson'  ), os.pathsep, conf_env['PYTHONPATH'])
#     except KeyError:
#         conf_env['PYTHONPATH'] = '{}'.format(    os.path.join(BUILD_DIR, 'meson'  ))
#     build_cmd = ['make', 'install']
#
#     configure(setup_cmd,              env=conf_env, cwd=glu_src_dir)
#     configure( conf_cmd,              env=conf_env, cwd=glu_src_dir)
#     build(    build_cmd, build_cores, env=conf_env, cwd=glu_bld_dir)


# ------------------------------------------------------------------------
# Build Gmsh with CGNS and OpenCASCADE (Static)
# ------------------------------------------------------------------------
def build_gmsh() -> None:
    print_header('BUILDING GMSH...')

    # Install system GLU
    # subprocess.check_call(['yum', 'install', '-y', 'mesa-libGLU-devel'])

    # Add system libGLU to PATH
    lfs = 'yes'
    lib = f'mesa-libGLU-9.0.0-4.el7.{arch}'
    subprocess.run([f'curl https://{LIB_GITLAB}/api/v4/projects/{LIB_PROJECT}/repository/files/{lib}.tar.gz/raw?lfs={lfs} --output {lib}.tar.gz'],  # noqa: E501
                   check=True,
                   cwd=BUILD_DIR,
                   shell=True)
    extract(os.path.join(BUILD_DIR, f'{lib}.tar.gz'), BUILD_DIR)
    os.environ['PKG_CONFIG_PATH'   ] = '{}'.format( os.path.join(BUILD_DIR, 'lib64', 'pkgconfig'))
    os.environ['CMAKE_PREFIX_PATH' ] = '{}'.format( os.path.join(BUILD_DIR))
    os.environ['CMAKE_LIBRARY_PATH'] = '{}'.format( os.path.join(BUILD_DIR, 'lib64'  ))
    os.environ['CMAKE_INCLUDE_PATH'] = '{}'.format( os.path.join(BUILD_DIR, 'include'))
    os.environ['GLU_INCLUDE_DIR'   ] = '{}'.format( os.path.join(BUILD_DIR, 'include'))
    os.environ['GLU_LIBRARY'       ] = '{}'.format( os.path.join(BUILD_DIR, 'lib64', 'libGLU.so'))

    # Move files into common directory
    if os.path.exists(os.path.join(WORK_DIR, 'build', 'lib64')):
        shutil.rmtree(os.path.join(WORK_DIR, 'build', 'lib64'))
    shutil.copytree(os.path.join(BUILD_DIR, lib, 'usr', 'lib64/')  , os.path.join(WORK_DIR, 'build', 'lib64') , dirs_exist_ok=True)

    lib = f'mesa-libGLU-devel-9.0.0-4.el7.{arch}'
    subprocess.run([f'curl https://{LIB_GITLAB}/api/v4/projects/{LIB_PROJECT}/repository/files/{lib}.tar.gz/raw?lfs={lfs} --output {lib}.tar.gz'],  # noqa: E501
                   check=True,
                   cwd=BUILD_DIR,
                   shell=True)
    extract(os.path.join(BUILD_DIR, f'{lib}.tar.gz'), BUILD_DIR)

    # Move files into common directory
    shutil.copytree(os.path.join(BUILD_DIR, lib, 'usr', 'lib64/')  , os.path.join(WORK_DIR, 'build', 'lib64'  ) , dirs_exist_ok=True, symlinks=True)
    shutil.copytree(os.path.join(BUILD_DIR, lib, 'usr', 'include/'), os.path.join(WORK_DIR, 'build', 'include') , dirs_exist_ok=True, symlinks=True)

    # Now, patch the file paths
    with open(os.path.join(BUILD_DIR, 'lib64', 'pkgconfig', 'glu.pc'), 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('/usr', os.path.join(WORK_DIR, 'build'))

    # Write the file out again
    with open(os.path.join(BUILD_DIR, 'lib64', 'pkgconfig', 'glu.pc'), 'w') as file:
        file.write(filedata)

    gmsh_dir = os.path.join(WORK_DIR, 'gmsh')
    if os.path.exists(gmsh_dir):
        shutil.rmtree(gmsh_dir)

    git_url = 'https://gitlab.onelab.info/gmsh/gmsh.git'
    subprocess.run(['git', 'clone'   , git_url, gmsh_dir], check=True)
    subprocess.run(['git', 'checkout', GMSH_STRING      ], check=True, cwd=gmsh_dir)

    conf_env = os.environ.copy()

    # We need to include a whole bunch of our libraries manually, so here we go ...
    fltk_flags = subprocess.check_output([f'{FLTK_DIR}/bin/fltk-config', '--use-images', '--ldstaticflags']).decode('utf-8')
    conf_env['LDFLAGS']   = f'-Wl,-lrt,{fltk_flags.rstrip()}'
    fltk_flags = subprocess.check_output([f'{FLTK_DIR}/bin/fltk-config', '--use-images', '--cflags'       ]).decode('utf-8')
    conf_env['CFLAGS']    = f'{fltk_flags.rstrip()}'
    fltk_flags = subprocess.check_output([f'{FLTK_DIR}/bin/fltk-config', '--use-images', '--cxxflags'     ]).decode('utf-8')
    conf_env['CXXFLAGS']  = f'{fltk_flags.rstrip()}'

    conf_env['CPPFLAGS'       ] = '-I{}'   .format(os.path.join(LIBPNG_DIR       , 'include' ))
    conf_env['CPPFLAGS'       ] = '-I{} {}'.format(os.path.join(LIBXFT_DIR       , 'include' ), conf_env['CPPFLAGS'])
    conf_env['CPPFLAGS'       ] = '-I{} {}'.format(os.path.join(FONTCONFIG_DIR   , 'include' ), conf_env['CPPFLAGS'])
    conf_env['CPPFLAGS'       ] = '-I{} {}'.format(os.path.join(LIBJPEG_TURBO_DIR, 'include' ), conf_env['CPPFLAGS'])

    conf_env['LDFLAGS'        ] = '-L{} {}'.format(os.path.join(LIBPNG_DIR       , 'lib'     ), conf_env['LDFLAGS' ])
    conf_env['LDFLAGS'        ] = '-L{} {}'.format(os.path.join(LIBXFT_DIR       , 'lib'     ), conf_env['LDFLAGS' ])
    conf_env['LDFLAGS'        ] = '-L{} {}'.format(os.path.join(FONTCONFIG_DIR   , 'lib'     ), conf_env['LDFLAGS' ])
    conf_env['LDFLAGS'        ] = '-L{} -ljpeg {}'.format(os.path.join(LIBJPEG_TURBO_DIR, 'lib64'   ), conf_env['LDFLAGS' ])
    if 'LD_LIBRARY_PATH' in conf_env:
        conf_env['LD_LIBRARY_PATH'] = '{}{}{}'.format(os.path.join(LIBPNG_DIR,        'lib'  ), os.pathsep,
                                                      conf_env['LD_LIBRARY_PATH'])
    else:
        conf_env['LD_LIBRARY_PATH'] = '{}'    .format(os.path.join(LIBPNG_DIR,        'lib'  ))
    conf_env['LD_LIBRARY_PATH'] = '{}{}{}'    .format(os.path.join(LIBXFT_DIR,        'lib'  ), os.pathsep,
                                                      conf_env['LD_LIBRARY_PATH'])
    conf_env['LD_LIBRARY_PATH'] = '{}{}{}'    .format(os.path.join(LIBJPEG_TURBO_DIR, 'lib64'), os.pathsep,
                                                      conf_env['LD_LIBRARY_PATH'])
    conf_env['LD_LIBRARY_PATH'] = '{}{}{}'    .format(os.path.join(FONTCONFIG_DIR,    'lib'  ), os.pathsep,
                                                      conf_env['LD_LIBRARY_PATH'])
    conf_env['CMAKE_INCLUDE_PATH'] = '{}{}{}'.format(os.path.join(LIBJPEG_TURBO_DIR, 'lib64', 'cmake', 'libjpeg-turbo'), os.pathsep,
                                                     conf_env['CMAKE_INCLUDE_PATH'])

    # Now, patch the file paths
    with open(os.path.join(WORK_DIR, 'gmsh', 'CMakeLists.txt'), 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace('--use-gl --use-images --ldflags', '--use-gl --use-images --ldstaticflags')

    # Write the file out again
    with open(os.path.join(WORK_DIR, 'gmsh', 'CMakeLists.txt'), 'w') as file:
        file.write(filedata)

    conf_env['PATH']      = '{}{}{}'.format(os.path.join(HDF5_DIR, 'bin'), os.pathsep, conf_env['PATH'])
    conf_env['PATH']      = '{}{}{}'.format(os.path.join(FLTK_DIR, 'bin'), os.pathsep, conf_env['PATH'])
    conf_env['PATH']      = '{}{}{}'.format(os.path.join(HDF5_DIR)       , os.pathsep, conf_env['PATH'])
    conf_env['FLTK_DIR']  = '{}'.format(    os.path.join(FLTK_DIR))
    conf_env['HDF5_DIR']  = '{}'.format(    os.path.join(HDF5_DIR))
    # conf_env['HDF5_ROOT'    ] = os.path.join(HDF5_DIR)
    # conf_env['FREETYPE_LIBS'] = os.path.join(FREETYPE_DIR, 'lib')
    conf_env['CASROOT']   = os.path.join(OCC_DIR , 'install')
    conf_env['CGNS_ROOT'] = os.path.join(CGNS_DIR, 'install')
    conf_cmd = [
        'cmake', '..',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_INSTALL_PREFIX={}'.format(INSTALL_DIR),
        '-DFREETYPE_LIBRARY={}/lib/libfreetype.a'.format(FREETYPE_DIR),
        '-DFREETYPE_INCLUDE_DIRS={}/include'.format(FREETYPE_DIR),
        '-DHDF5_INCLUDE_DIRS={}/include'.format(HDF5_DIR),
        '-DJPEG_LIBRARY={}/lib64/libjpeg.a'.format(LIBJPEG_TURBO_DIR),
        '-DJPEG_INCLUDE_DIR={}/include'.format(LIBJPEG_TURBO_DIR),
        '-DPNG_LIBRARY={}/lib/libpng.so'.format(LIBPNG_DIR),
        '-DPNG_INCLUDE_DIR={}/include'.format(LIBPNG_DIR),
        '-DPNG_PNG_INCLUDE_DIR={}/include'.format(LIBPNG_DIR),
        '-DENABLE_BUILD_LIB=ON',
        '-DENABLE_BUILD_SHARED=ON',
        '-DENABLE_MPEG_ENCODE=OFF',
        '-DENABLE_CGNS=ON',
        '-DENABLE_DOMHEX=OFF',
        '-DENABLE_FLTK=ON',
        '-DENABLE_GETDP=OFF',
        '-DENABLE_MESH=ON',
        '-DENABLE_MPEG_ENCODE=OFF',
        '-DENABLE_OCC=ON',
        '-DENABLE_OCC_STATIC=ON',
        '-DENABLE_PETSC=OFF',
        '-DENABLE_PLUGINS=ON',
        '-DENABLE_POPPLER=OFF',
        '-DENABLE_POST=ON',
        # '-DENABLE_SOLVER=OFF',
        '-DENABLE_TOUCHBAR=OFF',
        '-DENABLE_SYSTEM_CONTRIB=OFF',
        '-DENABLE_PACKAGE_STRIP=ON',
        '-DX11_ICE_LIB=OFF'  # Force disable libICE support with a very ugly hack
    ]

    build_cmd = ['make', 'install']

    gmsh_build_dir = os.path.join(gmsh_dir, 'build')
    os.makedirs(os.path.join(gmsh_dir, 'build'), exist_ok=True)

    # subprocess.run(conf_cmd, check=True, cwd=gmsh_build_dir)
    configure(conf_cmd,              env=conf_env, cwd=gmsh_build_dir)
    compile( build_cmd, build_cores, env=conf_env, cwd=gmsh_build_dir)

    print_step('Stripping generated libraries')

    subprocess.run(['strip', os.path.join('bin'  , 'gmsh'             ) ], check=True, cwd=INSTALL_DIR)
    subprocess.run(['strip', os.path.join('lib64', 'libgmsh.so.4.13.1') ], check=True, cwd=INSTALL_DIR)


def package() -> None:
    print_header('PREPARING GMSH PYTHON WHEEL...')
    PYTHON_DIR   = os.path.join(INSTALL_DIR, 'python')
    # PYTHON_WHEEL = f'gmsh-{GMSH_VERSION}-py3-none-manylinux_{arch}.whl'
    PYTHON_WHEEL = f'gmsh-{GMSH_VERSION}-py3-none-linux_{arch}.whl'
    GMESH_PY_API = os.path.join(WORK_DIR   , 'gmsh', 'api'    , 'gmsh.py')
    # GMESH_PY_DST = os.path.join(PYTHON_DIR , 'gmsh', 'gmsh.py')
    GMESH_PY_DST = os.path.join(WORK_DIR   , 'gmsh.py')

    # Create the required directory structure for packaging
    python_gmsh_dir = os.path.join(PYTHON_DIR, 'gmsh')
    os.makedirs(python_gmsh_dir, exist_ok=True)

    # Strip the Gmsh files
    subprocess.run(['strip', 'bin/gmsh'               ], check=True, cwd=INSTALL_DIR)
    subprocess.run(['strip', 'lib64/libgmsh.so.4.13.1'], check=True, cwd=INSTALL_DIR)

    # Copy the CGNS adf2hdf executable
    CGNS_ADF2HDF = os.path.join(WORK_DIR   , 'CGNS', 'install', 'bin', 'adf2hdf')
    CGNS_ADF_DIR = os.path.join(INSTALL_DIR, 'bin' , 'adf2hdf')
    print_step('Copying adf2hdf from {} to {}'.format(CGNS_ADF2HDF, CGNS_ADF_DIR))
    shutil.copy(CGNS_ADF2HDF, CGNS_ADF_DIR)

    # Copy the CGNS cgnsconvert executable
    CGNS_CGNSCON = os.path.join(WORK_DIR   , 'CGNS', 'install', 'bin', 'cgnsconvert')
    CGNS_CGN_DIR = os.path.join(INSTALL_DIR, 'bin' , 'cgnsconvert')
    print_step('Copying cgnsconvert from {} to {}'.format(CGNS_CGNSCON, CGNS_CGN_DIR))
    shutil.copy(CGNS_CGNSCON, CGNS_CGN_DIR)

    # Copy the Gmsh files to the Python directory
    # print_step('Copying Gmsh files to {}'.format(python_gmsh_dir))
    # gmsh_share_dir = os.path.join(INSTALL_DIR, 'share/gmsh')
    # shutil.copytree(gmsh_share_dir, python_gmsh_dir, dirs_exist_ok=True)

    # Copy the libJPEG library file
    # LIBJPG_NAM = 'libjpeg.so'
    # LIBJPG_LIB = os.path.join(LIBJPEG_TURBO_DIR, 'lib64', LIBJPG_NAM)
    # LIBJPG_DST = os.path.join(INSTALL_DIR      , 'lib64', LIBJPG_NAM)
    # print_step('Copying libJPEG from {} to {}'.format(LIBJPG_LIB, LIBJPG_DST))
    # if os.path.exists(os.path.join(LIBJPG_DST, LIBJPG_NAM)):
    #     shutil.rmtree(os.path.join(LIBJPG_DST, LIBJPG_NAM))
    # shutil.copy(LIBJPG_LIB, LIBJPG_DST, follow_symlinks=False)
    #
    # LIBJPG_NAM = 'libjpeg.so.62'
    # LIBJPG_LIB = os.path.join(LIBJPEG_TURBO_DIR, 'lib64', LIBJPG_NAM)
    # LIBJPG_DST = os.path.join(INSTALL_DIR      , 'lib64', LIBJPG_NAM)
    # print_step('Copying libJPEG from {} to {}'.format(LIBJPG_LIB, LIBJPG_DST))
    # if os.path.exists(os.path.join(LIBJPG_DST, LIBJPG_NAM)):
    #     shutil.rmtree(os.path.join(LIBJPG_DST, LIBJPG_NAM))
    # shutil.copy(LIBJPG_LIB, LIBJPG_DST, follow_symlinks=False)
    #
    # LIBJPG_NAM = 'libjpeg.so.62.4.0'
    # LIBJPG_LIB = os.path.join(LIBJPEG_TURBO_DIR, 'lib64', LIBJPG_NAM)
    # LIBJPG_DST = os.path.join(INSTALL_DIR      , 'lib64', LIBJPG_NAM)
    # print_step('Copying libJPEG from {} to {}'.format(LIBJPG_LIB, LIBJPG_DST))
    # if os.path.exists(os.path.join(LIBJPG_DST, LIBJPG_NAM)):
    #     shutil.rmtree(os.path.join(LIBJPG_DST, LIBJPG_NAM))
    # shutil.copy(LIBJPG_LIB, LIBJPG_DST, follow_symlinks=False)

    # Copy the Gmsh Python API file
    print_step('Copying gmsh.py from {} to {}'.format(GMESH_PY_API, GMESH_PY_DST))
    shutil.copy(GMESH_PY_API, GMESH_PY_DST)

    # Run setup to build the Python wheel
    print_step('Running setup to build the Python wheel...')

    # Write pyproject.toml file
    pyproject = """[build-system]
    requires      = ['setuptools>=42', 'wheel']
    build-backend = 'setuptools.build_meta'

    [project]
    name        = 'gmsh'
    version     = '4.13.1'
    description = 'Gmsh with updated CGNS, OpenCASCADE, and local static libraries'
    readme      = 'README.md'
    authors     = [
                    { name='Patrick Kopper', email='kopper@iag.uni-stuttgart.de' }
    ]
    requires-python = '>=3.6'
    license     = { text = 'GNU General Public License v2 (GPLv2)' }
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Operating System :: Linux',
    ]
    dependencies = []  # Add any required dependencies

    [project.urls]
    homepage    = 'https://gitlab.iag.uni-stuttgart.de/libs/python-gmsh'

    [tool.setuptools]
    packages    = ['gmsh']
    py-modules  = ['gmsh']
    package-dir = { 'gmsh' = 'gmsh' }

    [tool.setuptools.package-data]
    gmsh = [
        'share/gmsh/examples/api/*.py',
        'share/gmsh/examples/api/*.stl',
        'share/gmsh/examples/api/*.geo',
    ]

    [tool.setuptools.data-files]
    'share/gmsh' = [
        'gmsh_install/share/doc/gmsh/CHANGELOG.txt',
        'gmsh_install/share/doc/gmsh/CREDITS.txt'
    ]
    'bin' = [
        'gmsh_install/bin/adf2hdf',
        'gmsh_install/bin/cgnsconvert',
        'gmsh_install/bin/gmsh'
    ]
    'lib' = [
        'gmsh_install/lib64/libgmsh.so',
        'gmsh_install/lib64/libgmsh.so.4.13',
        'gmsh_install/lib64/libgmsh.so.4.13.1',
        # 'gmsh_install/lib64/libjpeg.so',
        # 'gmsh_install/lib64/libjpeg.so.62',
        # 'gmsh_install/lib64/libjpeg.so.62.4.0'
    ]
    """

    # Open the file in write mode and write the content
    with open(os.path.join(WORK_DIR, 'pyproject.toml'), 'w') as file:
        file.write(pyproject)

    builder = build.ProjectBuilder(WORK_DIR)
    builder.build('wheel', os.path.join(WORK_DIR, 'dist'))

    # Rename the generated Python wheel
    print_step('Renaming the wheel to {}'.format(PYTHON_WHEEL))
    shutil.move(os.path.join(WORK_DIR, 'dist', f'gmsh-{GMSH_VERSION}-py3-none-any.whl'), os.path.join(WORK_DIR, PYTHON_WHEEL))

    print_header('PYTHON WHEEL BUILD COMPLETED!')


# Run the build steps in order
if __name__ == '__main__':
    clean()
    # Build all dependencies
    build_hdf5()
    build_tcl()
    build_tk()
    build_freetype()
    build_gperf()
    build_fontconfig()
    build_libxft()
    build_libpng()
    build_libjpeg()
    build_fltk()
    build_cgns()
    build_occt()
    # build_glu()
    build_gmsh()

    # Package the Gmsh Python wheel
    package()
