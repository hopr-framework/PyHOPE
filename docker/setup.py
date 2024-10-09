import os
from setuptools import setup, find_packages

# Set up paths
package_name     = 'gmsh'
package_version  = '4.13.1'
description      = 'Gmsh with updated CGNS, OpenCASCADE, and local static libraries'
long_description = open('README.md').read() if os.path.exists('README.md') else ''
install_requires = []

# Define data files to include (examples, documentation, binary files, etc.)
data_files = [
    ('share/gmsh', [
        'gmsh_install/share/doc/gmsh/CHANGELOG.txt',
        'gmsh_install/share/doc/gmsh/CREDITS.txt',
    ]),
    ('bin', ['gmsh_install/bin/gmsh']),  # Treat the binary as a data file
    ('lib',        [
        'gmsh_install/lib64/libgmsh.so',
        'gmsh_install/lib64/libgmsh.so.4.13',
        'gmsh_install/lib64/libgmsh.so.4.13.1'
    ]),  # Treat the binary as a data file
]

# Specify packages and package data to include
packages = find_packages(where='gmsh')
package_data = {
    package_name: [
        'share/gmsh/examples/api/*.py',
        'share/gmsh/examples/api/*.stl',
        'share/gmsh/examples/api/*.geo',
    ]
}

# Define the setup configuration
setup(
    name             = package_name,
    version          = package_version,
    description      = description,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author           = 'Patrick Kopper',
    author_email     = 'kopper@iag.uni-stuttgart.de',
    url              = 'https://gitlab.iag.uni-stuttgart.de/flexi/codes/uvwxyz',
    packages         = packages,
    py_modules       = ['gmsh'],  # Include the gmsh.py file
    package_dir      = {package_name: 'gmsh'},
    package_data     = package_data,
    data_files       = data_files,  # The binary file is included here
    install_requires = install_requires,
    classifiers      = [ 'Programming Language :: Python :: 3',
                         'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
                         'Operating System :: OS Independent',
                       ],
    python_requires='>=3.6',  # Specify your Python version compatibility
)
