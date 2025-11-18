PyHOPE (Python High-Order Preprocessing Environment) is an open-source Python framework for the generation of three-dimensional unstructured high-order meshes. These meshes are needed by high-order numerical methods like Discontinuous Galerkin, Spectral Element Methods, or pFEM, in order to retain their accuracy if the computational domain includes curved boundaries.

PyHOPE has been developed by the Numerics Research Group (NRG) lead by Prof. Andrea Beck at the Institute of Aerodynamics and Gas Dynamics at the University of Stuttgart, Germany.

PyHOPE is heavily inspired by [HOPR (High Order Preprocessor)](https://github.com/hopr-framework/hopr) and shares the same input/output format. Furthermore, PyHOPE utilizes [Gmsh](https://gmsh.info) for the initial mesh generation and conversion before converting it to its internal representation. The internal representation is loosely based on [meshio](https://github.com/nschloe/meshio) but augmented with additional information required for high-order meshes.

This is a scientific project. If you use PyHOPE for publications or presentations in science, please support the project by citing our publications given at [numericsresearchgroup.org](https://numericsresearchgroup.org/publications.html).

# Installation
PyHOPE is built using standard Python packages. You can install PyHOPE by following these steps. 

1.  **Optional: Create and activate a virtual environment**  
    Creating a virtual environment is a recommended practice to manage project dependencies. It isolates the packages required for PyHOPE and prevents potential conflicts between different package versions. To create and activate a virtual environment named `venv`, use the following commands.
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    If you choose not to use a virtual environment, skip this step and proceed directly to the installation of PyHOPE.

> [!IMPORTANT]  
> For new shell sessions, the virtual environment must be re-sourced using `source venv/bin/activate` before using `pyhope` commands.

2.  **Install PyHOPE**  
    PyHOPE is installed using `pip`, the Python package installer. This command fetches the PyHOPE package and its dependencies from PyPI (Python Package Index) and installs them.
    ```bash
    python -m pip install pyhope
    ```

# Testing
PyHOPE features internal health checks to verify that everything works as expected. The checks can be invoked directly from the terminal.
```bash
pyhope --verify [tutorials]          # Run all health checks
pyhope --verify-health               # Run Python health checks
pyhope --verify-install [tutorials]  # Run PyHOPE mesh generation checks
```

> [!NOTE]  
>  By default, PyHOPE looks for the `tutorials` directory relative to the current working directory or the git root. If neither exists, PyHOPE downloads the tests from GitHub while using available authentication methods.

> [!IMPORTANT]  
> Be aware that the PyHOPE repository uses [Git Large File Storage (LFS)](https://git-lfs.com) for some of its larger mesh files.

# Getting Help
PyHOPE help output is formatted to serve as self-hosting [INI format](https://hopr-framework.github.io/PyHOPE/user-guide/parameter-file). A list of all options and the default values can be accessed by running the following command.
```bash
pyhope --help
```

# Documentation
Refer to the [PyHOPE documentation](https://hopr-framework.github.io/PyHOPE) for the getting started guide, examples, and usage instructions. PyHOPE is heavily inspired by [HOPR (High Order Preprocessor)](https://github.com/hopr-framework/hopr) and shares the same input/output format. For technical details and the reference mesh format specification, see the [HOPR documentation](https://hopr.readthedocs.io).

# Usage
PyHOPE can either be invoked directly from the command line or used as a Python library.

## Command Line Usage
PyHOPE can be invoked from the command line. After installation, its functionalities can be accessed directly from the terminal by passing a valid configuration file.
```bash
pyhope [parameter.ini]
```
The following output is obtained when running the example configuration file `tutorials/1-01-cartbox/parameter.ini`.
```
$ pyhope tutorials/1-01-cartbox/parameter.ini
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ P y H O P E — Python High-Order Preprocessing Environment
┃ PyHOPE version x.x.x
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ INIT PROGRAM...
│                        nThreads │ nProcs-2                        │ DEFAULT │
├─────────────────────────────────────────────
│ INIT OUTPUT...
│                     ProjectName │ 1-01-cartbox                    │ *CUSTOM │
│                    OutputFormat │ 0 [HDF5]                        │ *CUSTOM │
│                       DebugMesh │ T                               │ *CUSTOM │
│                       DebugVisu │ F                               │ *CUSTOM │
├─────────────────────────────────────────────
│ INIT MESH...
│                            Mode │ 1 [Internal]                    │ *CUSTOM │
│                            NGeo │ 9                               │ *CUSTOM │
├─────────────────────────────────────────────
│ GENERATE MESH...
├────
│                          nZones │ 1                               │ *CUSTOM │
├── Generating zone 1
│                          Corner │ (/0.,0.,0.,,1.,0.,0.,,1.,1.,... │ *CUSTOM │
│                          nElems │ (/8,8,8/)                       │ *CUSTOM │
│                        ElemType │ 108 [hexahedron]                │ *CUSTOM │
│                     StretchType │ (/0,0,0/)                       │ DEFAULT │
│                         BCIndex │ (/1,2,3,4,5,6/)                 │ *CUSTOM │
├────
├── Setting boundary conditions
├────
│                    BoundaryName │ BC_zminus                       │ *CUSTOM │
│                    BoundaryType │ (/4,0,0,0/)                     │ *CUSTOM │
│                    BoundaryName │ BC_yminus                       │ *CUSTOM │
│                    BoundaryType │ (/2,0,0,0/)                     │ *CUSTOM │
│                    BoundaryName │ BC_xplus                        │ *CUSTOM │
│                    BoundaryType │ (/2,0,0,0/)                     │ *CUSTOM │
│                    BoundaryName │ BC_yplus                        │ *CUSTOM │
│                    BoundaryType │ (/2,0,0,0/)                     │ *CUSTOM │
│                    BoundaryName │ BC_xminus                       │ *CUSTOM │
│                    BoundaryType │ (/2,0,0,0/)                     │ *CUSTOM │
│                    BoundaryName │ BC_zplus                        │ *CUSTOM │
│                    BoundaryType │ (/9,0,0,0/)                     │ *CUSTOM │
├────
│                              vv │ (/1.,0.,0./)                    │ *CUSTOM │
│                              vv │ (/0.,1.,0./)                    │ *CUSTOM │
│                              vv │ (/0.,0.,1./)                    │ *CUSTOM │
├────
├── Generated mesh with 512 cells
├─────────────────────────────────────────────
├── BUILD DATA STRUCTURE...
├────
├── Removing duplicate points
├── Ensuring normals point outward
├────
│             CheckSurfaceNormals │ True                            │ DEFAULT │
│             Processing Elements |█████████████████████████████████| 512/512 [100%] in 0.0s (24000.00/s)
├────
├── Generating sides
├─────────────────────────────────────────────
│ SORT MESH...
├────
│                     MeshSorting │ 1 [SFC]                         │ DEFAULT │
├────
├── Sorting elements along space-filling curve
├─────────────────────────────────────────────
│ CONNECT MESH...
├────
│               doPeriodicCorrect │ False                           │ DEFAULT │
│                       doMortars │ True                            │ DEFAULT │
├────
│  Number of sides                :         3072
│  Number of inner sides          :         2688
│  Number of mortar sides (big)   :            0
│  Number of mortar sides (small) :            0
│  Number of boundary sides       :          384
│  Number of periodic sides       :            0
├─────────────────────────────────────────────
│ CHECK CONNECTIVITY...
├────
│               CheckConnectivity │ True                            │ DEFAULT │
│             Processing Elements |█████████████████████████████████| 512/512 [100%] in 0.0s (24000.00/s)
├─────────────────────────────────────────────
│ CHECK WATERTIGHTNESS...
├────
│             CheckWatertightness │ True                            │ DEFAULT │
│             Processing Elements |█████████████████████████████████| 512/512 [100%] in 0.0s (24000.00/s)
├─────────────────────────────────────────────
│ CHECK JACOBIANS...
├────
│              CheckElemJacobians │ True                            │ DEFAULT │
│             Processing Elements |█████████████████████████████████| 512/512 [100%] in 0.0s (24000.00/s)
├────
│ Scaled Jacobians
├─────────────────
│<0.0      │  0.00
│ 0.0-0.1  │  0.00
│ 0.1-0.2  │  0.00
│ 0.2-0.3  │  0.00
│ 0.3-0.4  │  0.00
│ 0.4-0.5  │  0.00
│ 0.5-0.6  │  0.00
│ 0.6-0.7  │  0.00
│ 0.7-0.8  │  0.00
│ 0.8-0.9  │  0.00
│>0.9-1.0  │ ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 512.00
├─────────────────
├─────────────────────────────────────────────
│ OUTPUT MESH...
├────
│         Curved Hexahedra  :          512
├────
├── Writing HDF5 mesh to "1-01-cartbox_mesh.h5"
├── Writing XDMF mesh to "1-01-cartbox_DebugMesh.xdmf"
┢━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ PyHOPE completed in [0.25 sec]
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## Python Library Usage
PyHOPE can be included in other Python libraries. PyHOPE exposes its functionally via runtime contexts defined by [Context Managers](https://docs.python.org/3/library/stdtypes.html#typecontextmanager). The following Python code loads a HOPR HDF5 mesh and derived quantities. For a complete list of currently implemented functions, see the [source code](https://github.com/hopr-framework/PyHOPE/blob/main/pyhope/__init__.py).
```python
from pyhope import Basis, Mesh
with Mesh('1-01-cartbox_mesh.h5') as m:
    elems = m.elems
    lobatto_nodes = Basis.legendre_gauss_lobatto_nodes(order=m.nGeo)
```
