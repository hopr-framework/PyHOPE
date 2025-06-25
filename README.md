PyHOPE (Python High-Order Preprocessing Environment) is an open-source Python framework for the generation of three-dimensional unstructured high-order meshes. These meshes are needed by high-order numerical methods like Discontinuous Galerkin, Spectral Element Methods, or pFEM, in order to retain their accuracy if the computational domain includes curved boundaries.

PyHOPE has been developed by the Numerics Research Group (NRG) lead by Prof. Andrea Beck at the Institute of Aerodynamics and Gas Dynamics at the University of Stuttgart, Germany.

PyHOPE is heavily inspired by [HOPR (High Order Preprocessor)](https://github.com/hopr-framework/hopr) and shares the same input/output format. For more information and tutorials, please visit the [HOPR documentation](https://hopr.readthedocs.io). Furthermore, PyHOPE utilizes [Gmsh](https://gmsh.info) for the initial mesh generation and conversion before converting it to its internal representation.

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

2.  **Install PyHOPE**  
    PyHOPE is installed using `pip`, the Python package installer. This command fetches the PyHOPE package and its dependencies from PyPI (Python Package Index) and installs them.
    ```bash
    python -m pip install pyhope
    ```

3. **Run PyHOPE**
    PyHOPE is available as a command-line tool. After installation, its functionalities can be accessed directly from the terminal.
    ```bash
    pyhope --help
    ```

    > 🛈 Remark: For new shell sessions, the virtual environment must be re-sourced using `source venv/bin/activate` before using `pyhope` commands.

# Usage
PyHOPE is invoked from the command line. Run parameters are read from a configuration file. The following output is obtained when running the example configuration file `tutorials/1-01-cartbox/parameter.ini`.
```
$ pyhope tutorials/1-01-cartbox/parameter.ini
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ P y H O P E — Python High-Order Preprocessing Environment
┃ PyHOPE version x.x.x
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
│ INIT PROGRAM...
│                        nThreads │ 10                              │ DEFAULT │
├─────────────────────────────────────────────
│ INIT OUTPUT...
│                     ProjectName │ 1-01-cartbox                    │ *CUSTOM │
│                    OutputFormat │ 0 [HDF5]                        │ *CUSTOM │
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
│                          Corner │ (/0.,0.,0. ,,1.,0.,0. ,,1.,1... │ *CUSTOM │
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
│                              vv │ (/1., 0., 0./)                  │ *CUSTOM │
│                              vv │ (/0., 1., 0./)                  │ *CUSTOM │
│                              vv │ (/0., 0., 1./)                  │ *CUSTOM │
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
│                       doSortIJK │ False                           │ DEFAULT │
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
┢━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ PyHOPE completed in [0.25 sec]
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
