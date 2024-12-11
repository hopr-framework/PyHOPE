PyHOPE (Python High-Order Preprocessing Environment) is an open-source Python framework for the generation of three-dimensional unstructured high-order meshes. These meshes are needed by high-order numerical methods like Discontinuous Galerkin, Spectral Element Methods, or pFEM, in order to retain their accuracy if the computational domain includes curved boundaries.

PyHOPE has been developed by the Numerics Research Group (NRG) lead by Prof. Andrea Beck at the Institute of Aerodynamics and Gas Dynamics at the University of Stuttgart, Germany.

PyHOPE is heavily inspired by [HOPR (High Order Preprocessor)](https://github.com/hopr-framework/hopr) and shares the same input/output format. For more information and tutorials, please visit the [HOPR documentation](https://hopr.readthedocs.io).

This is a scientific project. If you use pyHOPE for publications or presentations in science, please support the project by citing our publications given at [numericsresearchgroup.org](https://numericsresearchgroup.org/publications.html).

# Installation
PyHOPE is built using standard Python packages. It is recommended to use a virtual environment for installation. The following commands create a virtual environment, here called `venv`, and install all required packages.
```
python -m venv venv
source venv/bin/activate
python -m pip install pyhope
```

# Usage
PyHOPE is invoked from the command line. Run parameters are read from a configuration file. The following output is obtained when running the example configuration file `tutorials/1-01-cartbox/parameter.ini`.
```
$ pyhope tutorials/1-01-cartbox/parameter.ini
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ P y H O P E — Python High-Order Preprocessing Environment
┃ PyHOPE version x.x.x
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
├─────────────────────────────────────────────
│ INIT PROGRAM...
│                        nThreads │ 4                               │ DEFAULT │
│ INIT PROGRAM DONE!
├─────────────────────────────────────────────
│ INIT OUTPUT...
│                     ProjectName │ 1-01-cartbox                    │ *CUSTOM │
│                    OutputFormat │ 0 [HDF5]                        │ *CUSTOM │
│                       DebugVisu │ F                               │ *CUSTOM │
│ INIT OUTPUT DONE!
├─────────────────────────────────────────────
│ INIT MESH...
│                            Mode │ 1                               │ *CUSTOM │
│                   BoundaryOrder │ 2                               │ DEFAULT │
│ INIT MESH DONE!
├─────────────────────────────────────────────
│ GENERATE MESH...
├────
│                          nZones │ 1                               │ *CUSTOM │
├── Generating zone 1
│                          Corner │ (/0.,0.,0. ,,1.,0.,0. ,,1.,1... │ *CUSTOM │
│                          nElems │ (/8,8,8/)                       │ *CUSTOM │
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
├── Generated mesh with 512 cells
├────
│ GENERATE MESH DONE!
├─────────────────────────────────────────────
│                       doSortIJK │ False                           │ DEFAULT │
├────
├── Sorting elements along space-filling curve
├────
├── Eliminating duplicate points
├────
├── Checking if surface normal vectors point outwards
├────
│             CheckSurfaceNormals │ True                            │ DEFAULT │
│             Processing Elements |█████████████████████████████████| 512/512 [100%] in 0.0s (24000.00/s) 
├─────────────────────────────────────────────
│ CONNECT MESH...
├────
│               doPeriodicCorrect │ True                            │ DEFAULT │
│                       doMortars │ True                            │ DEFAULT │
├─────────────────────────────────────────────
├────
│  Number of sides                :         3072
│  Number of inner sides          :         2688
│  Number of mortar sides (big)   :            0
│  Number of mortar sides (small) :            0
│  Number of boundary sides       :          384
│  Number of periodic sides       :            0
├────
│ CONNECT MESH DONE!
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
│   Planar-faced Hexahedra  :          512
├────
├── Writing HDF5 mesh to "1-01-cartbox_mesh.h5"
├────
│ OUTPUT MESH DONE!
┢━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┃ PyHOPE completed in [0.25 sec]
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
