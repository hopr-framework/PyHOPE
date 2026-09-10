# Mesh Checks / Visualization

Finding and resolving issues

---

Mesh generation might fail due to topological or accuracy issues in the input file. PyHOPE performs comprehensive mesh checks to ensure the generated mesh is valid and high-order accurate. These checks include the following and can be individually toggled in the [parameter file](parameter-file.md#mesh-checks).

- **Connectivity:** Verifies side connectivity and face orientation/flip
- **Watertightness:** Ensures no gaps or holes exist between connected elements
- **Surface Normals:** Confirms that surface normals point outward
- **Internal Boundaries:** Ensures internal faces do not have multiple boundary conditions attached
- **Element Jacobians:** Detects inverted or poorly shaped cells

Additionally, PyHOPE provides two visual debugging modes to help identify and resolve meshing issues.

## Gmsh Visualization
Directly after creating the intermediate Gmsh objects and prior to applying mesh transformation or computing connectivity, PyHOPE can launch an interactive Gmsh GUI by setting the following parameter.
```ini
DebugVisu = T
```
This presents a live view of the raw input geometry without mesh deformations applied. Any adjustments made within this view propagate directly into the mesh processed by PyHOPE. Use this mode to catch topological issues early in the setup process.

## XDMF Visualization
PyHOPE can generate a low-order visualization in [XDMF](https://www.xdmf.org) format, either when a mesh check fails or upon successful mesh generation, by setting the following parameter.
```
DebugMesh = T
```

PyHOPE outputs `<projectname>_DebugMesh.xdmf` with the accompanying heavy data stored in an adjacent `<projectname>_DebugMesh.h5` HDF5 file. XDMF files can be read, among others, with [ParaView](https://www.paraview.org), a free, open-source, and cross-platform visualization software. 

Data in the XDMF file is stored as a multiblock dataset containing volume, surface, and (when `doFEMConnect = T` is enabled) edge and vertex information. `[Error]` datasets are automatically created when a mesh check fails, alongside terminal output:
```ini
│ CONNECT MESH...
├────
│               doPeriodicCorrect │ False                           │ DEFAULT │
│                       doMortars │ True                            │ DEFAULT │
│                 doMortarRebuild │ 1 [auto]                        │ DEFAULT │
│                Processing Sides |█████████████████████████████████| 144/144 [100%] in 0.0s (24000.00/s)
│ 
│ ⚠ WARNING ⚠ ┃ > Element 1, Side z-, Side 1
│ ⚠ WARNING ⚠ ┃ - Coordinates  : [   0.00000000    0.00000000    0.00000000]
│ ⚠ WARNING ⚠ ┃ - Coordinates  : [   0.50000000    0.00000000    0.00000000]
│ ⚠ WARNING ⚠ ┃ - Coordinates  : [   0.00000000    0.33333333    0.00000000]
│ ⚠ WARNING ⚠ ┃ - Coordinates  : [   0.50000000    0.33333333    0.00000000]
│  
│ ⚠ WARNING ⚠ ┃ > Element 1, Side y-, Side 2
│ ⚠ WARNING ⚠ ┃ - Coordinates  : [   0.00000000    0.00000000    0.00000000]
│ ⚠ WARNING ⚠ ┃ - Coordinates  : [   0.00000000    0.00000000    0.25000000]
│ ⚠ WARNING ⚠ ┃ - Coordinates  : [   0.50000000    0.00000000    0.00000000]
│ ⚠ WARNING ⚠ ┃ - Coordinates  : [   0.50000000    0.00000000    0.25000000]
│ 
├── Writing XDMF mesh to "1-02-cartbox_periodic_DebugMesh.xdmf"
│🛈 Reason: Detected 2 / 144 erroneous sides, written to "Surface [Error]"

 !! Could not connect 2 / 144 sides !!
```
To isolate failing regions inside ParaView, apply the **Extract Block** filter to the XDMF reader source. Select the corresponding error blocks (e.g., `Surface [Error]`) in the pipeline properties to render only the problem geometry.

!!! info
    Ensure the block data is checked in the XDMF reader properties before applying the filter, otherwise the block will not be visible in the pipeline.
