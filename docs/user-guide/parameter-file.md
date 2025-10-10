# Parameter File Format

Description of the parameter file format and available options.

---

PyHOPE reads a self-hosting INI-style parameter file. The following documentation is grouped by sections describes the available parameters and their meaning. Parameters not specified in the parameter file are set to default values.

!!! info "General Remarks"
    - Boolean parameters can be set to `T`/`True` or `F`/`False` (case-insensitive)
    - Vectors are specified as comma-separated values using Fortran-like notation `(/ ... /)`
    - Arrays are flattened with each dimension separated by a double comma `,,`, so `(/ ... ,, ... /)`

??? example "Example Help Output"
    The following file is a complete example of a parameter file with all available options and their default values. The actual parameter file used for mesh generation can omit any parameters that should be set to their default values.
    ```ini
    !------------------------------------------------------------------------------
    ! Common
    !------------------------------------------------------------------------------
    nThreads                         =                      ! Number of threads for multiprocessing
    !------------------------------------------------------------------------------
    ! Output
    !------------------------------------------------------------------------------
    ProjectName                      =                      ! Name of output files
    OutputFormat                     =                 HDF5 ! Mesh output format
    DebugVisu                        =                    F ! Launch the GMSH GUI to visualize the mesh
    !------------------------------------------------------------------------------
    ! Mesh
    !------------------------------------------------------------------------------
    Mode                             =                      ! Mesh generation mode (1 - Internal, 3 - External [MeshIO])
    nZones                           =                      ! Number of mesh zones
    Corner                           =                      ! Corner node positions: (/ x_1,y_1,z_1,, x_2,y_2,z_2,, ... ,, x_8,y_8,z_8/)
    X0                               =                      ! Origin of a zone. Equivalent to a corner node.
    DX                               =                      ! Extension of the zone in each spatial direction starting from the origin X0 corner node
    nElems                           =                      ! Number of elements in each direction
    ElemType                         =           hexahedron ! Element type
    EliminateNearDuplicates          =                    T ! Enables elimination of near duplicate points
    Filename                         =                      ! Name of external mesh file
    MeshIsAlreadyCurved              =                    F ! Enables mesh agglomeration
    NGeo                             =                    1 ! Order of spline-reconstruction for curved surfaces
    BoundaryOrder                    =                    2 ! Order of spline-reconstruction for curved surfaces (legacy)
    vv                               =                      ! Vector for periodic BC
    doPeriodicCorrect                =                    F ! Enables periodic correction
    doSortIJK                        =                    F ! Sort the mesh elements along the I,J,K directions
    doSplitToHex                     =                    F ! Split simplex elements into hexahedral elements
    doMortars                        =                    T ! Enables mortars
    !------------------------------------------------------------------------------
    ! Boundaries
    !------------------------------------------------------------------------------
    BoundaryName                     =                      ! Name of domain boundary
    BoundaryType                     =                      ! (/ Type, curveIndex, State, alpha /)
    BCIndex                          =                      ! Index of BC for each boundary face
    !------------------------------------------------------------------------------
    ! Mesh Checks
    !------------------------------------------------------------------------------
    CheckElemJacobians               =                    T ! Check the Jacobian and scaled Jacobian for each element
    CheckConnectivity                =                    T ! Check if the side connectivity, including correct flip
    CheckWatertightness              =                    T ! Check if the mesh is watertight
    CheckSurfaceNormals              =                    T ! Check if the surface normals point outwards
    !------------------------------------------------------------------------------
    ! Transformation
    !------------------------------------------------------------------------------
    meshScale                        =                  1.0 ! Scale the mesh
    meshTrans                        =         (/0.,0.,0./) ! Translate the mesh
    meshRot                          = (/1.,0.,0.,0.,1.,0.,0.,0.,1./) ! Rotate the mesh around rotation center
    meshRotCenter                    =         (/0.,0.,0./) ! Rotate the mesh around rotation center
    MeshPostDeform                   =                 none ! Mesh post-transformation template
    !------------------------------------------------------------------------------
    ! Stretching
    !------------------------------------------------------------------------------
    StretchType                      =            (/0,0,0/) ! Stretching type for individual zone per spatial direction.
    Factor                           =                      ! Stretching factor of zone for geometric stretching for each spatial direction.
    l0                               =                      ! Smallest desired element in zone per spatial direction.
    DXmaxToDXmin                     =                      ! Ratio between the smallest and largest element per spatial direction
    !------------------------------------------------------------------------------
    ! Finite Element Method (FEM) Connectivity
    !------------------------------------------------------------------------------
    doFEMConnect                     =                    F ! Generate finite element method (FEM) connectivity
    ```

## Common

Common parameters for all mesh generation modes.

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `nThreads`                               | int                                | `nProcs-2`                            | 0, ≥1                                  | Maximum number of worker threads for multiprocessing. Use the number of physical cores you want active during mesh generation/processing. Set to `0` to disable multiprocessing globally. |

## Output

Output parameter controlling the output file name and format.

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `ProjectName`                            | string                             | —                                     | string                                 | Base name for output files. If omitted, a tool- or input-derived name may be used. |
| `OutputFormat`                           | int &#124; string                  | `HDF5`                                | `HDF5`, `VTK`, `GMSH` (experimental)   | Mesh output format for the generated mesh. `HDF5` is the native PyHOPE format. `VTK` and `GMSH` are provided as output formats for interoperability with other tools. |
| `DebugMesh`                              | bool                               | `F`                                   | `T` &#124; `F`                         | Emit an auxiliary XDMF mesh for low-order visualization and troubleshooting.         |
| `DebugVisu`                              | bool                               | `F`                                   | `T` &#124; `F`                         | Launch the Gmsh GUI after mesh generation for interactive validation and spot checks. |

## Mesh

Mesh parameters controlling the respective mesh generation mode.

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `Mode`                                   | int &#124; string                  | —                                    | `1` (internal) &#124; `3` (external)    | Mesh generation mode. `1` builds meshes using the internal mesh generator. `3` reads and converts meshes from external mesh generators. |
| `NGeo`                                   | int                                | `1`                                   | ≥ 1                                    | Polynomial order used for representation of curved elements. Higher orders improve geometric fidelity. |
| `BoundaryOrder`                          | int                                | `2`                                   | ≥ 2                                    | Legacy parameter for polynomial order used for representation of curved elements. Prefer `NGeo` where applicable; kept for backward compatibility. |
| `doSortIJK`                              | bool                               | `F`                                   | `T` &#124; `F`                         | Reorder elements primarily along I, then J, then K instead of the default space-filling Hilbert curve. |

### Internal Mesh Generator (Mode = `1` | `internal`)

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `nZones`                                 | int                                | —                                     | ≥ 1                                    | Number of mesh zones (logical blocks). Multi-zone meshes allow different discretizations or stretching per region. |
| `Corner`                                 | vector                             | —                                     | `(/ x1,y1,z1,, …,, x8,y8,z8 /)`        | Corner node positions for a mesh block. Provide the eight 3D corner coordinates to define the block extents. |
| `X0`                                     | vector                             | —                                     | `(/ x, y, z /)`                        | Zone origin (equivalent to first corner node). Must be combined with `DX` to define the zone extents. |
| `DX`                                     | vector                             | —                                     | `(/ Δx, Δy, Δz /)`                     | Zone extents along each axis. Must be combined with `X0` to define the zone extents. |
| `nElems`                                 | vector (int)                       | —                                     | `(/ Nx, Ny, Nz /)`                     | Number of elements per axis within the zone. Controls resolution of the transfinite mesh along each direction before stretching/splitting. |
| `ElemType`                               | string                             | `108`                                 | `104` &#124; `105` &#124; `106` &#124; `108` | Element topology. Hexahedral (`108`) elements are default. Other element types are identified by the number of corner nodes and generated by splitting hexahedral elements. |
| `StretchType`                            | vector (int)                       | `(/.../)`                             | `(/ Sx, Sy, Sz /)`                     | Stretching scheme per axis. `0` disables stretching; other positive integers select supported schemes (`constant`, `factor`, `ratio`, `combination`). |
| `Factor`                                 | vector (float)                     | —                                     | `(/ fx, fy, fz /)`                     | Controls element growth/decay per axis for geometric schemes. Values > 1 produce growth; between 0 and 1 produce decay. |
| `l0`                                     | vector (float)                     | —                                     | `(/ lx, ly, lz /)`                     | Target smallest element size per axis to resolve small-scale near-boundary features. |
| `DXmaxToDXmin`                           | vector or float                    | —                                     | ratio ≥ 1                              | Ratio between the largest and smallest element sizes per axis. |

### External Mesh Generator (Mode = `3` | `external`)

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `Filename`                               | string                             | —                                     | path                                   | Path to an external mesh when `Mode = 3`. PyHOPE might temporarily convert the mesh to an intermediate format for processing. |
| `MeshIsAlreadyCurved`                    | bool                               | `F`                                   | `T` &#124; `F`                         | Indicates that the input geometry is curved. Enables mesh agglomeration if `NGeo>1`.

## Boundary Conditions

Parameters controlling boundary conditions and zone interfaces.

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `BoundaryName`                           | string                             | —                                     | identifier                             | Human-readable name for a domain boundary. Use consistent names for BC references when using external meshes. |
| `BoundaryType`                           | tuple                              | —                                    | `(/ Type, curveIndex, State, alpha /)`  | Boundary condition specification. Must be compatible with the boundary conditions in the DG solver. |
| `BCIndex`                                | array (int)                        | —                                    | per-face indices                        | Boundary condition index for each boundary face. Must match the block face ordering. |
| `vv`                                     | vector                             | —                                     | `(/ vx, vy, vz /)`                     | Periodic boundary offset vector. Translates one periodic face to the other to enforce periodic matching. |
| `doMortars`                              | bool                               | `T`                                   | `T` &#124; `F`                         | Enable mortar interfaces (non-conforming face coupling with hanging nodes) between elements. |

## Transformation

Transform mesh coordinates with translation, rotation, and scaling, or apply a user-defined transformation.

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `meshScale`                              | float                              | `1.0`                                 | > 0                                    | Uniform scaling factor for all coordinates. |
| `meshTrans`                              | vector                             | `(/.../)`                             | `(/ Δx, Δy, Δz /)`                     | Translation applied for all coordinates. |
| `meshRot`                                | matrix (3×3)                       | `(/.../)`                             | row-major 9-tuple                      | Rotation matrix applied about `meshRotCenter`. Provide nine values forming an orthonormal rotation matrix. |
| `meshRotCenter`                          | vector                             | `(/.../)`                             | `(/ x, y, z /)`                        | Pivot point for rotation. Set to the model's logical pivot/centroid to avoid unintended offsets. |
| `MeshPostDeform`                         | string                             | `none`                                | template name                          | Optional post-transform deformation template applied after scale/translate/rotate. |

## Post-Processing

Post-processing options for mesh cleanup and optimization.

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `EliminateNearDuplicates`                | bool                               | `T`                                   | `T` &#124; `F`                         | Merge nodes closer than an internal tolerance to avoid sliver elements and improve connectivity consistency. |
| `doPeriodicCorrect`                      | bool                               | `F`                                   | `T` &#124; `F`                         | Apply corrective adjustments for periodic matching after initial pairing. |
| `doSplitToHex`                           | bool                               | `F`                                   | `T` &#124; `F`                         | Split simplex elements into hexahedra when supported to homogenize element types for hex-only toolchains. |

## Mesh Checks

Perform mesh quality checks and report statistics.

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `CheckElemJacobians`                     | bool                               | `T`                                   | `T` &#124; `F`                         | Compute Jacobian and scaled Jacobian for each element to detect inverted or poorly shaped cells. |
| `CheckConnectivity`                      | bool                               | `T`                                   | `T` &#124; `F`                         | Verify side connectivity and correct face orientation/flip between adjacent elements to ensure topological consistency. |
| `CheckWatertightness`                    | bool                               | `T`                                   | `T` &#124; `F`                         | Ensure there are no gaps/holes between connected elements. |
| `CheckSurfaceNormals`                    | bool                               | `T`                                   | `T` &#124; `F`                         | Confirm surface normals point outward consistently. |

## FEM Connectivity

Create explicit connectivity arrays for finite element method (FEM) solvers that require node/face-to-element connectivity.

| <div style="width:200px">Parameter</div> | <div style="width:90px">Type</div> | <div style="width:50px">Default</div> | <div style="width:200px">Allowed</div> | Explanation                            |
| ---------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------------- | ---------------------------------------- |
| `doFEMConnect`                           | bool                               | `F`                                   | `T` &#124; `F`                         | Generate FEM connectivity (node/face-to-element). Enable when exporting to FEM solvers that require extended connectivity information. |
