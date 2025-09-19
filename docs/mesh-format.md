# Mesh Format

HOPR HDF5 curved mesh format used by PyHOPE

---

The [HOPR](https://hopr.readthedocs.io/en/latest/userguide/meshformat.html) [HDF5](http://www.hdfgroup.org) mesh format is designed for fast, scalable, parallel I/O of high-order unstructured 3D meshes. Mesh data is organized in non-overlapping arrays and packaged per element to minimize synchronization. A space-filling curve or structured ordering enables simple domain decomposition.

The following spectral element solvers have (optional) support for meshes generated in PyHOPE.

| <div style="width:120px">Framework</div> | <div style="width:100px">Language</div> | <div style="width:200px">Equation System</div> | <div style="width:300px">Reference</div> |
| ---------------------------------------- | --------------------------------------- | ---------------------------------------------- | ---------------------------------------- |
| FLEXI                                    | Fortran                                 | NSE                                            | [Krais et al., 2021](https://doi.org/10.1016/j.camwa.2020.05.004) |
| ƎLEXI                                    | Fortran                                 | NSE/MRG                                        | [Kopper et al., 2023](https://doi.org/10.1016/j.cpc.2023.108762)  |
| GALÆXI                                   | Fortran/C                               | NSE                                            | [Kurz et al., 2025](https://doi.org/10.1016/j.cpc.2024.109388)    |
| FLUXO                                    | Fortran                                 | NSE/MHD/Maxwell                                | [Rueda-Ramirez et al., 2017](https://github.com/project-fluxo/fluxo) |
| HORSES3D                                 | Fortran                                 | NSE/Cahn-Hilliard                              | [Ferrer et al., 2023](https://doi.org/10.1016/j.cpc.2023.108700)  |
| PICLas                                   | Fortran                                 | Maxwell/Poisson                                | [Fasoulas et al., 2019](https://doi.org/10.1063/1.5097638)        |

<style>.footnote {  font-size: 0.8em; }
</style>
<div class="footnote">
<p>Equation Systems: NSE - Navier-Stokes, MRG - Maxey-Riley-Gatignol, MHD - Magnetohydrodynamics</p>
</div>
<!-- +--------------+-------------+-------------------------+--------------------------------------------+ -->
<!-- | Framework    | Language    | Equation System         | Reference                                  | -->
<!-- +:=============+:============+:========================+:===========================================+ -->
<!-- | FLEXI        | Fortran     | NSE                     | [@Krais2021]                               | -->
<!-- +--------------+-------------+-------------------------+--------------------------------------------+ -->
<!-- | ƎLEXI        | Fortran     | NSE/MRG                 | [@Kopper2023]                              | -->
<!-- +--------------+-------------+-------------------------+--------------------------------------------+ -->
<!-- | GALÆXI       | Fortran/C   | NSE                     | [@Kurz2025]                                | -->
<!-- +--------------+-------------+-------------------------+--------------------------------------------+ -->
<!-- | FLUXO        | Fortran     | NSE/MHD/Maxwell         | [@RuedaRamirez2017]                        | -->
<!-- +--------------+-------------+-------------------------+--------------------------------------------+ -->
<!-- | HORSES3D     | Fortran     | NSE/Cahn-Hilliard       | [@Ferrer2023]                              | -->
<!-- +--------------+-------------+-------------------------+--------------------------------------------+ -->
<!-- | PICLas       | Fortran     | Maxwell/Poisson         | [@Fasoulas2019]                            | -->
<!-- +==============+=============+=========================+============================================+ -->
<!-- | ~*Equation Systems: NSE - Navier-Stokes, MRG - Maxey-Riley-Gatignol, MHD - Magnetohydrodynamics*~ | -->
<!-- +==============+=============+===========================+==========================================+ -->

<!-- - Supports straight and curved elements of arbitrary order (tetrahedra, pyramids, prisms, hexahedra) -->
<!-- - Parallel-friendly: contiguous per-element blocks and array hyperslabs for MPI I/O -->
<!-- - Optional FEM connectivity for solvers that require edge/vertex topology -->

## Global Attributes

| <div style="width:120px">Attribute</div> | <div style="width:240px">Type</div> | Description                                                 |
| ---------------------------------------- | ----------------------------------- | ----------------------------------------------------------- |
| `Version`                                | REAL                                | Mesh file format version                                    |
| `Ngeo`                                   | INTEGER                             | Polynomial degree of the curved element mapping             |
| `nElems`                                 | INTEGER                             | Total number of elements                                    |
| `nSides`                                 | INTEGER                             | Total number of sides (element faces)                       |
| `nNodes`                                 | INTEGER                             | Total number of nodes                                       |
| `nUniqueSides`                           | INTEGER                             | Total number of geometrically unique sides                  |
| `nUniqueNodes`                           | INTEGER                             | Total number of geometrically unique nodes                  |
| `nBCs`                                   | INTEGER                             | Number of entries in the boundary-condition list            |
| `FEMconnect`                             | `ON`/`OFF`                          | `ON` if FEM edge/vertex connectivity is present in the file |

## Data Arrays

| <div style="width:120px">Array</div> | <div style="width:80px">Type</div> | <div style="width:200px">Size</div>  | Description              |
| ------------------------------------ | ----------------------------------- | ----------------------------------- | ------------------------ |
| `ElemInfo`                           | INTEGER                             | (1:6, 1:nElems)                     | Per-element data containing element type, zone, and offsets into side and node arrays: `(ElemType, Zone, offsetIndSIDE, lastIndSIDE, offsetIndNODE, lastIndNODE)`. |
| `SideInfo`                           | INTEGER                             | (1:6, 1:nSides)                     | Per-side data stored contiguously per element range from `ElemInfo`. Fields: `(SideType, GlobalSideID, nbElemID, 10*nbLocSide+Flip, BCID, [ElemID,locSideID])`. `Flip` encodes side orientation; `GlobalSideID<0` marks slave side. |
| `NodeCoords`                         | REAL                                | (1:3, 1:nNodes)                     | Node coordinates, stored per element range from `ElemInfo`. High-order nodes included. |
| `GlobalNodeIDs`                      | INTEGER                             | (1:nNodes)                          | Globally unique node IDs aligned with `NodeCoords` |
| `BCNames`                            | STRING                              | (1:nBCs)                            | List of user-defined boundary-condition names |
| `BCType`                             | INTEGER                             | (1:4, 1:nBCs)                       | Four-integer code per boundary condition; see [boundary conditions](#boundary-conditions) below |

## Element Definitions

Elements are defined using non-unique node IDs stored in `ElemInfo` and `NodeCoords`. Each element has a type code and a zone ID. The zone ID can be used to group elements into physical zones or blocks.

### Element Types

The element encoding follows CGNS-inspired conventions. The last digit of the surface type corresponds to corner count; 3D element codes distinguish linear, bilinear, and non-linear variants.

| Element Type               | Index | Element Type               | Index | Element Type               | Index |
| -------------------------- | ----- | -------------------------- | ----- | -------------------------- | ----- |
| Tetrahedron, linear        | 104   | Tetrahedron, bilinear      | 114   | Tetrahedron, curved        | 204   |
| Pyramid, linear            | 105   | Pyramid, bilinear          | 115   | Pyramid, curved            | 205   |
| Prism/wedge, linear        | 106   | Prism/wedge, bilinear      | 116   | Prism/wedge, curved        | 206   |
| Hexahedron, linear         | 108   | Hexahedron, bilinear       | 118   | Hexahedron, curved         | 208   |

### High-Order Nodes

High-order nodes are stored in tensor-product style using (i,j,k) with uniform spacing in reference space \([-1, 1]^3\). Note that for `NGeo=1`, this node ordering differs from CGNS corner ordering for pyramids and hexahedra as the nodes `3`/`4` and `7`/`8` are swapped. The number of nodes per element depends on `NGeo` and element type.

<dl class="aligned-list">
    <dt><code>Tetrahedron</code>:</dt>
    <dd>\((Ngeo+1)(Ngeo+2)(Ngeo+3)/6\)</dd>
    <dt><code>Pyramid</code>:</dt>
    <dd>\((Ngeo+1)(Ngeo+2)(2Ngeo+3)/6\)</dd>
    <dt><code>Prism/Wedge</code>:</dt>
    <dd>\((Ngeo+1)^2(Ngeo+2)/2\)</dd>
    <dt><code>Hexahedron</code>:</dt>
    <dd>\((Ngeo+1)^3\)</dd>
</dl>

## Boundary Conditions

`BCNames` and `BCType` define the available boundary conditions. `BCID` in `SideInfo` references a 1-based index into these arrays and uses `0` for interior sides. `BCType = (BoundaryType, CurveIndex, StateIndex, PeriodicIndex)` contains four integers per boundary condition:

<dl class="aligned-list">
    <dt><code>BoundaryType</code>:</dt>
    <dd>Integer code for the BC kind. Reserved values: <code>1</code> = periodic; <code>100</code> = inner/analyze sides. Periodic and inner sides also have neighbor links defined.</dd>
    <dt><code>CurveIndex</code>:</dt>
    <dd><i>Geometry/CAD tag to distinguish BCs or trigger curving behavior. Currently unused in PyHOPE.</i></dd>
    <dt><code>BoundaryState</code>:</dt>
    <dd>User-defined index for solver reference states; not interpreted by the format.</dd>
    <dt><code>PeriodicIndex</code>:</dt>
    <dd>Used only for periodic sides; two matching BCs must share the same absolute value, with opposite signs.</dd>
</dl>

<!-- ## Parallel Read-In (Overview) -->
<!---->
<!-- The [HOPR](https://hopr.readthedocs.io/en/latest/userguide/meshformat.html) format supports efficient parallel reading with minimal communication. The [HDF5](http://www.hdfgroup.org) library allows usage of parallel MPI-I/O, enabling scalable read-in of large meshes. By default, elements in [HOPR](https://hopr.readthedocs.io/en/latest/userguide/meshformat.html) files are ordered along structured dimensions or a space-filling curve, which facilitates simple domain decomposition. This permits using the same element ordering and thus mesh file with an arbitrary number of computational domains (≥ number of elements).  -->
<!---->
<!-- Neighbor connectivity information of element sides and element node information (index and position) are stored as a package per element, allowing reading contiguous data blocks for a given range of elements. To enable a fast parallel read-in, the coordinates of the same physical nodes are stored multiple times, but can be still associated by a unique global node index. -->
<!---->
<!-- The following outlines a high-level approach to load the mesh into a supported solver: -->
<!---->
<!-- 1. **Domain Decomposition**: Distribute contiguous element ranges across ranks. The element range for each domain \(\text{dom} \in [0:\text{nDom}-1]\) is \( \text{range}(\text{dom})=[\text{offsetElem}(\text{dom})+1; \text{offsetElem}(\text{myDom}+1)] \). -->
<!-- 2. **Element Read-In**: Read the local subarray of `ElemInfo` (HDF5 hyperslabs), then compute local ranges for `SideInfo` and `NodeCoords` via `offsetInd*`/`lastInd*`. -->
<!-- 3. **Connectivity Building**: Build local connectivity; for inter-domain connections, locate the owning domain via element-range offsets (bisection over offsets). -->
<!-- 4. **Side Matching**: Group sides per neighbor domain and sort by `GlobalSideID` to obtain matching lists on both sides without communication. -->
<!-- 5. **Orientation Handling**: Master/slave and orientation are encoded consistently in `GlobalSide` sign and `Flip`/edge-orientation fields. -->

<!-- <dl class="aligned-list"> -->
<!--     <dt>Domain Decomposition:</dt> -->
<!--     <dd>Distribute contiguous element ranges across ranks. The element range for each domain \(\text{dom} \in [0:\text{nDom}-1]\) is \( \text{range}(\text{dom})=[\text{offsetElem}(\text{dom})+1; \text{offsetElem}(\text{myDom}+1)] \).</dd> -->
<!--     <dt>Element Read-In:</dt> -->
<!--     <dd>Read the local subarray of <code>ElemInfo</code> (HDF5 hyperslabs), then compute local ranges for <code>SideInfo</code> and <code>NodeCoords</code> via <code>offsetInd*</code>/<code>lastInd*</code>.</dd> -->
<!--     <dt>Connectivity Building:</dt> -->
<!--     <dd>Build local connectivity; for inter-domain connections, locate the owning domain via element-range offsets (bisection over offsets).</dd> -->
<!--     <dt>Side Matching:</dt> -->
<!--     <dd>Group sides per neighbor domain and sort by <code>GlobalSideID</code> to obtain matching lists on both sides without communication.</dd> -->
<!--     <dt>Orientation Handling:</dt> -->
<!--     <dd>Primary/replica and orientation are encoded consistently in <code>GlobalSideID</code> signs and <code>Flip</code>/edge-orientation fields.</dd> -->
<!-- </dl> -->

## FEM Connectivity

PyHOPE can optionally store additional connectivity information required by Finite Element Method (FEM)-based solvers. This includes topological connectivity of edges and vertices, as well as their connections across elements. This information is included when the global attribute `FEMconnect` is set to `ON`.

### Global Attributes

| <div style="width:120px">Attribute</div> | <div style="width:240px">Type</div> | Description                                                 |
| ---------------------------------------- | ----------------------------------- | ----------------------------------------------------------- |
| `nEdges`                                 | INTEGER                             | Total number of entries in `EdgeInfo`                       |
| `nVertices`                              | INTEGER                             | Total number of entries in `VertexInfo`                     |
| `nUniqueEdges`                           | INTEGER                             | Total number of geometrically unique edges                  |
| `nFEMSides`                              | INTEGER                             | Number of topologically (incl. periodicity) unique sides    |
| `nFEMEdges`                              | INTEGER                             | Number of topologically unique edges                        |
| `nFEMEdgeConnections`                    | INTEGER                             | Size of `EdgeConnectInfo`                                   |
| `nFEMVertices`                           | INTEGER                             | Number of topologically unique vertices                     |
| `nFEMVertexConnections`                  | INTEGER                             | Size of `VertexConnectInfo`                                 |

### Data Arrays

| <div style="width:120px">Array</div> | <div style="width:80px">Type</div> | <div style="width:200px">Size</div>  | Description              |
| ------------------------------------ | ----------------------------------- | ----------------------------------- | ------------------------ |
| `FEMElemInfo`                        | INTEGER                             | (1:4, 1:nElems)                     | Per-element offsets into `EdgeInfo` and `VertexInfo`: `(offsetIndEDGE, lastIndEDGE, offsetIndVERTEX, lastIndVERTEX)`. |
| `EdgeInfo`                           | INTEGER                             | (1:3, 1:nEdges)                     | For each local element edge (CGNS order): `(±FEMEdgeID, offsetIndEDGEConnect, lastIndEDGEConnect)`. Sign encodes local-to-global orientation. |
| `EdgeConnectInfo`                    | INTEGER                             | (1:2, 1:nFEMEdgeConns)              | Edge connections: `(±nbElemID, ±nbLocEdgeID)`. Sign on `nbElemID` encodes primary/replica; sign on `nbLocEdgeID` encodes orientation. |
| `VertexInfo`                         | INTEGER                             | (1:3, 1:nVertices)                  | For each local vertex (CGNS corner order): `(FEMVertexID, offsetIndVERTEXConnect, lastIndVERTEXConnect)`. |
| `VertexConnectInfo`                  | INTEGER                             | (1:2, 1:nFEMVertexConns)            | Vertex connections: `(±nbElemID, nbLocVertexID)`. Sign encodes primary/replica. |
