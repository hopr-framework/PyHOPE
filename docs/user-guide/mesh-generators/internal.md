# Internal Mesh Generator

A simple internal mesh generator for generating curved meshes of basic block-structured geometries

---

The internal generator creates structured, block-wise meshes (Cartesian boxes) with options for multi-zone setups, periodic couplings, and graded (stretched) element distributions.

## Mesh Specification

Internal meshes are specified using exclusively the [INI parameter file format](../parameter-file.md). As internal meshes are created from zones with 8 corner nodes, each zone is limited to a Cartesian box geometry (before mesh post-processing). The following minimal parameter file can be found in `tutorials/1-01-cartbox` and generates a simple Cartesian box mesh.

```ini
!================================================================================================================================= !
! OUTPUT
!================================================================================================================================= !
ProjectName  = 1-01-cartbox                         ! Name of output files
OutputFormat = HDF5                                 ! Mesh output format [HDF5, VTK, GMSH]

!================================================================================================================================= !
! MESH
!================================================================================================================================= !
Mode         = 1                                    ! Mode for Cartesian boxes
nZones       = 1                                    ! Number of boxes
Corner       = (/0.,0.,0. ,,1.,0.,0. ,,1.,1.,0. ,,  0.,1.,0.,, 0.,0.,1. ,,1.,0.,1. ,,1.,1.,1. ,,  0.,1.,1. /)
                                                    ! Corner node positions: (/ x_1,y_1,z_1, x_2,y_2,z_2,..... , x_8,y_8,z_8/)
nElems       = (/8,8,8/)                            ! Number of elements in each direction
BCIndex      = (/1,2,3,4,5,6/)                      ! Indices of boundary conditions for six boundary faces (z-,y-,x+,y+,x-,z+)
ElemType     = 108                                  ! Element type (104: Tetrahedra, 105: Pyramid, 106: Prism, 108: Hexahedral)
nGeo         = 9

!================================================================================================================================= !
! BOUNDARY CONDITIONS
!================================================================================================================================= !
BoundaryName = BC_zminus                            ! BC index 1 (from position in parameterfile)
BoundaryType = (/4,0,0,0/)                          ! (/ Type, curveIndex, State, alpha /)
BoundaryName = BC_yminus                            ! BC index 2
BoundaryType = (/2,0,0,0/)
BoundaryName = BC_xplus                             ! BC index 3
BoundaryType = (/2,0,0,0/)
BoundaryName = BC_yplus                             ! BC index 4
BoundaryType = (/2,0,0,0/)
BoundaryName = BC_xminus                            ! BC index 5
BoundaryType = (/2,0,0,0/)
BoundaryName = BC_zplus                             ! BC index 6
BoundaryType = (/9,0,0,0/)
```

!!! info
    Mesh output in VTK / Gmsh format is provided mainly for visualization purposes and might not support all high-order features.

### Multiple Cartesian Boxes

![Internal mesh with multiple Cartesian boxes](../../assets/internal_mesh.png#only-light){ width='300', loading=lazy, align=right }
![Internal mesh with multiple Cartesian boxes](../../assets/internal_mesh_dark.png#only-dark){ width='300', loading=lazy, align=right }
PyHOPE supports multi-zone setups consisting of multiple Cartesian boxes. Each box, referred to as a zone, is defined by its 8 corner nodes specified in the `Corner` vector. The number of elements in each direction for each box is specified in the `nElems` vector. If boxes are connected with each other, the corner nodes of the touching element faces must either be identical or be placed at the center of the element edge (non-conforming interfaces with hanging nodes).

## Boundary Conditions

Boundary conditions are specified in the parameter file using the `BoundaryName` and `BoundaryType` keywords. The order of the boundary conditions is determined by the order of the boundary faces given as `z-,y-,x+,y+,x-,z+` inside the `BCIndex` vector. Indices can be applied repeatedly.

### Periodic Boundary Conditions

Boundary faces can be coupled periodically by specifying the `BoundaryType = (/1,0,0,-1/)` where the last entry `alpha` specifies the periodic displacement vector to be used. A negative `alpha` corresponds to a displacement vector applied in the opposite direction. Displacement vectors are indexed in the order of their appearance in the parameter file. The following minimal parameter file can be found in `tutorials/1-02-cartbox-periodic` and generates a Cartesian box mesh with periodic coupling in x- and y-directions.

```ini
!================================================================================================================================= !
! OUTPUT
!================================================================================================================================= !
ProjectName  = 1-02-cartbox_periodic                ! Name of output files
DebugVisu    = F                                    ! Launch the GMSH GUI to visualize the mesh
OutputFormat = HDF5                                 ! Mesh output format [HDF5, VTK, GMSH]

!================================================================================================================================= !
! MESH
!================================================================================================================================= !
Mode         = 1                                    ! Mode for Cartesian boxes
nZones       = 1                                    ! Number of boxes
Corner       = (/0.,0.,0. ,,1.,0.,0. ,,1.,1.,0. ,,  0.,1.,0.,, 0.,0.,1. ,,1.,0.,1. ,,1.,1.,1. ,,  0.,1.,1. /)
                                                    ! Corner node positions: (/ x_1,y_1,z_1, x_2,y_2,z_2,..... , x_8,y_8,z_8/)
nElems       = (/2,3,4/)                            ! Number of elements in each direction
BCIndex      = (/1,3,6,4,5,2/)                      ! Indices of boundary conditions for six boundary faces (z-,y-,x+,y+,x-,z+)
ElemType     = 108                                  ! Element type (108: Hexahedral)
doFEMConnect = T                                    ! Generate finite element method (FEM) connectivity

!================================================================================================================================= !
! BOUNDARY CONDITIONS
!================================================================================================================================= !
BoundaryName = BC_zminus                            ! BC index 1 (from position in parameterfile)
BoundaryType = (/1,0,0,1/)                          ! (/ Type, curveIndex, State, alpha /)
BoundaryName = BC_zplus                             ! BC index 2
BoundaryType = (/1,0,0,-1/)                         ! Direction of vector 1 is reversed, because this is the opposite side
vv = (/0.,0.,1./)                                   ! Vector for periodic BC in z direction (zminus,zplus), index=1

BoundaryName = BC_yminus                            ! BC index 3
BoundaryType = (/1,0,0,2/)
BoundaryName = BC_yplus                             ! BC index 4
BoundaryType = (/1,0,0,-2/)                         ! (/ BCType=1: periodic, 0, 0, Index of second vector vv in parameter file /)
vv = (/0.,1.,0./)                                   ! Vector for periodic BC in y direction (yminus,yplus), index=2

BoundaryName = BC_inflow                            ! BC index 5
BoundaryType = (/2,0,0,0/)
BoundaryName = BC_outflow                           ! BC index 6
BoundaryType = (/2,0,0,0/)
```

## Stretching Functions

By default, elements are distributed uniformly in each direction within each transfinite mesh zone. Stretching (grading) changes this distribution, so that a box can contain a stretched element arrangement. The following parameters control stretching and are defined separately for each zone.

<dl class="aligned-list">
    <dt><code>StretchType</code>:</dt>
    <dd>Stretching type. Possible values are <code>0</code> (no stretching, uniform element distribution), <code>1</code> (stretching with a progression factor), <code>2</code> (stretching with a length ratio) and <code>3</code> (stretching with a bell function).</dd>
    <dt><code>Factor</code>:</dt>
    <dd>Stretching factor. Values > 1.0 lead to an exponential increase of the element size in the respective direction, values < 1.0 lead to an exponential decrease. A value of 1 has no effect on the element sizes, and a value of 0 disables the stretching function for this axis. The stretching behavior can be mirrored by adding a negative sign to the value. Combining this parameter with <code>l0</code> ignores the element number of the defined box.</dd>
    <dt><code>l0</code>:</dt>
    <dd>Length scale. The length of the first element of a stretched element arrangement of a Cartesian box. Each component of the vector represents an axis of the Cartesian coordinate system. A value of 0 disables the stretching function for that axis. A negative sign defines the length of the first element on the opposite side of the box. Combining this parameter with <code>Factor</code> ignores the element number of the defined box.</dd>
    <dt><code>DXmaxToDXmin</code>:</dt>
    <dd>Size ratio. The ratio of the largest to the smallest element size in the respective direction. If the <code>StretchType</code> vector component for an axis is <code>3</code>, the element arrangement is controlled by <code>DXmaxToDXmin</code> instead of the parameter <code>Factor</code>.</dd>
</dl>

### Calculation Formulas

The stretching functions are applied to each direction independently. The following formulas are used to calculate the element sizes and positions.

- Calculation of the element size for `StretchType=1`:

$$
\Delta x_{i+1} = f \cdot \Delta x_{i}, \qquad f = \left(\frac{\Delta x_{max}}{\Delta x_{min}}\right)^{1/(n_{\text{Elems}} - 1)}
$$

- Calculation of the element size for `StretchType=3`:

$$
\Delta x(\xi) \sim 1 + \left(\frac{\Delta x_{max}}{\Delta x_{min}} - 1\right) \cdot \left(\frac{\exp[-(\xi \cdot f)^2] - \exp[-f^2]}{\exp[0] - \exp[-f^2]}\right)
$$

## Post-Deformation

Mesh zones can be deformed after the initial mesh generation using a set of pre-defined deformation templates. The resulting curved multi-block mesh is composed of several structured boxes, which are first assembled and then globally mapped to a curved domain. A template is selected with the `MeshPostDeform` parameter, for example `MeshPostDeform = sphere`. The default, `none`, applies no deformation.

<dl class="aligned-list">
    <dt><code>convtest</code>:</dt>
    <dd>Convergence test mapping. A simple sinusoidal deformation in all directions to test the convergence behavior of high-order methods on curved meshes.</dd>
    <dt><code>cylinder</code>:</dt>
    <dd>Cylindrical mapping. The 2D square region is mapped to a cylindrical or toroidal coordinate system.</dd>
    <dt><code>default</code>:</dt>
    <dd>Identity mapping. The points are returned unchanged.</dd>
    <dt><code>phill</code>:</dt>
    <dd>Periodic hill mapping. A 2D mapping that generates an extruded periodic hill geometry.</dd>
    <dt><code>sine</code>:</dt>
    <dd>Sinusoidal deformation with several selectable types.</dd>
    <dt><code>sphere</code>:</dt>
    <dd markdown="1">Spherical mapping. The 3D cubic region is mapped to a spherical coordinate system inside \([-1, 1]^3\).</dd>
    <dt><code>sphere_box</code>:</dt>
    <dd markdown="1">Spherical mapping with a cubic outer region. The cube \([-1, 1]^3\) is mapped to a sphere, and points outside the cube are smoothly blended back to a cube.</dd>
</dl>
