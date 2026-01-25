lc = 1e-2;                              // Characteristic length (only matters for non-transfinite sizing)

Point(1) = {0,  0,  0, lc};             // Corner 1
Point(2) = {1,  0,  0, lc};             // Corner 2
Point(3) = {1,  1,  0, lc};             // Corner 3
Point(4) = {0,  1,  0, lc};             // Corner 4

Line(1) = {1, 2};                       // Bottom edge
Line(2) = {3, 2};                       // Right edge
Line(3) = {3, 4};                       // Top edge
Line(4) = {4, 1};                       // Left edge

Curve Loop(1) = {4, 1, -2, 3};          // Boundary loop for the surface
Plane Surface(1) = {1};                 // Create the planar surface bounded by Curve Loop(1)

Transfinite Curve {1,2,3,4} = 2;        // 2 points per edge (endpoints only) -> 1 division -> 1 element along each edge
Transfinite Surface {1} = {1,2,3,4};    // Use a structured transfinite map on the surface
Recombine Surface {1};                  // Recombine the structured triangles into quads

Physical Curve("BC_Bottom")     = {1};  // Put all curves in boundary conditions
Physical Curve("BC_Right" )     = {2};  // Put all curves in boundary conditions
Physical Curve("BC_Top")        = {3};  // Put all curves in boundary conditions
Physical Curve("BC_Left")       = {4};  // Put all curves in Zone1
Physical Surface("BC_Front")    = {1};  // Put the surface (and its elements) in Zone1

Mesh.ElementOrder = 2;                  // Linear elements
