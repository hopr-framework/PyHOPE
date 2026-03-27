lc = 1e-2;                                      // Characteristic length (only matters for non-transfinite sizing)

// -------------------- Points --------------------
Point(1) = {0,  0,  0, lc};                     // Quad corner 1
Point(2) = {1,  0,  0, lc};                     // Quad corner 2 (shared with triangle)
Point(3) = {1,  1,  0, lc};                     // Quad corner 3 (shared with triangle)
Point(4) = {0,  1,  0, lc};                     // Quad corner 4
Point(5) = {2,  0,  0, lc};                     // Triangle extra node (apex)

// -------------------- Lines --------------------
Line(1) = {1, 2};                               // Quad bottom (outer)
Line(2) = {2, 3};                               // Shared face (internal interface)
Line(3) = {3, 4};                               // Quad top (outer)
Line(4) = {4, 1};                               // Quad left (outer)

Line(5) = {2, 5};                               // Triangle bottom (outer)
Line(6) = {5, 3};                               // Triangle slanted edge (outer)

// -------------------- Surfaces --------------------
Curve Loop(1) = {1, 2, 3, 4};                   // Quad boundary
Plane Surface(1) = {1};                         // Quad surface

Curve Loop(2) = {5, 6, -2};                     // Triangle boundary: 2->5->3->2 (uses -Line(2) to go 3->2)
Plane Surface(2) = {2};                         // Triangle surface

// -------------------- Mesh controls --------------------
Transfinite Curve {1,2,3,4,5,6} = 2;            // 2 points per edge -> 1 division (1 element) on each edge

Transfinite Surface {1} = {1,2,3,4};            // Structured transfinite quad
Recombine Surface {1};                          // Make Surface(1) a quad

Transfinite Surface {2} = {2,5,3};              // Structured transfinite triangle (exactly 1 tri)

// -------------------- Physical groups (BCs) --------------------
Physical Curve("BC_Bottom") = {1, 5};           // Outer boundary
Physical Curve("BC_Top")    = {3};              // Outer boundary
Physical Curve("BC_Left")   = {4};              // Outer boundary
Physical Curve("BC_Right")  = {6};              // Triangle outer boundary

Physical Surface("BC_Front") = {1,2};           // Entire front (2D) domain
Physical Surface("Zone_1")   = {1};             // Zone 1 (all extruded quads)
Physical Surface("Zone_2")   = {2};             // Zone 2 (all extruded wedges)

Mesh.ElementOrder = 4;                          // Linear elements
