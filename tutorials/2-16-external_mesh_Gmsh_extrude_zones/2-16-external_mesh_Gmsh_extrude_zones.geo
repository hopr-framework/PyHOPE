//+
SetFactory("OpenCASCADE");

//--------------------------------------
// Points
//--------------------------------------
Point(1)  = {0, 0, 0};
Point(2)  = {10, 0, 0};
Point(3)  = {30, 0, 0};
Point(4)  = {0, 5, 0};
Point(5)  = {10, 5, 0};
Point(6)  = {30, 5, 0};
Point(7)  = {0, 10, 0};
Point(8)  = {10, 10, 0};
Point(9)  = {30, 10, 0};
Point(10) = {0, 20, 0};
Point(11) = {10, 20, 0};
Point(12) = {30, 20, 0};

//--------------------------------------
// Lines
//--------------------------------------
Line(1)  = {1, 2};
Line(2)  = {2, 3};
Line(3)  = {4, 5};
Line(4)  = {5, 6};
Line(5)  = {7, 8};
Line(6)  = {8, 9};
Line(7)  = {10,11};
Line(8)  = {11,12};

Line(9)  = {1, 4};
Line(10) = {4, 7};
Line(11) = {7,10};

Line(12) = {2, 5};
Line(13) = {5, 8};
Line(14) = {8,11};

Line(15) = {3, 6};
Line(16) = {6, 9};
Line(17) = {9,12};

// Surfaces
//--------------------------------------
Line Loop(21) = {1, 12, -3, -9};
Plane Surface(31) = {21};

Line Loop(22) = {3, 13, -5, -10};
Plane Surface(32) = {22};

Line Loop(23) = {5, 14, -7, -11};
Plane Surface(33) = {23};

Line Loop(24) = {2, 15, -4, -12};
Plane Surface(34) = {24};

Line Loop(25) = {4, 16, -6, -13};
Plane Surface(35) = {25};

Line Loop(26) = {6, 17, -8, -14};
Plane Surface(36) = {26};

// Physical Curves
//--------------------------------------
Physical Curve("ANODE", 25) = {10};
Physical Curve("DIEANODE", 26) = {9, 11};
Physical Curve("DIELECTRIC", 27) = {3, 12, 5, 14};
Physical Curve("WALL", 28) = {7};
Physical Curve("OUTLET", 29) = {8, 15, 16, 17};
Physical Curve("SYMAXIS", 30) = {1, 2};
Physical Curve("UNUSED", 31) = {4, 6, 13};

Physical Surface("ROTSYM") = {31,32,33,34,35,36};

Physical Surface("Zone_1") = {31};
Physical Surface("Zone_2") = {33};
Physical Surface("Zone_3") = {32,34,35,36};

//--------------------------------------
// Mesh refinement: Fully refine surface 32
//--------------------------------------
Mesh.MeshSizeMin = 0.15;
Mesh.MeshSizeMax = 15.0;

// Create a box field around Surface 32 (from x=0 to 10, y=5 to 10)
Field[1] = Box;
Field[1].VIn = 0.3;
Field[1].VOut = 0.5;
Field[1].XMin = 0;
Field[1].XMax = 10;
Field[1].YMin = 5;
Field[1].YMax = 10;
Field[1].ZMin = -1;
Field[1].ZMax = 1;

// Optional: Use additional field to control transition further to the right
Field[2] = Distance;
Field[2].EdgesList = {3,5,13};
Field[2].NumPointsPerCurve = 40;

Field[3] = Threshold;
Field[3].InField = 2;
Field[3].SizeMin = 0.2;
Field[3].SizeMax = 15.0;
Field[3].DistMin = 2.0;
Field[3].DistMax = 40.0;

// Combine the two using minimum
Field[4] = Min;
Field[4].FieldsList = {1, 3};

Background Field = 4;

//--------------------------------------
// Meshing options
//--------------------------------------
Mesh.Algorithm = 8;
Mesh.RecombinationAlgorithm = 2;
Mesh.RecombineAll = 1;
Mesh 2;

// Apply an elliptic smoother to the grid to have a more regular mesh:
Mesh.Smoothing = 100;

//--------------------------------------
// Output
//--------------------------------------
Mesh.SaveAll = 1;
Mesh.Binary = 0;
Mesh.MshFileVersion = 4.1;

Save "2-16-external_mesh_Gmsh_extrude_zones.msh";
