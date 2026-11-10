//+
SetFactory("OpenCASCADE");

l_airbox = 0.10;

//------------------------------------------ airbox---------------------------------------------------------------
Box(1) = {-l_airbox/2, -l_airbox/2, -l_airbox/2 + l_airbox/4+l_airbox/8, l_airbox, l_airbox, l_airbox};

//--------------------------------------------sample---------------------------------------------------------- 
lx = 0.075;
ly = lx;
lz = 0.003;
Box(2) = {-lx/2, -ly/2, -lz/2, lx, ly, lz};



//------------------------------------ core (upper section)----------------------------------------------------------
lx = 0.0622;
ly = 0.0195;
lz = 0.0166;
hmax = 0.0567 + 0.003/2;
Box(3) = {-lx/2, -ly/2, hmax-lz, lx, ly, lz};

//------------------------------------ core (left section)----------------------------------------------------------
lx = 0.0166;
ly = 0.0195;
lz = 0.0567;
Box(4) = {-0.0622/2, -ly/2, 0.0015, lx, ly, lz};

//------------------------------------ core (right section)------------------------------------------------------------

Box(5) = {0.0622/2-0.0166, -ly/2, 0.0015, lx, ly, lz};


//------------------------------------- unified core--------------------------------------------------------------------------
BooleanUnion{ Volume{3}; Delete; }{ Volume{4}; Volume{5}; Delete; }


//---------------------------------------- Field coil--------------------------------------------------------------------------

//+++++++ legth in x for coil layers +++++++++++

diameter_wire = 0.000511;
lx_first = 51*diameter_wire;  // first layer has 51 turns 
lx_last = 24*diameter_wire;

//+++++++++++++++++++++++++++ z+ points ++++++++++++++++++++++++++

// layers length in y and height in z for z+ and z- points

aux = 2*diameter_wire;
ly_first = 0.0195; // coil has to touch the core 
ly_last = 0.0195 + 27*aux;

layer_max_height = hmax + diameter_wire*27;
// ----- references ---

rfx = 0;
rfy = 0;
rfz = hmax;

// first layer points
Point(2000) = {rfx-lx_first/2, rfy-ly_first/2, rfz, 1};
Point(2001) = {rfx-lx_first/2, rfy+ly_first/2, rfz, 1};
Point(2002) = {rfx+lx_first/2, rfy-ly_first/2, rfz, 1};
Point(2003) = {rfx+lx_first/2, rfy+ly_first/2, rfz, 1};

//last layer points

Point(2004) = {rfx-lx_last/2, rfy-ly_last/2, layer_max_height, 1};
Point(2005) = {rfx-lx_last/2, rfy+ly_last/2, layer_max_height, 1};
Point(2006) = {rfx+lx_last/2, rfy-ly_last/2, layer_max_height, 1};
Point(2007) = {rfx+lx_last/2, rfy+ly_last/2, layer_max_height, 1};

//+++++++++++++++++++++++++++ z- points ++++++++++++++++++++++++++

rfz = hmax - 0.0166;
layer_min_height = rfz - 27*diameter_wire;

// first layer points
Point(2008) = {rfx-lx_first/2, rfy-ly_first/2, rfz, 1};
Point(2009) = {rfx-lx_first/2, rfy+ly_first/2, rfz, 1};
Point(2010) = {rfx+lx_first/2, rfy-ly_first/2, rfz, 1};
Point(2011) = {rfx+lx_first/2, rfy+ly_first/2, rfz, 1};

//last layer points

Point(2012) = {rfx-lx_last/2, rfy-ly_last/2, layer_min_height, 1};
Point(2013) = {rfx-lx_last/2, rfy+ly_last/2, layer_min_height, 1};
Point(2014) = {rfx+lx_last/2, rfy-ly_last/2, layer_min_height, 1};
Point(2015) = {rfx+lx_last/2, rfy+ly_last/2, layer_min_height, 1};



//+ ---------------------- adding lines and surfaces to conform coil volume --------------------------------------------
Line(49) = {2004, 2006};
//+
Line(50) = {2005, 2007};
//+
Line(51) = {2012, 2014};
//+
Line(52) = {2015, 2013};
//+
Line(53) = {2000, 2004};
//+
Line(54) = {2002, 2006};
//+
Line(55) = {2006, 2007};
//+
Line(56) = {2005, 2004};
//+
Line(57) = {2012, 2004};
//+
Line(58) = {2014, 2006};
//+
Line(59) = {2014, 2010};
//+
Line(60) = {2008, 2012};
//+
Line(61) = {2003, 2007};
//+
Line(62) = {2005, 2001};
//+
Line(63) = {2005, 2013};
//+
Line(64) = {2007, 2015};
//+
Line(65) = {2011, 2015};
//+
Line(66) = {2013, 2009};
//+
Line(67) = {2013, 2012};
//+
Line(68) = {2014, 2015};
//+
Line(69) = {2009, 2001};
//+
Line(70) = {2011, 2003};
//+
Line(71) = {2010, 2002};
//+
Line(72) = {2008, 2000};
//+
Line(73) = {2002, 2003};
//+
Line(74) = {2000, 2001};
//+
Line(75) = {2011, 2010};
//+
Line(76) = {2009, 2008};//+
Curve Loop(23) = {54, 55, -61, -73};
//+
Plane Surface(23) = {23};
//+
Curve Loop(24) = {56, 49, 55, -50};
//+
Plane Surface(24) = {24};
//+
Curve Loop(25) = {56, -53, 74, -62};
//+
Plane Surface(25) = {25};
//+
Curve Loop(26) = {49, -58, -51, 57};
//+
Plane Surface(26) = {26};
//+
Curve Loop(27) = {58, -54, -71, -59};
//+
Plane Surface(27) = {27};
//+
Curve Loop(28) = {53, -57, -60, 72};
//+
Plane Surface(28) = {28};
//+
Curve Loop(29) = {51, 68, 52, 67};
//+
Plane Surface(29) = {29};
//+
Curve Loop(30) = {67, -60, -76, -66};
//+
Plane Surface(30) = {30};
//+
Curve Loop(31) = {68, -65, 75, -59};
//+
Plane Surface(31) = {31};
//+
Curve Loop(32) = {64, 52, -63, 50};
//+
Plane Surface(32) = {32};
//+
Curve Loop(33) = {63, 66, 69, -62};
//+
Plane Surface(33) = {33};
//+
Curve Loop(34) = {61, 64, -65, 70};
//+
Plane Surface(34) = {34};

// -------------------adding extra surfaces to conform coil volume----------------------------------------------

//+
Line(77) = {2008, 2010};
//+
Line(78) = {2000, 2002};
//+
Curve Loop(35) = {78, -71, -77, 72};
//+
Plane Surface(35) = {35};
//+
Line(79) = {2003, 2001};
//+
Line(80) = {2009, 2011};
//+
Curve Loop(36) = {79, -69, 80, 70};
//+
Plane Surface(36) = {36};
//+
Curve Loop(37) = {-79, -73, -78, 74};
//+
Plane Surface(37) = {37};
//+
Curve Loop(39) = {75, -77, -76, 80};
//+
Plane Surface(39) = {39};//+
Surface Loop(4) = {24, 25, 28, 26, 27, 23, 34, 32, 29, 31, 39, 35, 37, 36, 33, 30};
//+
Volume(4) = {4};

// ---------------------- fragmentation and asigning Physical volumes ---------------------------------------

BooleanFragments{ Volume{1,2,3,4}; Delete; }{}

// resulting Frags are [2:sample, 3:core, 4:coil, 5:air], same ids are used to name Physical volumes

For i In{2:5}
    Physical Volume(i) = {i};
EndFor

// ---------
Physical Surface("all_surfaces") = Surface{:}; // this is just for correct visualization of the surfaces, for the solver only physical volumes are required 
// without this the msh file won't display 2d element edges or faces when loaded
// ---------------


// ----------------- Field for finer mesh -----------------------------------------------------------------------
/*
factor = 1.05; // works with [1.1, 1.05]
Field[1] = Box;
lx_field = 0.075*factor;
ly_field = 0.075*factor;
lz_field = (layer_max_height + 0.0015)*factor; // ??

z_cord_center_shift =  (layer_max_height + 0.0015)/2 - 0.0015; // ???

Field[1].VIn = 0.001;// works with [0.002, 0.001]
Field[1].VOut = 1; // works with 1

Field[1].XMin = -lx_field/2; Field[1].XMax = lx_field/2;
Field[1].YMin = -ly_field/2; Field[1].YMax = ly_field/2;
Field[1].ZMin = z_cord_center_shift-lz_field/2; Field[1].ZMax = z_cord_center_shift+lz_field/2;

Background Field = 1;
*/