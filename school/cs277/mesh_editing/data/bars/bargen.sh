#!/bin/sh
# bargen.sh - generate a bar mesh
# usage: bargen.sh n
# where n is the discretization
# (number of times each of the 14 squares is subdivided into four more)

if [ $# != 1 ]; then
    echo "Usage: $0 n" 1>&2
    exit 1
fi

awk < /dev/null '

function printout(   i) {
    printf "%d %d\n", numpoints, numtris;
    for (i=1;i<=numpoints;i++) {
	printf "%s\n", allpoints[i];
    }
    for (i=1;i<=numtris;i++) {
	printf "%s\n", tris[i];
    }
}

function point(x,y,z,    s) {
    s = sprintf("%.8g %.8g %.8g", x, y, z);
    if (points[s]) {
	return points[s];
    }
    numpoints++;
    allpoints[numpoints] = s;
    points[s] = numpoints;
    return numpoints;
}

function tri0(x1,y1,z1, x2,y2,z2, x3,y3,z3,   p0,p1,p2) {
    p0 = point(x1,y1,z1);
    p1 = point(x2,y2,z2);
    p2 = point(x3,y3,z3);
    tris[++numtris] = sprintf("%d %d %d", p0-1, p1-1, p2-1);
    #printf "%d: %g %g %g  %g %g %g  %g %g %g\n", numtris, x1,y1,z1, x2,y2,z2, x3,y3,z3;
}

# 1 2
# 4 3
function sq0(x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4) {
    tri0(x1,y1,z1, x2,y2,z2, x4,y4,z4);
    tri0(x2,y2,z2, x3,y3,z3, x4,y4,z4);
}

#  1  5  2
#  8  9  6
#  4  7  3
function sq(n, x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4,
               x5,y5,z5, x6,y6,z6, x7,y7,z7, x8,y8,z8, x9,y9,z9) {
    if (n == 0) {
       sq0(x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4);
       return;
    }

    x5 = (x1+x2)/2;
    y5 = (y1+y2)/2;
    z5 = (z1+z2)/2;

    x6 = (x2+x3)/2;
    y6 = (y2+y3)/2;
    z6 = (z2+z3)/2;

    x7 = (x3+x4)/2;
    y7 = (y3+y4)/2;
    z7 = (z3+z4)/2;

    x8 = (x4+x1)/2;
    y8 = (y4+y1)/2;
    z8 = (z4+z1)/2;

    x9 = (x6+x8)/2;
    y9 = (y6+y8)/2;
    z9 = (z6+z8)/2;

    sq(n-1, x1,y1,z1, x5,y5,z5, x9,y9,z9, x8,y8,z8);
    sq(n-1, x5,y5,z5, x2,y2,z2, x6,y6,z6, x9,y9,z9);
    sq(n-1, x6,y6,z6, x3,y3,z3, x7,y7,z7, x9,y9,z9);
    sq(n-1, x7,y7,z7, x4,y4,z4, x8,y8,z8, x9,y9,z9);
}

#  1 2
#  5 6
#  7 8
#  4 3
function sq3(n, x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4,
                x5,y5,z5, x6,y6,z6, x7,y7,z7, x8,y8,z8,  a, b) {
    a = 2.0/3.0;
    b = 1.0/3.0;

    x5 = a*x1 + b*x4;
    y5 = a*y1 + b*y4;
    z5 = a*z1 + b*z4;

    x7 = b*x1 + a*x4;
    y7 = b*y1 + a*y4;
    z7 = b*z1 + a*z4;

    x6 = a*x2 + b*x3;
    y6 = a*y2 + b*y3;
    z6 = a*z2 + b*z3;

    x8 = b*x2 + a*x3;
    y8 = b*y2 + a*y3;
    z8 = b*z2 + a*z3;

    sq(n, x1,y1,z1, x2,y2,z2, x6,y6,z6, x5,y5,z5);
    sq(n, x5,y5,z5, x6,y6,z6, x8,y8,z8, x7,y7,z7);
    sq(n, x7,y7,z7, x8,y8,z8, x3,y3,z3, x4,y4,z4);
}

function bar(n) {
    # top
    sq(n, -0.5, 3.0, 0.5,  -0.5, 3.0, -0.5,   0.5, 3.0, -0.5,   0.5, 3.0, 0.5);
    # sides: front right back left
    sq3(n, -0.5, 3.0,  0.5,  0.5, 3.0,  0.5,  0.5, 0.0,  0.5, -0.5, 0.0,  0.5);
    sq3(n,  0.5, 3.0,  0.5,  0.5, 3.0, -0.5,  0.5, 0.0, -0.5,  0.5, 0.0,  0.5);
    sq3(n,  0.5, 3.0, -0.5, -0.5, 3.0, -0.5, -0.5, 0.0, -0.5,  0.5, 0.0, -0.5);
    sq3(n, -0.5, 3.0, -0.5, -0.5, 3.0,  0.5, -0.5, 0.0,  0.5, -0.5, 0.0, -0.5);
    # bottom
    sq(n,  0.5, 0.0, 0.5,   0.5, 0.0, -0.5,  -0.5, 0.0, -0.5,  -0.5, 0.0, 0.5);
}
END {
    bar(loops);
    printout();
}
' "loops=$1"
