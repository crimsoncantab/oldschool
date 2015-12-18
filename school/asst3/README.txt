
Loren McGinnis
Assignment 3

Files submitted:
this README.txt
HelloWorld3D.cpp
ppm.cpp
helloWorld.sln
helloWorld.vcproj
<header files, ".h">
-ppm
-shader
-vec
-shaderstate
-matrix4
-cube
-floor
-object3d
-shape
-skycam
-arcball
-quaternion
-rbt
in shaders/
-basic.vshader
-diffuse.fshader
-solid.fshader

Platform:  Windows Vista x64

To compile and run:  Use MS VS 2008

Problem Set Requirements have been met.

Overview of my code changes:
Changed datatype of frame_ in all objects to be Rbt.  Implemented the Rbt class.  Implemented an Arcball class that handles rotations.

To run and test the program:
'f' key - changes fragment shading mode
'v'('V') key - changes view.  cycles between the skycam and the cubes, shift+v cycles backwards
'o'('O') key - changes current mutable object. also cycles between the skycam and the cubes, shift+o cycles backwards.  The skycam is not in this cycle unless it is the current view.
'h' key - help
's' key - print a screenshot
'm' key - available while looking from and modifying skycam.  toggles between sky-sky and sky-world coord system
mouse left button = rotates current mutable object.
mouse right button = translates current mutable object.
mouse both buttons/third button

Bonus feature:
'?' key - make me a cube!  Generates randomly colored, sized and position cube.  Also makes it editable.

Questions? Email at mcginn@fas.harvard.edu.

