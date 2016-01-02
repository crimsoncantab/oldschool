
Loren McGinnis
Assignment 2

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
in shaders/
-basic.vshader
-diffuse.fshader
-solid.fshader

Platform:  Windows Vista x64

To compile and run:  Use MS VS 2008

Problem Set Requirements have been met.

Overview of my code changes:
Developed "Object3D" class structure to describe different elements of scene.  Each Object3D understands how to draw itself and keeps track of its frame.  Added functionality to alter between objects for modification or eyepose. Keeping track of an auxilary frame for the current modified object, the code displays more user-friendly behavior for translation/rotation of an object.

To run and test the program:
'f' key - changes fragment shading mode
'v'('V') key - changes view.  cycles between the skycam and the cubes, shift+v cycles backwards
'o'('O') key - changes current mutable object. also cycles between the skycam and the cubes, shift+o cycles backwards.  The skycam is not in this cycle unless it is the current view.
'h' key - help
's' key - print a screenshot
'm' key - available while looking from and modifying skycam.  toggles between sky-sky and sky-world coord system

Bonus feature:
'?' key - make me a cube!  Generates randomly colored, sized and position cube.  Also makes it editable.

Questions? Email at mcginn@fas.harvard.edu.

