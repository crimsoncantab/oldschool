
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
-robot
-joint
-ellipsoid
in shaders/
-basic.vshader
-diffuse.fshader
-solid.fshader

Platform:  Windows Vista x64

To compile and run:  Use MS VS 2008

Problem Set Requirements have been met.

Overview of my code changes:
Added some new shapes: ellipsoids, rectangular prisms, and robots.  Implemented the concept of "children" shapes.  When drawn, an shape calls the draw
method of its children, appending it's frame transformation after the inverse eye.  Shapes also have the ability to ignore the main implementation of
mouse movement and substitute their own.  The Joint class does this, and changes its own private fields to modify how its children are transformed.
The Joint class also uses a beforeChild_ transformation in the Object3D class to have an intermediate step between the Joint's frame and it's children. 

To run and test the program:
'v'('V') key - changes view.  cycles between the skycam and the cubes, shift+v cycles backwards
'o'('O') key - changes current mutable object. also cycles between the skycam and the cubes, shift+o cycles backwards.  The skycam is not in this cycle unless it is the current view.
'h' key - help
's' key - print a screenshot
'm' key - available while looking from and modifying skycam.  toggles between sky-sky and sky-world coord system
'p' key - followed by left click, picks clicked object as current mutable.
mouse left button - rotates current mutable object.
mouse right button - translates current mutable object.
mouse both buttons/third button - translates in z-direction

Bonus feature:
'?' key - make me a robot!  Generates randomly colored and positioned robot.  Also makes it editable.

Questions? Email at mcginn@fas.harvard.edu.

