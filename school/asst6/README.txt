Loren McGinnis
Assignment 6

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
-keyframe
-framestate
in shaders/
-basic.vshader
-diffuse.fshader
-solid.fshader

Platform:  Windows Vista x64

To compile and run:  Use MS VS 2008

Problem Set Requirements have been met.

Overview of my code changes:
For everything in the previous assignment with a lerp() function, I added a "bezerp()" (BEZier intERPolation) function that takes
the same arguments as lerp(), with the additional prev and post frames, from which it creates 4 points to do the cubic bezier.

To run and test the program:
After running, type 'h' for a list of commands.

Questions? Email at mcginn@fas.harvard.edu.

