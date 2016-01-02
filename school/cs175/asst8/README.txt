
Loren McGinnis
Assignment 8

Files submitted:
this README.txt
ppm.cpp
asst8.cpp
asst8.sln
asst8.vcproj
bunny.mesh
<header files, ".h">
-shader
-vec
-matrix4
-arcball
-quaternion
-rbt
-mesh
in shaders/
-shader.frag
-shader.vert
-shader_fins.frag
-shader_fins.vert
-shader_shells.frag
-shader_shells.vert
in textures/
fin.ppm
shell.ppm

Platform:  Windows 7 x64

To compile and run:  Use MS VS 2008, Compile with Release Option

Problem Set Requirements have been met.

Overview of my code changes:
Implemented init_tips() to create list of tips (with zero velocity at point s) in world coords.
Implemented idle(), simulating physics on the tips.
Added section to draw() to show curved hair.

To run and test the program:
Hit 'o' to toggle between moving skycam and bunny.  Use mouse buttons to rotate/translate.  Get
some popcorn, and enjoy.

Questions? Email at mcginn@fas.harvard.edu.

