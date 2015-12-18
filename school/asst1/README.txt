
Loren McGinnis
Assignment 1

Files submitted:
mountain.ppm
reachup.ppm
HelloWorld2D.cpp
helloWorld.sln
helloWorld.vcproj
ppm.cpp
ppm.h
shader.h
in shaders/
-asst1.vshader
-asst1.fshader

Platform:  Windows Vista x64

To compile and run:  Use MS VS 2008

Problem Set Requirements have been met (Even fixed the weird GL "invalid operation" error!)

Overview of my code changes:
Added a handle to keep track of the radio between the window height and width, then used that in the vertex shader to maintain aspect ratio.
Modified the fragment shader to do two things:
-Added a color mask that would filter out colors depending on which was pressed
-Created a gradient between the two textures

To run and test the program:
Change size of window.  Image will maintain aspect ratio.
Left click and drag horizontally.  This will move the gradient toward one texture or the
other.
Right click and drag.  Same functionality: stretches/shrinks the image
Use the "r", "g", and "b" keys to toggle red, green, and blue filters.

Questions? Email at mcginn@fas.harvard.edu.

