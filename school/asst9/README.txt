
Loren McGinnis
Assignment9

Files submitted:
README.txt
/filt
-/shaders
--texmapping0.frag
--texmapping0.vert
--texmapping1.frag
--texmapping1.vert
-asst9filt.cpp
-asst9filt.vcproj
-matrix4.h
-ppm.cpp
-ppm.h
-reachup.ppm
-shader.h
-vec.h
/pers
-/shaders
--persptex1.fshader
--persptex1.vshader
--persptex2.fshader
--persptex2.vshader
-asst9pers.cpp
-asst9pers.vcproj
-matrix4.h
-ppm.cpp
-ppm.h
-reachup.ppm
-shader.h
-vec.h
/proj
-/shaders
--projtex.fshader
--projtex.vshader
-asst9proj.cpp
-asst9proj.vcproj
-matrix4.h
-ppm.cpp
-ppm.h
-reachup.ppm
-shader.h
-vec.h

Platform:  Windows 7 x64

To compile and run:  Use MS VS 2008

Problem Set Requirements have been met.

Overview of my code changes:
For filter, implemented bilinear reconstruction in the texmapping1.frag fragment shader.
For perspective, added code to pass 1/w_n as varying variable, then divided it out in
persptex2.fshader fragment shader.
For projection, modified fragment shader to set gl_FragCOlor to the texture coords
x and y of pTexCoord, after dividing w out.

To run and test the program:
All three programs should run as defined in the assignment spec

Questions? Email at mcginn@fas.harvard.edu.

