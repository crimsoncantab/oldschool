Loren McGinnis
Assignment 5

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
Implemented key frames.  The program keeps a list of key frames, and one special FrameState object that inherits from KeyFrame, but also draws the scene.  Various commands will add/remove new frames from the list (or totally repopulate).  Animation also added.  Object3D, Rbt, Vector, and Quaternion objects have lerp() functions added, which are called recursively in animation.  The FrameState updates with a lerped version of two KeyFrame's when animating.

To run and test the program:
After running, type 'h' for a list of commands.

Questions? Email at mcginn@fas.harvard.edu.

