See Simulator.hpp and Action.hpp doxygen comments for documentation.

In our integrated simulation, we used the normal() methods that Kenny and Tony provided for us to add lighting to our simulator.

Our visualizer (Simulator.hpp) works on both the shallow_water and mass_spring (with Mesh) simulations.  The features include:

Different mesh visualization modes:
-Toggle triangle shading with 't'
-Toggle the mesh being rendered at all with 'd'
-Toggle a vector field on the mesh with 'v'

Other features:
-pause the simulation with 'p'
-switch to 'selection mode' (turns off camera rotations/translations) with 's'
--In selection mode, drag a box with mouse to select nodes;  left-click dragging selects, right-click dragging deselects.

We also allow plugins for different key bindings using the Action.hpp
framework.
In the shallow water model, we have a couple examples of this:
-hitting 'e' will print out on the console the indices of all selected nodes
-hitting 'b' will add height to the selected nodes.

For the best experience, compile using:
make mode=FAST mass_spring_mesh shallow_water

Some example run commands to try (usage is basically the same):
./mass_spring_mesh input/sphere1*
./shallow_water input/pond4* 1
./shallow_water input/dam4* 2
