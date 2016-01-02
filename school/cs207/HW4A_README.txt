See PERF_README.txt for details on performance changes.

For the visualizer, we implemented several things:

Drawing triangles/shading meshes
Selecting Nodes
Acting on selected nodes with callbacks
Drawing vectors
Pausing the simulation
and a few other tidbits

Our improved visualizer is in Simulator.hpp

To play with it, run shallow_water as normal.  The following key commands are
built into Simulator:

c - Center camera as usual
p - (Un)Pause Simulation
v - Toggle vector drawing
s - Toggle select mode (see below)
t - Toggle shading vs. wire mesh
d - Toggle drawing of the mesh (vectors will still be drawn if active)
q/ESC - exit as usual


Select Mode:
While in select mode, the mouse no longer rotates/translates.  By left-clicking
on a node, you can add it to a selection set.  Right clicking on a selected node
will remove it from the selection set.

Action callbacks:
What use is it to select a set of nodes you say?  To perform actions on them!
By registering a callback with the simulator, you can bind a key to that callback,
and when the key is pressed, the callback function is called with a node iterator
range as parameters.

See shallow_water.cpp and forthcoming documentation for details.
