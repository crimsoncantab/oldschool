Representation:
Please refer to the "CS207 PA3 Design Document.pdf" in this folder.

How to run:
To run the shallow water program with different initial conditions, the 
following arguments are used.

./shallow_water [NODES_FILE] [TRI_FILE] [INIT_SETTINGS]
[NODES_FILE]    is the path of the mesh nodes that are to be loaded
[TRI_FILE]      is the path of the mesh triangles that are to be loaded
[INIT_SETTINGS] is either 0,1, or 2
                For value 0: We simulate the effect of having a pebble dropped
                onto the mesh.
                For value 1: We simulate having a large column of water 
                released onto the mesh.
                For value 2: We simulate having a dam break and a resulting
                temporary waterfall.
