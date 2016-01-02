For the extra credit, I used BIGMIN to try to reduce the number of nodes
iterated by a neighborhood iterator.  I first tried doing this by having
a morton code iterator, which would iterate through all the codes of cells
that intersected a bounding box.  The neighborhood iterator used one of these
to only look at nodes close to the bounding box.  The problem here is similar
to the problems faced in schemes we talked about in class; iterating over the
cells directly fails to take into account the distribution of nodes, and
becomes roughly O(cells + nodes).  When the former is too large (e.g. for a
morton code of level 10), the iteration becomes very slow.

My improved approach instead simply iterated through the map of nodes,
using BIGMIN to skip over a range when it encounters a node whose cell does not
intersect the bounding box.  Since the bottleneck of the program is not the constraint
subsystem, I used a different metric to measure how well the iterator did.  I added
counters to determine how many nodes had to be iterated over in total, and of those
nodes, how many were not in the bounding box and had to be skipped.  The first number
displayed in the lower right corner is the fraction of nodes that were *not* skipped.
Before the improvements, using just the range between the morton code of the min
corner to that of the max corner, this value was often quite close to 0, except when
many nodes were being constrained.  After the improvements, the ratio was much higher,
but I noticed that there was a bit of hidden cost in calculating BIGMIN and jumping
to new locations in the map that in some setups made the improvements run slower.

The nodes are color coded:
purple - unexamined
blue - examined by neighborhood iterator and skipped
green - inside of the bounding box
red - constrained

the second number is ratio of virtual seconds to real seconds 
