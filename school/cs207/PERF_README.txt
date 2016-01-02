The perf/ directory contains information about performance improvements we did
on our codebase.  The performance metrics we ran focused on improving the
throughput of the mass_spring and shallow_water simulations.  The image files
perf/*-graph.png summarize these performance improvements with two graphs.

In both simulations, the tests ran for 10 real seconds, without the visualizer
running.  mass_spring was executed on the grid3* files, and shallow_water ran
on the pond4* files.  The code was compiled with assertions turned off,
debugging symbols on, and both no optimization and -O2. See profile.sh for
details.

Each graph shows the throughput for a particular simulation ran with various
performance/feature enhancements to the code.  Each test is cumulative--it
includes all the changes of previous tests (More precisely, the commit
represented by a particular bar is a descendant of the commit to it's left; some
changes may have overwritten portions of their ancestors).  Additionally, each
test has a corresponding git tag for the code ran to produce those results;
roughly speaking, the if the test label is "spatial set", the git tag is
"perf_spatial_set."

Additionally, all changes have a corresponding cachegrind profile output (e.g.
"perf/*_spatial_set.out", which were used to target performance hotspots.

The output of all tests is in "perf/PERF.txt".

Some notes:

Mass Spring:
Overall, a 200% increase in throughput.
Most modifications had a small positive effect on the throughput of mass_spring.
A couple modifications had significant impact:
>"spatial update" - When a node's position was updated, the morton mapping was
updated regardless of whether the node's morton code changed.  This update
simply removed all map updates where the code did not change.
>"msf" - Consistently, the largest bottleneck in the simulation was the
MassSpringForce functor.  This simple change took advantage of Newton's Third
Law in terms of a spring force; the force accross an edge is now only calculated
once for the first node, and memoized for the latter node.
>"sloppy spatial" - This was an attempt to apply the same improvements to the
spatial set that Eddie did in class; it failed, because BIGMIN could no longer
be used, and BIGMIN did better at improving throughput.

Shallow Water:
Overall, a 100% increase in throughput.
Most improvements had little effect on shallow water; except for a few key
changes, not much stood out that could be easily optimized.
>"edge length" - This was the first performance change, which merely involved
changing the implementation of Graph::Edge::length() to access nodes directly.
It's impact turned out to be so great because this function was called much
more often than it needed to be.
>"sw pre" - By far the best improvement in terms of throughput increase.  This
involved some changes to the Mesh interface that made it more efficient to find
the shared edge between two triangles.  In addition, much of the information
used in the hyperbolic_step could be memoized, like edge lengths and normals,
since those values never changed.
