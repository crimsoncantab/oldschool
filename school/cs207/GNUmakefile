#
# 'make'        build executable file
# 'make clean'  removes all .o and executable files
#

MODE := normal


DEPSDIR := $(shell mkdir -p .deps/CS207; echo .deps)

# Get the shell name to determine the OS
UNAME := $(shell uname)

# define the C compiler to use
CC := gcc
ifeq ($(UNAME), Linux)
CXX := g++ -std=gnu++0x
endif
ifeq ($(UNAME), Darwin)
CXX := $(shell for i in 4.7 4.6 4.5; do if g++-mp-$$i -v >/dev/null 2>&1; then echo g++-mp-$$i; exit; fi; done; echo false) -std=gnu++0x
OBJC := gcc
endif
LINK := $(CXX)

# define any compile-time flags
ifeq ($(MODE), normal)
CFLAGS := -O2
endif
ifeq ($(MODE), debug)
CFLAGS :=
endif
ifeq ($(MODE), prof)
CFLAGS := -DNDEBUG
endif
ifeq ($(MODE), fast)
CFLAGS := -O2 -DNDEBUG
endif
CFLAGS :=$(CFLAGS) -g -W -Wall -Wextra -DGL_GLEXT_PROTOTYPES
DEPCFLAGS = -MD -MF $(DEPSDIR)/$*.d -MP

# define any directories containing header files other than /usr/include
#   include directories like -Ipath/to/files
INCLUDES = -I.

# define any directories containing libraries other than /usr/lib
#   include directories like -Lpath/to/libraries
ifeq ($(UNAME), Linux)
SDLLIBS := -lSDL -lGL -lGLU
endif
ifeq ($(UNAME), Darwin)
SDLLIBS := -framework SDL -framework Cocoa -framework OpenGL
SDLOBJS := CS207/SDLMain.o
endif

# define any libraries to link into executable
#   To link in libraries (libXXX.so or libXXX.a) use -lXXX options
LIBS += -lm

##################
# The following part of the makefile is generic; it can be used to
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
##################

# 'make' - default rule
all: final

# generic rule - make sure object files are up-to-date, then compile MAIN
final: final.o $(SDLOBJS)
	$(LINK) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(SDLLIBS) $(LIBS)

shallow_water: shallow_water.o $(SDLOBJS)
	$(LINK) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(SDLLIBS) $(LIBS)

mass_spring: mass_spring.o $(SDLOBJS)
	$(LINK) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(SDLLIBS) $(LIBS)

mass_spring_mesh: mass_spring_mesh.o $(SDLOBJS)
	$(LINK) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(SDLLIBS) $(LIBS)

cs207viewer: cs207viewer.o $(SDLOBJS)
	$(LINK) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(SDLLIBS) $(LIBS)

test_edges: test_edges.o
	$(LINK) $(CFLAGS) $(LDFLAGS) -o $@ $^

test_nodes: test_nodes.o
	$(LINK) $(CFLAGS) $(LDFLAGS) -o $@ $^

test_mesh: test_mesh.o
	$(LINK) $(CFLAGS) $(LDFLAGS) -o $@ $^

test_mem: test_mem.o
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $<

test_morton: test_morton.o
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $<

# suffix replacement rule for building .o's from .cpp's
#   $<: the name of the prereq of the rule (a .cpp file)
#   $@: the name of the target of the rule (a .o file)
.cpp.o:
	$(CXX) $(CFLAGS) $(DEPCFLAGS) $(DEFS) $(INCLUDES) -c -o $@ $<

# Objective C (Mac)
.m.o:
	$(OBJC) $(CFLAGS) $(DEPCFLAGS) $(DEFS) $(INCLUDES) -c -o $@ $<

# 'make clean' - deletes all .o and temp files, exec, and dependency file
clean:
	-$(RM) *.o CS207/*.o *~
	-$(RM) final mass_spring cs207viewer test_edges test_nodes test_mem test_morton test_mesh shallow_water mass_spring_mesh
	$(RM) -r $(DEPSDIR)

always:
	@:

DEPFILES := $(wildcard $(DEPSDIR)/*.d) $(wildcard $(DEPSDIR)/*/*.d)
ifneq ($(DEPFILES),)
include $(DEPFILES)
endif

# define rules that do not actually generate the corresponding file
.PHONY: clean all always
