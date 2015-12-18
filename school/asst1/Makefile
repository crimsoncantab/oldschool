
BASE = HelloWorld2D

all: ${BASE}

CPP = cc
CPPFLAGS = -I/home/l/i/lib175/usr/glew/include
LDFLAGS += -L/home/l/i/lib175/usr/glew/lib -L/usr/X11R6/lib


${BASE}: ${BASE}.o ppm.o
	${LINK.cpp} -o $@ ppm.o $< -lGLEW -lGL -lglut


${BASE}.o: ppm.cpp ${BASE}.cpp 

clean:
	-rm -f *.o ${BASE}

