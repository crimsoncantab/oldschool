# Makefile v2.0

CC = g++

FILES = main DictList

OBJECTS = $(FILES:%=%.o)

EXENAME = DictList
DEBUGFLAGS = -g -DDEBUG $(FLAGS)
SHIPFLAGS = -O2 $(FLAGS)

default: debug

debug: $(OBJECTS)
	$(CC) $(DEBUGFLAGS) -o $(EXENAME) $(OBJECTS) $(LIBS)

release: $(OBJECTS)
	$(CC) -o $(EXENAME) $(OBJECTS) $(LIBS)

%.o: %.cpp
	$(CC) $(DEBUGFLAGS) -c -o $@ $(SHIPFLAGS) $<

clean:
	rm -f *.o
	rm -f $(EXENAME)
