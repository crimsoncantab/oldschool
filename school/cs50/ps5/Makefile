#
# Makefile
#
# Computer Science 50
# Problem Set 5
#

CC      = gcc
CFLAGS  = -std=c99 -Wall

EXE  = speller

SRCS = speller.c dictionary.c
OBJS = $(SRCS:.c=.o)


$(EXE): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS)

clean:
	rm -f core $(EXE) *.o

