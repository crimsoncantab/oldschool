#
# Makefile
#
# Computer Science 50
# Problem Set 6
#

CC      = gcc
CFLAGS  = -ggdb -std=c99 -Wall

HEADERS = forest.h huffile.h tree.h

DUMPER      = dumper
DUMPER_SRCS = dumper.c forest.c huffile.c tree.c
DUMPER_OBJS = $(DUMPER_SRCS:.c=.o)

HUFF      = huff
HUFF_SRCS = huff.c forest.c huffile.c tree.c
HUFF_OBJS = $(HUFF_SRCS:.c=.o)

LDLIBS = -lm


$(DUMPER): $(DUMPER_OBJS) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(DUMPER_OBJS) $(LDLIBS)

$(HUFF): $(HUFF_OBJS) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(HUFF_OBJS) $(LDLIBS)

clean:
	rm -f core $(DUMPER) *.o
	rm -f core $(HUFF) *.o

