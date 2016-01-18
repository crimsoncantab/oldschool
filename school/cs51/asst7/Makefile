# Makefile for Asst 7
# LM, VS

LIB51  = /home/l/i/lib51
PATH = $(LIB51)/pub/bin:$(PLT)/bin:/usr/local/bin:/usr/bin/X11:/usr/bin:/bin
CSRC = Rule.cpp PatternRule.cpp MailMessage.cpp Mailbox.cpp Filter.cpp Spamfilter.cpp
CPPFLAGS = -g -I$(LIB51)/pub/include -W -Wall -pedantic -ansi


all: spamfilter

spamfilter: Rule.o PatternRule.o MailMessage.o Mailbox.o Filter.o Spamfilter.o
	g++ -L$(LIB51)/pub/lib *.o -o $@ -lmbox

%.o: %.cpp
	g++ -c $< $(CPPFLAGS)


clean:
	rm -f *.o a.out core spamfilter


submit:
	submit lib51 7 `pwd`
