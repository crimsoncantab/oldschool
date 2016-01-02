BARS=\
	bar-tiny.vp \
	bar-small.vp \
	bar-medium.vp \
	bar-foo.vp \
	bar-large.vp \
	bar-huge.vp

all: $(BARS)
clean:
	rm -f $(BARS)

.PHONY: all clean

bar-tiny.vp: bargen.sh Makefile
	./bargen.sh 0 > $@

bar-small.vp: bargen.sh Makefile
	./bargen.sh 1 > $@

bar-medium.vp: bargen.sh Makefile
	./bargen.sh 2 > $@

bar-foo.vp: bargen.sh Makefile
	./bargen.sh 3 > $@

bar-large.vp: bargen.sh Makefile
	./bargen.sh 4 > $@

bar-huge.vp: bargen.sh Makefile
	./bargen.sh 5 > $@
