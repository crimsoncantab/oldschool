#!/usr/bin/python
import common

#input: key - title; value - [links...] pagerank
#
#output: key - pagerank; value - title
def sort_map(line):
    title, links = line.split('\t', 1)
    pr = float(links.split('\t').pop())
    yield ('%15.2f' % pr, title)

if __name__ == "__main__":
    common.main(sort_map, None)
