#!/usr/bin/python
import re
import common


titleregex = re.compile(r'<title>(.*?)</title>' )
linkregex = re.compile(r'\[\[([^\[\]]*?)(\|[^\[\]]*?)?\]\]', \
    re.I | re.U)
    
dampening = 0.85
pr_start = 1 - dampening

#input: raw data
#output: key - title; value - [links...]
def linkgraph_map(line):
    tmatch = titleregex.search(line)
    if tmatch:
        title = tmatch.group(1).strip().replace('\t', ' ')
        links = map( \
            lambda x : x[0].strip().replace('\t', ' '), \
            linkregex.findall(line))
        yield (title, '\t'.join(links))
    
#input: key - title; value - [links...]
#output: key - title; value - [links...] pagerank
def linkgraph_reduce(title, links):
    links.append(str(pr_start))
    yield (title, '\t'.join(links))

if __name__ == "__main__":
    common.main(linkgraph_map, linkgraph_reduce)
