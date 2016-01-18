#!/usr/bin/python
import common

dampening = 0.85

#input: key - page-title; value - [links...] pagerank
#
#outputs:
#  key - link-title; value - link-pagerank 'r'
#  key - title; value - [links...] 'l'
def pagerank_map(line):
    title, links = line.split('\t', 1)
    links = links.split('\t')
    pr = float(links.pop())
    if links:
        link_pr = pr / len(links)
        for link in links:
            yield (link, str(link_pr) + '\tr')
    links.append('l')
    yield (title, '\t'.join(links))
        
#inputs:
#  key - title; value - link-pagerank 'r'
#  key - title; value - [links...] 'l'
#
#output: key - title; value - [links...] pagerank
def pagerank_reduce(title, values):
    pr = 0.0
    has_links = False
    for value in values:
        value_list = value.split('\t')
        value_type = value_list.pop()
        if value_type == 'r':
            pr = pr + float(value_list[0])
        elif value_type == 'l':
            has_links = True
            links = value_list
    if (has_links):
        pr = 1 - dampening + (dampening * pr)
        links.append(str(pr))
        yield(title, '\t'.join(links))

if __name__ == "__main__":
    common.main(pagerank_map, pagerank_reduce)
