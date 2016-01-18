import sys
machine = 'laptop'

script = \
'''set terminal png nocrop enhanced font verdana 12 size 1280,960
set xlabel "Stride Size (pages)"
set ylabel "Ratio"
set style line 1 lw 5 lc rgb "red"
set style line 2 lw 5 lc rgb "green"
set style line 3 lw 5 lc rgb "blue"
set style line 4 lw 5 lc rgb "orange"
set style line 5 lw 5 lc rgb "yellow"
set style line 6 lw 5 lc rgb "purple"
set style line 7 lw 5 lc rgb "brown"
set style line 8 lw 5 lc rgb "pink"
set style data lines
set style fill solid
set key right bottom
set log x
set log y'''



output = 'set output "laptop-samediffratio-loglog.png"'
title = 'set title "Same CPU/Diff CPU Ratio"'
benches = {'write':'Writes', 'plock':'Pthread Locks'}
mem = {'global':'Global Mem'}

print script
print output
print title
print 'plot',
i = 1
s = ''
for b, bv in benches.items():
        for m, mv in mem.items():
            s+= '"%s-%s" using 1:2 ' + \
                'ls %d title "%s,  %s",'
            s = s % (m,b,i,bv,mv)
            i+=1
s = s[:-1]
print s
