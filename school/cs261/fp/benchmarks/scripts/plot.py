import sys
import os

script = \
'''set terminal png nocrop enhanced font verdana 12 size 1280,960
set xlabel "Stride Size (pages)"
set ylabel "Time (s)"
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
set log y
set log x'''



output = 'set output "graphs/%s-%s.png"'
title = 'set title "Stride with %s on %s"'
mach = {'laptop':'Hyper-threaded', 'vm':'8-core VM'}
benches = {'write':'Writes', 'read':'Reads', 'plock':'Pthread Locks', 'slock':'Spinlocks'}
affins = {'noaff':'No Affinity', 'sameaff':'Same Core', 'diffaff':'Different Cores', 'onethread':'One Thread'}
mem = {'global':'Global Mem', 'local':'Local Mem'}

print script
for ma, mav in mach.items():
    for b, bv in benches.items():
        i = 1
        s=''
        for m, mv in mem.items():
            for a, av in affins.items():
                fname = "data/%s/%s.%s.%s" % (ma,a,m,b)
                if os.path.exists(fname):
                    s+= '"%s" using 1:2 ' + \
                        'ls %d title "%s,  %s",'
                    s = s % (fname,i,mv,av)
                    i+=1
        if s:
            print (output % (ma, b))
            print (title % (bv, mav))
            s = s[:-1]
            print 'plot',s
        

output = 'set output "graphs/%s-%s-tlb.png"'
title = 'set title "Two Striding Processes Doing %s on Global Memory on %s"'
affins = {'diffaff':'Conflicted', 'sep':'Separated'}
procs = {1:'Process A', 2:'Process B'}


for ma, mav in mach.items():
    for b, bv in benches.items():
        i = 1
        s=''
        for a, av in affins.items():
            for p, pv in procs.items():
                fname = "data/%s/double.%s.%s.%d" % (ma,a,b,p)
                if os.path.exists(fname):
                    s+= '"%s" using 1:2 ' + \
                        'ls %d title "%s (%s)",'
                    s = s % (fname,i,av,pv)
                    i+=1
        if s:
            print (output % (ma, b))
            print (title % (bv, mav))
            s = s[:-1]
            print 'plot',s
