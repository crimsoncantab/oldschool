benches = {'write':'Writes', 'read':'Reads', 'plock':'Pthread Locks'}
affins = {'noaff':'No Affinity', 'sameaff':'Same CPU', 'diffaff':'Different CPUs', 'onethread':'One Thread'}
mem = {'global':'Global Mem', 'local':'Local Mem'}

filename = 'data/%s.%s.%s'
#for b, bv in benches.items():
#    for a, av in affins.items():
#    print bv
for m, mv in mem.items():
    print mv
    samel = open(filename % ('onethread', m, 'plock')).readlines()
    diffl = open(filename % ('onethread', m, 'write')).readlines()
    for s, d in zip(samel, diffl):
        print float(s.split()[1]) / float(d.split()[1])
        
            
