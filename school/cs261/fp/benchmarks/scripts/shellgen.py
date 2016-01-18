import itertools

setup = """set -x on

write=0
read=1
plock=2
slock=3

noaff=0
sameaff=1
diffaff=2
oddaff=3
evenaff=4
onethread=5

global=0
local=1
"""

print setup

benches = ["read", "write", "plock", "slock"]
sizes = range(1, 16)
affins = ["noaff", "sameaff", "diffaff", "onethread"]
mem = ["global", "local"]
                      
#double proc stuff (for TLB benchmarks)
prog = "./prog $%s %d $%s $%s >> data/double.%s.%s.%d &"
m = 'global'
for b, s in itertools.product(benches, sizes):
    a = 'diffaff'
    diffaff1 = prog % (b, s, a, m, a, b, 1)
    diffaff2 = prog % (b, s, a, m, a, b, 2)
    oddaff = prog % (b, s, 'oddaff', m, 'sep', b, 1)
    evenaff = prog % (b, s, 'evenaff', m, 'sep', b, 2)
    print diffaff1
    print 'PID=$!'
    print diffaff2
    print 'PID2=$!'
    print 'wait $PID'
    print 'wait $PID2'
    print oddaff
    print 'PID=$!'
    print evenaff
    print 'PID2=$!'
    print 'wait $PID'
    print 'wait $PID2'
#single proc stuff
prog ="./prog $%s %d $%s $%s >> data/%s.%s.%s"
for b, s, a, m in itertools.product(benches, sizes, affins, mem):
    print (prog % (b, s, a, m, a, m, b))
    
