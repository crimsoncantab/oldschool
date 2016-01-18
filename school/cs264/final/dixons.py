import math

N = 84923
B = 29


FB = []
def isPrime(p):
    for f in FB:
        if p % f == 0:
            return False
        if f * f > p:
            break
    return True

def genFB(B):
    for n in range(2, B + 1):
        if isPrime(n):
            FB.append(n)

def checkPerm(lst, output):
    num = 1
    mod = 1
    for item in lst:
        num *= item
        mod *= (item * item) % N
    if int(math.sqrt(mod)) == math.sqrt(mod):
        output.append((num, mod))

def generatePerms(lst, remainder, output):
    if remainder != []:
        item = remainder.pop()
        generatePerms(lst[:], remainder[:], output)
        lst.append(item)
        generatePerms(lst, remainder, output)
    else:
        checkPerm(lst, output)

def findBSNs():
    BSNs = []
    for i in range(int(math.sqrt(N)), N):
        num = (i * i) % N
        for f in FB:
            while num % f == 0:
                num /= f
        if num == 1:
            BSNs.append(i)
            if len(BSNs) == len(FB) + 2:
                break
    return BSNs

def gcd(a, b):
    while b > 0:
        (a, b) = (b, a % b)
    return a


genFB(B)
BSNs = findBSNs()
sqrts = []
generatePerms([], BSNs, sqrts)
factors = set()
for pair in sqrts:
    (num, mod) = pair
    mod = math.sqrt(mod)
    factors.add(int(gcd(abs(num - mod), N)))
    factors.add(int(gcd(num + mod, N)))
print list(factors)
