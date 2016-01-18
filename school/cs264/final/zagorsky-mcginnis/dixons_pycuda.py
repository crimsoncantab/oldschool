import math
import sys
import pycuda.autoinit
import pycuda.tools as cudatools
import pycuda.driver as drv
import numpy
import time
from pycuda.compiler import SourceModule

block_dim = 512
factor_base = []

#bignum constants
bn_bucket_size = 32
bn_bucket_max = 2**bn_bucket_size

class Timer():
    def __init__(self, message):
        self.message = message
    def __enter__(self):
        print self.message + '...',
        self.start = time.time()
    def __exit__(self, *args):
        print 'done.'
        print 'runtime:', time.time() - self.start, 's'

def tobignum(num, bn_num_buckets):
   out = [0]*bn_num_buckets
   for i in range(bn_num_buckets):
       tmp = num % (2**(bn_bucket_size * (bn_num_buckets - i)))
       out[i] = int(tmp / (2**(bn_bucket_size * (bn_num_buckets - i - 1))))
   return out
   


def gen_bsmooth_kernel(n, numThreads, bn_num_buckets):
    sqrtn = int(math.sqrt(n)) + 1
    kernel = ''
    # we assume "overflow" of *really* big numbers doesn't happen
    kernel += \
    """
inline __device__ void clean(ulong * num){
    for (int i = """ + str(bn_num_buckets - 1) + """; i > 0; i--) {
        num[i-1] += num[i] >> """ + str(bn_bucket_size) + """;
        num[i] %= """ + str(bn_bucket_max) + """;
    }
}
    """
    kernel += \
    """
inline __device__ void clean2(ulong * num){
    for (int i = """ + str(bn_num_buckets*2 - 1) + """; i > 0; i--) {
        num[i-1] += num[i] >> """ + str(bn_bucket_size) + """;
        num[i] %= """ + str(bn_bucket_max) + """;
    }
}
    """
    kernel += \
    """
inline __device__ void square(ulong * in, ulong * out) {
    ulong temp[""" + str(bn_num_buckets+1) + """];
    for (int i = 0; i < """  + str(bn_num_buckets*2) + """; i++) {
        out[i] = 0;
    }
    for (int i = 0; i < """  + str(bn_num_buckets) + """; i++) {
        temp[0] = 0;
        for (int j = 0; j < """  + str(bn_num_buckets) + """; j++) {
            temp[j + 1] = in[i] * in[j];
        }
        for (int j = """ + str(bn_num_buckets) + """; j > 0; j--) {
            temp[j-1] += temp[j] >> """ + str(bn_bucket_size) + """;
            temp[j] %= """ + str(bn_bucket_max) + """;
        }
        for (int j = 0; j < """  + str(bn_num_buckets+1) + """; j++) {
            out[i + j] += temp[j];
        }
    }
    clean2(out);
}
    """
    
    kernel += \
    """
inline __device__ void bigshift(ulong * in, ulong * out, int shift){
    int blockshift = shift / """ + str(bn_bucket_size) + """;
    int bitshift = shift % """ + str(bn_bucket_size) + """;
    for (int i = 0; i < """ + str(bn_num_buckets * 2) + """; i++) {
        if ((i >= (""" + str(bn_num_buckets) + """ - blockshift)) && (i < (""" + str(2 * bn_num_buckets) + """ - blockshift))) {
            out[i] = in[i + blockshift -"""+str(bn_num_buckets)+"""] << bitshift;
        }
        else {
            out[i] = 0;
        }
    }
    clean2(out);
}
    """
    
    kernel += \
    """
inline __device__ void subtract(ulong * out, ulong * sub){
    ulong underflow = 0;
    for (int i = """+ str(bn_num_buckets*2 - 1) + """; i >= 0; i--) {
        if (out[i] < sub[i] + underflow) {
            out[i] += """ + str(bn_bucket_max) + """ - underflow;
            underflow = 1;
        }
        else {
            out[i] -= underflow;
            underflow = 0;
        }
        out[i] -= sub[i];
    }
}
    """
    
    #comp returns true (1) if num1 >= num2
    kernel += \
    """
inline __device__ int comp(ulong * num, ulong * num2) {
    for (int i = 0; i < """+ str(bn_num_buckets * 2) + """; i++) {
        if (num[i] > num2[i]) return 1;
        if (num[i] < num2[i]) return 0;
    }
    return 1;
}
    """
    
    kernel += \
    """
inline __device__ void mod(ulong * num, ulong * num2) {
    ulong tmp[""" + str(bn_num_buckets * 2) + """];
    for (int i = """ + str((bn_num_buckets * bn_bucket_size) - 1) + """; i >= 0; i--) {
        bigshift(num2, tmp, i);
        if (comp(num, tmp)) {
            subtract(num, tmp);
        }
    }
}
    """
    
    kernel += \
    """
inline __device__ int isfactorof(ulong * num, int factor) {
    ulong tmp = num[0];
    for (int i = 0; i < """ + str(bn_num_buckets*2 - 1) + """; i++) {
        tmp = ((tmp % factor) << """ + str(bn_bucket_size) + """);
        tmp += num[i+1];
    }
    return (tmp % factor == 0);
}
    """

    kernel += \
    """
inline __device__ void smalldiv(ulong * num, int divisor) {
    for (int i = 0; i < """ + str(bn_num_buckets*2 - 1) + """; i++) {
        num[i+1] += (num[i] % divisor) << """ + str(bn_bucket_size) + """;
        num[i] /= divisor;
    }
    num[""" + str(bn_num_buckets*2 - 1) + """] /= divisor;
    clean2(num);
}
    """
    
    kernel += \
    """

inline __device__ bool eq(ulong * num, int smallnum) {
    for (int i = 0; i < """ + str(bn_num_buckets*2 - 1) + """; i++) {
        if (num[i]) {
            return 0;
        }
    }
    return (num[""" + str(bn_num_buckets*2 - 1) + """] == smallnum);
}
    """
    
    n = tobignum(n, bn_num_buckets)
    kernel += \
    """
__global__ void do_bsmooth_kernel(ulong * out) {
"""
    kernel += '    __shared__ int flag;\n'
    kernel += '    flag = 0;\n'
    kernel += '    /*__shared__*/ ulong n[' + str(bn_num_buckets) + '];\n'
    for i in range(bn_num_buckets):
        kernel += '    n['+str(i)+'] = ' + str(n[i]) + ';\n'
    kernel += '    __syncthreads();\n'

    sqrtn = tobignum(sqrtn, bn_num_buckets)
    kernel += '    ulong x[' + str(bn_num_buckets) + '];\n'
    for i in range(bn_num_buckets):
        kernel += '    x['+str(i)+'] = ' + str(sqrtn[i]) + ';\n'
    kernel += '    x[' + str(bn_num_buckets-1) + '] += blockIdx.x * blockDim.x + threadIdx.x;\n'
    
    kernel += \
    """
    while (true) {
    """


    kernel += '        ulong tmp[' + str(2 * bn_num_buckets) + '];\n'
    kernel += '        clean(x);\n'
    kernel += '        square(x,tmp);\n'
    kernel += '        mod(tmp,n);\n'
    kernel += '        if (!eq(tmp, 0)) {\n'
    for f in factor_base:
        kernel += '            while (isfactorof(tmp, '+str(f)+')) {\n'
        kernel += '                smalldiv(tmp, '+str(f)+');\n'
        kernel += '            }\n'
    kernel += '            if (eq(tmp, 1)) {\n'
    kernel += '                flag = threadIdx.x + 1;\n'
    kernel += '            }\n'
    kernel += '        }\n'
    kernel += """        __syncthreads();
        if (flag && flag == threadIdx.x + 1) {\n"""
    for i in range(bn_num_buckets):
        kernel += '            out[blockIdx.x * ' + str(bn_num_buckets) + ' + ' + str(i) + '] = x[' + str(i) + '];\n'
    kernel += \
"""        }
        if (flag) {
            break;
        }\n"""
    kernel += '        x[' + str(bn_num_buckets-1) + '] += '+str(numThreads)+';\n'
    kernel += '    }\n'
    kernel += '\n}'
    return kernel


def checkPerm(lst, output, n):
    num = 1
    mod = 1
    for item in lst:
        num *= item
        mod *= (item * item) % n
    if int(math.sqrt(mod)) == math.sqrt(mod):
        output.append((num, mod))


def generatePerms(lst, remainder, output, n):
    if remainder != []:
        item = remainder.pop()
        generatePerms(lst[:], remainder[:], output, n)
        lst.append(item)
        generatePerms(lst, remainder, output, n)
    else:
        checkPerm(lst, output, n)

def frombignum(bignum):
    num = 0L
    for i in reversed(range(len(bignum))):
        num += bignum[i] * (bn_bucket_max ** (len(bignum) - i - 1))
    return num
    
def calc_bn_num_buckets(n):
    lg = math.ceil(math.log(n,2))
    print 'size of n:', lg, 'bits'
    return int(math.ceil(lg / bn_bucket_size))

def findBSNs(n):
    bn_num_buckets = calc_bn_num_buckets(n)
    GRID_DIM = len(factor_base) + 5
    dest = numpy.zeros(GRID_DIM * bn_num_buckets).astype(numpy.int64)    
    the_kernel = gen_bsmooth_kernel(n, GRID_DIM * block_dim, bn_num_buckets)
    bsmooth_module = SourceModule(the_kernel)
    bsmooth = bsmooth_module.get_function("do_bsmooth_kernel")
    with  Timer('Finding bsmooth numbers'):
        bsmooth(drv.InOut   (dest), block=(block_dim,1,1), grid=(GRID_DIM,1))
    bignums = dest.tolist()
    BSNs = []
    for i in range(GRID_DIM):
        bignum = bignums[i*bn_num_buckets:i*bn_num_buckets + bn_num_buckets]
        BSNs.append(frombignum(bignum))
    return BSNs

    

def gcd(a, b):
    while b > 0:
        (a, b) = (b, a % b)
    return int(a)

def isPrime(p):
    for f in factor_base:
        if p % f == 0:
            return False
        if f * f > p:
            break
    return True

def gen_factor_base(b):
    for n in range(2, b + 1):
        if isPrime(n):
            factor_base.append(n)

def do_it(n, b):

    gen_factor_base(b)
    BSNs = findBSNs(n)
    sqrts = []
    generatePerms([], BSNs, sqrts, n)
    factors = set()
    
    with Timer('Calculating factors'):
        for pair in sqrts:
            (num, mod) = pair
            mod = math.sqrt(mod)
            a,b = gcd(abs(num - mod), n), gcd(num + mod, n)
            if a != 1 and a != n:
                factors.add(a)
            if b != 1 and a != n:
                factors.add(b)
    print 'factors:', list(factors)
        
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'usage:', sys.argv[0], 'n', '[b]'
        sys.exit()
    
    n = int(sys.argv[1])
    if len(sys.argv) > 2:  b = int(sys.argv[2])
    else: b = 29
    do_it(n, b)
    
