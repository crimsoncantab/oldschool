#include <stdio.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>

void kernel_bsmooth(int n);

int main()
{
    unsigned int timer;
    cutilCheckError(  cutCreateTimer(&timer)  );

    cutilCheckError(  cutResetTimer(timer)    );
    cutilSafeCall( cudaThreadSynchronize() );
    cutilCheckError( cutStartTimer(timer) );

    kernel_bsmooth(5944405);

    cutilSafeCall( cudaThreadSynchronize() );
    cutilCheckError( cutStopTimer(timer) );

    float totalTime = cutGetTimerValue(timer);
    printf("Time: %f\n", totalTime);
    return 0;
}
