#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <time.h>

#include "mm.h"
#include "memlib.h"
#include "fsecs.h"
#include "config.h"
int main()
{
    
    mem_init();
    mm_init();
    
    int *p[4];
    p[0] = mm_malloc(0x100);
    p[1] = mm_malloc(0x10);
    p[2] = mm_malloc(0x100);
    
    
    mm_free(p[0]);
    mm_free(p[2]);

    print_lists();
    
    return 0;
}


