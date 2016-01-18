/*
 * mm-naive.c - The fastest, least memory-efficient malloc package.
 * 
 * In this naive approach, a block is allocated by simply incrementing
 * the brk pointer.  A block is pure payload. There are no headers or
 * footers.  Blocks are never coalesced or reused. Realloc is
 * implemented directly using mm_malloc and mm_free.
 *
 * NOTE TO STUDENTS: Replace this header comment with your own header
 * comment that gives a high level description of your solution.
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h> // for memcpy

#include "mm.h"
#include "memlib.h"

/*********************************************************
 * NOTE TO STUDENTS: Before you do anything else, please
 * provide your team information in the following struct.
 ********************************************************/
team_t team = {
    /* Team name */
    "Lacheses",
    /* First member's full name */
    "Chris Simmons",
    /* First member's email address */
    "csimmons@fas.harvard.edu",
    /* Second member's full name (leave blank if none) */
    "Loren McGinnis",
    /* Second member's email address (leave blank if none) */
    "mcginn@fas.harvard.edu"
};

/* single word (4) or double word (8) alignment */
#define ALIGNMENT 16
#define WSIZE 4
#define DSIZE 8
#define CHUNKSIZE (1<<12)
#define OVERHEAD 8
#define LISTCOUNT 33

#define MAX(x, y) ((x) > (y)? (x) : (y))

#define PACK(size, alloc) ((size) | (alloc))

/* Read and write a word at address p */
#define GET(p)      (*(size_t*) (p))
#define PUT(p, val) (*(size_t*) (p) = (val))

/* Read the block pointer and allocated fields from address p */ 
#define GET_SIZE(p)     (GET(p) & ~0x1)
#define GET_PBLOCK(p)   (GET(p) & ~0x1)
#define GET_ALLOC(p)    (GET(p) & 0x1)

/* Given block pointer p, get address of header or footer */
#define PHDR(pb)    ((char*)(pb) - WSIZE)    
#define PFTR(pb)    ((char*)(pb) + GET_SIZE(PHDR(pb)) - DSIZE)

/* Find the next or previous block in the heap */
#define PNEXT_BLK(pb)	((char*)(pb) + GET_SIZE(((char*)(pb) - WSIZE)))
#define PPREV_BLK(pb)	((char*)(pb) - GET_SIZE(((char*)(pb) - DSIZE)))

/* 
 * Find the addresses where pointers to predicate and succesor blocks
 * are stored 
 */
#define PP_PRED(pb)		((char**)(pb))
#define PP_SUCC(pb)		((char**)((char*)(pb) + WSIZE))

// Find the address of the predicate or successor block
// in the segregated list
#define P_SUCC(pb)   ((char*)(GET(PP_SUCC(pb))))
#define P_PRED(pb)   ((char*)(GET(PP_PRED(pb))))
 
/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(size) (((size) + (ALIGNMENT-1)) & ~0xf)

#define SIZE_T_SIZE (ALIGN(sizeof(size_t)))

// global variables
char **gpplists; // points to an array of pointers to segregated lists
char *gpepilogue;

// prototypes
int mm_init(void);
static void *extend_heap(size_t size);
void *mm_malloc(size_t size);
void mm_free(void *pb);
void *mm_realloc(void *ptr, size_t size);
static void *coalesce(void *pb);
static void remove_from_list(void *pb);
static void push(void *pb);
static int size_to_index(size_t size);
static void print_lists();
static void move_epilogue(char *pepilogue);

/* 
 * mm_init - initialize the malloc package.
 */
int mm_init(void)
{
	char *pheap;
	char **pplists;	// a local substitute for gpplists

    //create initial empty heap
    int listsize = ALIGN(LISTCOUNT * sizeof(char*));
    if ((pheap = mem_sbrk(listsize + 4 * WSIZE)) == NULL)
        return -1;
        
    pplists = (char**)pheap;    // place the array at the start of the heap
	int i;
    for(i = 0; i < LISTCOUNT; ++i)	// initialize the array
		 pplists[i] = NULL;
	gpplists = pplists; // initialize gpplists
    
    pheap += listsize;  // move pheap to the end of the array
    
    PUT(pheap, 0);
    PUT(pheap+WSIZE, PACK(OVERHEAD, 1)); //prologue header
    PUT(pheap+DSIZE, PACK(OVERHEAD, 1)); //prologue footer
    PUT(pheap+DSIZE+WSIZE, PACK(0, 1)); //epilogue header
    
    move_epilogue(pheap+DSIZE+WSIZE);
    
    if ((char*)extend_heap(CHUNKSIZE) == NULL)
        return -1;
	
    printf("After init:\n");
    print_lists();
    return 0;
}

// extends the heap by MAX(size, CHUNKSIZE) bytes
// returns a pointer to the last block in the heap,
// returns null if heap could not be extended
static void *extend_heap(size_t size)
{
    char *p;

	size = MAX(size, CHUNKSIZE);

    // Allocate a multiple of ALIGNMENT bytes to maintain alignment
    size = ALIGN(size);
    
    if ((int)(p = mem_sbrk(size)) == -1)
        return NULL;

    /* Initialize free block header/footer and the epilogue header */
    PUT(PHDR(p), PACK(size, 0));
    PUT(PFTR(p), PACK(size, 0));
    PUT(PHDR(PNEXT_BLK(p)), PACK(0, 1));
    
    move_epilogue(PHDR(PNEXT_BLK(p)));
    
    push(p); // place the newly freed block
	return coalesce(p);
}

/* 
 * mm_malloc - Allocate a block by incrementing the brk pointer.
 *     Always allocate a block whose size is a multiple of the alignment.
 */
void *mm_malloc(size_t size)
{
    //printf("\nbefore malloc:\n");
    print_lists();
	
    char *pb, *pbest; 
    int this_size, best_size, excess, index;
	
    size = ALIGN(size + OVERHEAD);		// align size
	index = size_to_index(size);	// get index of appropriate list
	printf("index for malloc = %d\n",index);
    fflush(0);

	// iterate through lists of increasing size until a block *pbest is found
	pbest = NULL;
	pb = gpplists[index];
	if (pb)	// if there are any free blocks of this index's size
	{
		pbest = pb;
		best_size = GET_SIZE(PHDR(pbest));
		
	    pb = P_SUCC(pb);
		while (pb && size != best_size)
		{
			// move to the next block in the list
			this_size = GET_SIZE(PHDR(pb));
			
			// if this *pb is a better fit than the current *pbest
			if (this_size < best_size && size <= this_size)
			{
				pbest = pb;
				best_size = this_size;
			}
			pb = P_SUCC(pb);
		}
	}	
	for (++index; index < LISTCOUNT && pbest == NULL; ++index)
	{
		pb = gpplists[index];
		if (pb)	// if there are any free blocks of this index's size
		{
			pbest = pb;
			best_size = GET_SIZE(PHDR(pbest));
			
			pb = P_SUCC(pb);    // move to the next block in the list
			while (pb != NULL)
			{
	//		    printf("pb = %x\n",(int)pb);
                this_size = GET_SIZE(PHDR(pb));
				
				// if this *pb is a better fit than the current *pbest
				if (this_size < best_size && size <= this_size)
				{
					pbest = pb;
					best_size = this_size;
				}
			    pb = P_SUCC(pb);    // move to the next block in the list
			}
		}	
	}

	if (pbest == NULL)
    {
        pbest = extend_heap(size);
        best_size = GET_SIZE(PHDR(pbest));
    }
	
    if (pbest)	// if a block was found
	{
        // remove the block from it's list of free blocks
        remove_from_list(pbest); 
       
        // excess: how many unused bytes would be in this block
        // were the entire block allocated for this request
        // excess will always be a multiple of ALIGNMENT b/c
        // it is the difference of multiples of ALIGNMENT
        excess = best_size - size;
        
        if (excess >= ALIGNMENT)
		{
            // mark the start of pbest as a free block of size excess
            PUT(PHDR(pbest), PACK(excess, 0));
            PUT(PFTR(pbest), PACK(excess, 0));
            push(pbest);    // place the start of pbest in a list
            
            printf("Header of excess block: 0x%x\n", *(int*)PHDR(pbest));
            printf("Footer of excess block: 0x%x\n", *(int*)PFTR(pbest));
            
            pbest += excess;
            best_size -=excess;
        }
        
        // mark *pbest as allocated
		PUT(PHDR(pbest), PACK(best_size, 1));
		PUT(PFTR(pbest), PACK(best_size, 1));
        
        printf("Header of malloced block: 0x%x\n", *(int*)PHDR(pbest));
        printf("Footer of malloced block: 0x%x\n", *(int*)PFTR(pbest));
	}
    //printf("\nafter malloc:\n");
    print_lists();
	return pbest;
}

/*
 * mm_free - free a block of memory
 */
void mm_free(void *pb)
{
    //printf("\nbefore free:\n");
    print_lists();

    size_t size = GET_SIZE(PHDR(pb));

    if (!GET_ALLOC(PHDR(pb)))
    {
        //return;
        //exit(1);
    }

    // set the allocated bits to zero
	PUT(PHDR(pb), PACK(size, 0));
	PUT(PFTR(pb), PACK(size, 0));
	
    push(pb);   // place *pb in a list
    coalesce(pb);

    //printf("\nafter free:\n");
    print_lists();
}

/*
 * mm_realloc - Implemented simply in terms of mm_malloc and mm_free
 */
void *mm_realloc(void *ptr, size_t size)
{
    void *oldptr = ptr;
    void *newptr;
    size_t copySize;
    
    newptr = mm_malloc(size);
    if (newptr)
	{     
		copySize = *(size_t *)((char *)oldptr - SIZE_T_SIZE);
		if (size < copySize)
		  copySize = size;
		memcpy(newptr, oldptr, copySize);
		mm_free(oldptr);
	}
	return newptr;
}

// coalesces *pb w/ adjacent blocks
static void *coalesce(void *pb)
{
	//get pointers to previous and next blocks
	char *pprev = PPREV_BLK(pb);
	char *pnext = PNEXT_BLK(pb);
    printf("prev = %x\nnext = %x\n",(int)pprev,(int)pnext);
    fflush(0);

	// tells whether the prev block is allocated
	size_t prev_alloc = GET_ALLOC(PHDR(pprev));

	// tells whether the next block is allocated
	size_t next_alloc = GET_ALLOC(PHDR(pnext));

    printf("prev_alloc = %d\nnext_alloc = %d\n",(int)prev_alloc,(int)next_alloc);
    fflush(0);
    
    printf("prev_size: 0x%x\n", GET_SIZE(PHDR(pprev))); 
    printf("next_size: 0x%x\n", GET_SIZE(PHDR(pnext))); 
	
    size_t size = GET_SIZE(PHDR(pb));	// size of *pb
    
    printf("value of size before coa: 0x%x\n",size);
	// if either of the adjoining blocks is free
	if (!prev_alloc || !next_alloc)
	{
		// remove *pb from its list
		remove_from_list(pb);

		if (!next_alloc)
		{
			remove_from_list(pnext);	// remove next from its list
			size += GET_SIZE(PHDR(pnext));
		}

		if (!prev_alloc)
		{
			remove_from_list(pprev);	// remove prev from its list
			size += GET_SIZE(PHDR(pprev));
		    pb = PPREV_BLK(pb); // move pb to the start of the previous block
		}
		
		PUT(PHDR(pb), PACK(size, 0));	// change header
		PUT(PFTR(pb), PACK(size, 0));	// change footer
		
		// place *pb in a segregated list		
        push(pb);
	}
    printf("value of pb after coalescing: %x\nvalue of size: 0x%x\n\n",(int)pb,GET_SIZE(PHDR(pb)));
    fflush(0);
	
    return pb;
}

// removes a block from the list that it's in
static void remove_from_list(void *pb)
{
    int index;
	char *psucc = P_SUCC(pb);
	char *ppred = P_PRED(pb);
	
    if (ppred)
	    PUT(PP_SUCC(ppred), (size_t)psucc); // link PRED to SUCC
    else
    {
        // link the root to succ
	    index = size_to_index(GET_SIZE(PHDR(pb)));
        gpplists[index] = psucc;
    }
    
    if (psucc)
        PUT(PP_PRED(psucc), (size_t)ppred); // link SUCC to PRED
}

// place *pb in the appropriate segregated list
static void push(void *pb)
{
    char **pplists = gpplists;  // local copy of gpplists
    
	//get index for block's size
	int index = size_to_index(GET_SIZE(PHDR(pb)));

	//put block at front of appropriate list
	PUT(PP_SUCC(pb), (size_t)pplists[index]);   // link *pb to SUCC
    if (P_SUCC(pb))
        PUT(PP_PRED(P_SUCC(pb)), (size_t)pb);   // link SUCC to *pb
    
	PUT(PP_PRED(pb), (size_t)NULL);				// set *pb's PRED to NULL
	pplists[index] = pb;				// point the root to *pb
}

// takes a size in bytes and returns the index of the
// appropriate segregated list
static int size_to_index(size_t size)
{
	size = ALIGN(size) / 16;
	
	//returns the index for sizes 1, 2, and 3 (times 16 bytes)
	if (size < 4) 
		return size - 1;

	//returns the index for powers of 2 above 3
	int log = 1;

	while (size) {
		++log;
		size >>= 1;
	}
	return log;
}

static void print_lists()
{
    char *pb, **pplists;
   return; 
    pplists = gpplists;
    int list;
    for (list = 0; list < LISTCOUNT; ++list)
    {
        printf("list %d:\n", list);
        
        pb = pplists[list];
        while(pb)
        {
            printf("0x%x, size 0x%x\n", (int)pb, GET_SIZE(PHDR(pb)));
            pb = P_SUCC(pb);
            
        }
    }
}

static void move_epilogue(char *pepilogue)
{
    gpepilogue = pepilogue;
    printf("pepilogue: %x\n", pepilogue);
}
