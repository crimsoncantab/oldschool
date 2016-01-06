/***************************************************************************
 * generate.c
 *
 * Computer Science 50
 * Problem Set 3
 *
 * Generates pseudorandom numbers in [0,LIMIT), one per line.
 *
 * Usage: generate n [s]
 *
 * where n is number of pseudorandom numbers to print
 * and s is an optional seed
 ***************************************************************************/
       
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define LIMIT 65536

int
main(int argc, char * argv[])
{
    // Checks for correct command line arguments
    if (argc != 2 && argc != 3)
    {
        printf("Usage: %s n [s]\n", argv[0]);
        return 1;
    }

    // Converts first argument to int and assigns it to n
    int n = atoi(argv[1]);

    /* Stores given seed with srand(), if one is not given
    uses time as a seed
    */
    if (argc == 3)
        srand((unsigned int) atoi(argv[2]));
    else
        srand((unsigned int) time(NULL));

    /* Prints the desired number of random numbers,
    modulated to fall within the range of LIMIT
    */
    for (int i = 0; i < n; i++)
        printf("%d\n", rand() % LIMIT);
}

