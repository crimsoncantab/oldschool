/***************************************************************************
 * fifteen.c
 *
 * Computer Science 50
 * Problem Set 3
 *
 * Implements The Game of Fifteen (generalized to d x d).
 *
 * Usage: fifteen d
 *
 * whereby the board's dimensions are to be d x d,
 * where d must be in [DIM_MIN,DIM_MAX]
 *
 * Note that usleep is obsolete, but it offers more granularity than
 * sleep and is simpler to use than nanosleep; `man usleep` for more.
 ***************************************************************************/
 
#define _XOPEN_SOURCE 500

#include <cs50.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>


/* constants */
#define DIM_MIN 3
#define DIM_MAX 9


/* global board */
int board[DIM_MAX][DIM_MAX];
int d;


// prototypes
void clear();
void greet();
void init();
void draw();
bool move();
bool won();


int
main(int argc, char * argv[])
{
    // greet user with instructions
    greet();

    // ensure proper usage
    if (argc != 2)
    {
        printf("Usage: %s d\n", argv[0]);
        return 1;
    }

    // ensure valid dimensions
    d = atoi(argv[1]);
    if (d < DIM_MIN || d > DIM_MAX)
    {
        printf("Board must be between %d x %d and %d x %d, inclusive.\n",
               DIM_MIN, DIM_MIN, DIM_MAX, DIM_MAX);
        return 2;
    }

    // initialize the board
    init();

    // accept moves until game is won
    while (TRUE)
    {
        // clear the screen
        clear();

        // draw the current state of the board
        draw();

        // check for win
        if (won())
        {
            printf("ftw!\n");
            break;
        }

        // prompt for move
        printf("Tile to move: ");
        int tile = GetInt();

        // move if possible, else report illegality
        if (!move(tile))
        {
            printf("\nIllegal move.\n");
            usleep(100000);
        }

        // sleep thread for animation's sake
        usleep(100000);
    }
}


/*
 * void
 * clear()
 *
 * Clears screen using ANSI escape sequences.
 */

void
clear()
{
    printf("\033[2J");
    printf("\033[%d;%dH", 0, 0);
}


/*
 * void
 * greet()
 *
 * Greets player.
 */

void
greet()
{
    clear();
    printf("WELCOME TO THE GAME OF FIFTEEN\n");
    usleep(2000000);
}


/*
 * void
 * init()
 *
 * Initializes the game's board with tiles (numbered 1 through d*d - 1),
 * i.e., fills 2D array with values but does not actually print them).  
 */

void
init()
{
	int n;
	int arr[d*d - 1];
	//makes arr[] contain each number
	for (int i = 0; i < d*d - 1; i++)
		arr[i] = i + 1;
	//assigns random number from arr[] to each square in board[]
	srand((unsigned int) time(NULL));
    for (int i = 0; i < d*d - 1; i++) {
   		n = rand() % (d*d - 1 - i);
   		board[i / d][i % d] = arr[n];
   		//removes used number from arr
   		for (int j = n; j < d*d - 1 - i; j++)
   			arr[j] = arr[j+1];
    }
    //ensures last spot is "empty"
    board[d-1][d-1] = 0;
    //checks solvability
    int counter = 0, val;
    for (int i = 0; i < d*d - 1; i++) {
    	val = board[i / d][i % d];
    	for (int j = i + 1; j < d*d - 1; j++)
    		if (val > board[j /d][j % d])
    			counter++;
    }
    printf("counter = %d\n", counter);
    usleep(500000);
    //swaps last two values if counter is odd
    if (counter % 2 == 1) {
    	int temp = board[d][d - 3];
    	board[d][d - 3] = board[d][d - 2];
    	board[d][d - 2] = temp;
    	exit(0);
    }
    
    return;
}


/* 
 * void
 * draw()
 *
 * Prints the board in its current state.
 */

void
draw()
{
	for (int k = 0; k <= d*5; k++)
		printf("-");
	printf("\n");
	for (int i = 0; i < d; i++) {
		for (int j = 0; j < d; j++) {
			if (board[i][j] == 0)
				printf("|    ");
			else
    			printf("|%3d ",board[i][j]);
    	}
    	printf("|\n");
    	for (int k = 0; k <= d*5; k++)
    		printf("-");
    	printf("\n");
    }
}


/* 
 * bool
 * move(int tile)
 *
 * If tile borders empty space, moves tile and returns TRUE, else
 * returns FALSE. 
 */

bool
move(int tile)
{
    //checks bogus number
    if (tile >= d*d)
    	return FALSE;
    //checks adjacency
    int r, c, r0, c0;
    for (int i = 0; i < d*d; i++) {
    	if (board[i / d][i % d] == tile) {
    		r = i / d;
    		c = i % d;
    	}
    	if (board[i / d][i % d] == 0) {
    		r0 = i / d;
    		c0 = i % d;
    	}
    }
    if (abs(r - r0) + abs(c - c0) == 1) {
    	board[r0][c0] = tile;
    	board[r][c] = 0;
    	return TRUE;
    }
    return FALSE;
}


/*
 * bool
 * won()
 *
 * Returns TRUE if game is won (i.e., board is in winning configuration), 
 * else FALSE.
 */

bool
won()
{
    for (int i = 1; i < d*d; i++) {
    	if (board[(i - 1) / d][(i - 1) % d] != i)
    		return FALSE;
    }
	return TRUE;
}

