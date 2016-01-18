/* 
 * pollsrv - a server to track a voting poll
 * 
 * Loren McGinnis
 * Chris Simmons
 */

#include <string.h>
#include "csapp.h"


#define NUM_CANDIDATES 3


/* You will implement these functions */
void handle_reset();
int  handle_vote();
int  send_vote_totals(int connfd);
int  send_candidates(int connfd);
int  parse(int connfd);
void init();
void *thread(void *vargp);

/* TODO: PUT ANY GLOBAL VARIABLES HERE */
unsigned int gpcounts[NUM_CANDIDATES], gnum_reading = 0, gwriter_waiting = 0;
char *gppcandidates[NUM_CANDIDATES];
char *gptitle;
pthread_mutex_t lock;
pthread_mutex_t write_mutex, read_mutex;
pthread_cond_t cv_writer_waiting, cv_num_reading;

/* TODO: Complete the following functions */


// handle_reset
// ------------
// Use this function to set all of your vote counters to zero

void handle_reset()
{
/*
    pthread_mutex_lock(&write_mutex);   // lock against writing
    
	// lock waiting
    gwriter_waiting = 1;    // a write operation is pending
    // unlock waiting

	// lock reading
    // while threads are reading, wait
    while (gnum_reading)
        pthread_cond_wait(&cv_num_reading, /* reading lock */);
	// unlock reading
*/



    //grab the lock
    pthread_mutex_lock(&lock);


    int i;
    for (i = 0; i < NUM_CANDIDATES; ++i)
        gpcounts[i] = 0;

    //release the lock
    pthread_mutex_unlock(&lock);



/*
    // zero each candidate's vote counter
    int i;
    for (i = 0; i < NUM_CANDIDATES; ++i)
        gpcounts[i] = 0;
        
    gwriter_waiting = 0;    // writing complete
    
    // broadcast to resume reading
    pthread_cond_broadcast(&cv_writer_waiting);
    
    pthread_mutex_unlock(&write_mutex); // unlock writing
*/


}

// handle_vote
// -----------
// Use this function to update your vote counts
// for a particular vote (ballot). Note that ballot 
// should be a 1-indexed vote (only 
// 1, 2, and 3 are valid votes)
// Returns: 0 if successful, non-zero value otherwise

int handle_vote(int ballot)
{
/*
    int ret = 1;
    
    pthread_mutex_lock(&write_mutex);   // lock against writing
    
    gwriter_waiting = 1;    // a write operation is pending

    // while threads are reading, wait
    while (gnum_reading)
        pthread_cond_wait(&cv_num_reading, NULL);
*/


    if (ballot < 1 || ballot > 3)
        return 0;
    //grab the lock
    pthread_mutex_lock(&lock);


    ++gpcounts[ballot - 1];

    //release the lock
    pthread_mutex_unlock(&lock);

    return 0;


/*
    // check for a valid ballot
    if (0 < ballot && ballot <= NUM_CANDIDATES)
    {
        ++gpcounts[ballot - 1];
        ret = 0;
    }
    
    gwriter_waiting = 0;    // writing complete

    // broadcast to resume reading
    pthread_cond_broadcast(&cv_writer_waiting);
    
    pthread_mutex_unlock(&write_mutex); // unlock writing

    return ret;
*/
}

// send_vote_totals
// ----------------
// Use this function to respond to a client 
// (noted by connfd) with a total of all the votes. 
// The response should be in the format "x,y,z\n", 
// where x, y, and z are the vote totals for 
// candidates 1, 2, and 3 respectively.
// Returns: 0 if successful, 1 if unsuccessful.

int send_vote_totals(int connfd)
{
/*
  int ret;
    
	// lock waiting
    // don't start reading when a writer is waiting to write
    while (gwriter_waiting)
        pthread_cond_wait(&cv_writer_waiting, waiting lock );
    // unlock waiting

    pthread_mutex_lock(&read_mutex);       // lock gnum_reading
    
    ++gnum_reading;
    
    pthread_mutex_unlock(&read_mutex);     // unlock gnum_reading
*/


    //grab the lock
    pthread_mutex_lock(&lock);


    // TODO: read

    //release the lock
    pthread_mutex_unlock(&lock);



/*
    pthread_mutex_lock(&read_mutex);       // lock gnum_reading
    
    --gnum_reading;

    // if no threads are reading, send signal to begin writing
    if (gnum_reading == 0)
        pthread_cond_signal(&cv_num_reading);

    pthread_mutex_unlock(&read_mutex);     // unlock gnum_reading
    
    // TODO: send
    

  return ret;
*/



}

// send_candidates
// ---------------
// Use this function to send the names of the 
// candidates. This should be in the form
// "title,x,y,z\n", where title is the name of
// the poll, and x, y, and z are the names
// of candidates 1, 2, and 3 respectively.
// Returns: 0 if successful, non-zero value otherwise

int send_candidates(int connfd)
{
  
  return 0;
}


// parse
// -----
// Use the parse function to read in a client's
// transmission, parse it, and react accordingly.
// Returns: 0 if successful, non-zero value otherwise.

int parse(int connfd) 
{
    return 0;
}

// init
// ----
// Use this function to initialize any global variables

void init()
{
    gptitle = "Frozen Dairy Election";
    gppcandidates[0] = "Chocalate";
    gppcandidates[1] = "McCandyCain";
    gppcandidates[2] = "Yes, Pecan!";
    
    pthread_mutex_init(&lock, NULL);
    pthread_cond_init(&cv_writer_waiting, NULL);
    pthread_cond_init(&cv_num_reading, NULL);

    handle_reset();
}

// thread
// ------
// All new threads should initially call this function.
// Use this function to handle incoming client transmissions
// and process them accordingly. The return value can be
// returned to the thread's creator.

void *thread(void *vargp)
{
  return NULL;
}

// main
// ----
// use this function to do any necessary initialization,
// and listen for incomming transmissions. On an 
// incoming transmission, you should spawn off a 
// thread to handle it.

int main(int argc, char **argv)
{
  return 0;
}

