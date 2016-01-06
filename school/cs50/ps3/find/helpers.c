/**************************************************************************** 
 * helpers.c
 *
 * Computer Science 50
 * Problem Set 3
 *
 * Helper functions for Problem Set 3.
 ***************************************************************************/
       
#include <cs50.h>
#include <stdio.h>
#include <stdlib.h>
#include "helpers.h"

/*
 * bool
 * search(int value, int array[], int n)
 *
 * Returns TRUE iff value is in array of n values.
 */

bool 
search(int value, int array[], int n)
{
    int l = 0, h = n - 1,  m;
    while (l < h) {
    	m = (h-l)/2 + l;
    	if (value == array[m])
    		return TRUE;
    	if (value < array[m])
    		h = m;
    	else if (value > array[m])
    		l = m + 1;
    }
    return FALSE;
}


/*
 * void
 * sort(int values[], int n)
 *
 * Sorts array of n values.
 */

void 
sort(int values[], int n)
{
	const int MAX_SIZE = 65536;
	//initialize array of length MAX_SIZE
    int *arr = (int *) malloc(MAX_SIZE * sizeof(int));
    //set all values of arr[] to 0
    for (int i = 0; i < MAX_SIZE; i++)
    	arr[i] = 0;
    //counts occurences of each value in values[]
    //and stores in arr[]
    for (int j = 0; j < n; j++)
    	arr[values[j]]++;
    //returns occurences of values to values[], sorted
    int l = 0;
    	for (int k = 0; k < MAX_SIZE; k++) {
    		while (arr[k] != 0) {
    			arr[k]--;
    			values[l] = k;
    			l++;
    		}
    	}
    return;
}
