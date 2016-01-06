/****************************************************************************
 * dictionary.c
 *
 * Computer Science 50
 * Problem Set 5
 *
 * Implements a dictionary's functionality.
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "dictionary.h"

#define LENGTH 45 
#define NEXT_LEN 27

node * root = NULL;
int dictsize = 0;
int nodesize = sizeof(node);
/*
 * bool
 * check(char *word)
 *
 * Returns TRUE if word is in dictionary else FALSE.
 */

bool 
check(char *word)
{
    node * temp = root;
    int i = 0, index;
    
    //loops through word until found/not found
    while (TRUE) {
        //if at end of word, checks trie for the endpoint
        if (word[i] == '\0') {
            if (temp->isEnd == 1)
                return TRUE;
            return FALSE;
        }

        //creates index from character
        if ((int) word[i] > 64)
            index = (word[i] | 32) - 97; //lowercases letters
        else
            index = 26; //for apostrophe
        
        //checks if pointer exists for index
        if (temp->next[index] != NULL)
            temp = temp->next[index];
        else
            return FALSE;

        i++;
    }
}


/*
 * bool
 * load(char *dict)
 *
 * Loads dict into memory.  Returns TRUE if successful else FALSE.
 */

bool 
load(char *dict)
{
    //initializes root node
    root = (node *) malloc(nodesize);
    int index;

    //opens file
    FILE * d = fopen(dict, "r");
    if (d == NULL)
        return FALSE;
    
    //reads chars from file
    node * temp = root;
    for (int c = fgetc(d); c != EOF; c = fgetc(d)) {
        
        //adds char to trie until not a letter or apostrophe
        if (c > 32) {

            //creates index from current character
            if (c == '\'')
                index = 26; //for apostrophe
            else
                index = c - 97;

            //creates a new node
            if (temp->next[index] == NULL)
                temp->next[index] = (node *) malloc(nodesize);
            temp = temp->next[index];
            if (temp == NULL)
                return FALSE;
            
        }

        //indicates end of trie and jumps back to root
        else {
            temp->isEnd = 1;
            temp = root;
            dictsize++;
        }
    }
    fclose(d);
    return TRUE;
}

/*
 * int
 * size()
 *
 * Returns number of words in dictionary if loaded else 0 if not yet loaded.
 */

int 
size()
{
    return dictsize;
}
