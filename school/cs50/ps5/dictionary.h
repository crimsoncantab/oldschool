/****************************************************************************
 * dictionary.h
 *
 * Computer Science 50
 * Problem Set 5
 *
 * Declares a dictionary's functionality.
 ***************************************************************************/

#include <cs50.h>

/*
 *Trie style node, with pointers to up to 27 other nodes
 *and a boolean to indicate the end of a word
 */

typedef struct _node {
    bool isEnd;
    struct _node * next[27];
}
node;

/*
 * bool
 * check(char *word)
 *
 * Returns TRUE if word is in dictionary else FALSE.
 */

bool check(char *word);

/*
 * bool
 * load(char *dict)
 *
 Loads dict into memory.  Returns TRUE if successful else FALSE.
 */

bool load(char *dict);

/*
 * int
 * size()
 *
 * Returns number of words in dictionary if loaded else 0 if not yet loaded.
 */

int size();

