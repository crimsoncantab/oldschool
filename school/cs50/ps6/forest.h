/****************************************************************************
 * forest.h
 *
 * Computer Science 50
 * Problem Set 6
 *
 * Defines a forest for Huffman trees.
 ***************************************************************************/

#ifndef FOREST_H
#define FOREST_H

#include <cs50.h>

#include "tree.h"


/*
 * Plot
 *
 * Space for a tree in a forest.
 */

typedef struct plot
{
    Tree *tree;
    struct plot *next;
}
Plot;


/*
 * Forest
 *
 * A forest for Huffman trees, implemented as a singly linked list.
 */

typedef struct forest
{
    Plot *first;
}
Forest;


/*
 * Forest *
 * mkforest()
 *
 * Makes a forest, initially barren, returning pointer thereto
 * or NULL on error.
 */

Forest * mkforest();


/*
 * Tree *
 * pick(Forest *f)
 *
 * Removes a tree with lowest weight from the forest, returning
 * pointer thereto or NULL if forest is barren.
 */

Tree * pick(Forest *f);


/*
 * bool
 * plant(Forest *f, Tree *t)
 *
 * Plants a tree in the forest, provided that tree's frequency is non-0.
 *
 * Returns TRUE on success and FALSE on error.
 */

bool plant(Forest *f, Tree *t);


/*
 * bool
 * rmforest(Forest *f)
 *
 * Deletes a forest (and all of its trees).
 *
 * Returns TRUE on success and FALSE on error.
 */

bool rmforest(Forest *f);


#endif

