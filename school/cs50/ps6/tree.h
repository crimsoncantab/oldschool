/****************************************************************************
 * tree.h
 *
 * Computer Science 50
 * Problem Set 6
 *
 * Defines a tree for Huffman coding.
 ***************************************************************************/

#ifndef TREE_H
#define TREE_H


/*
 * Tree
 *
 * Defines a Huffman tree (or, more generally, a node thereof).
 */

typedef struct tree
{
    unsigned char symbol;
    int frequency;
    struct tree *left;
    struct tree *right;
}
Tree;


/*
 * Tree *
 * mktree()
 *
 * Makes a tree with no children, returning pointer thereto
 * or NULL on error.
 */

Tree * mktree();


/*
 * void
 * rmtree(Tree *t)
 *
 * Deletes a tree and all its descedants, if any.
 */

void rmtree(Tree *t);


#endif

