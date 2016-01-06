#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "forest.h"
#include "huffile.h"
#include "tree.h"


int sum = 0;
int freq[SYMBOLS];
unsigned char * bits[SYMBOLS];
unsigned char buff[SYMBOLS];

bool getFreq(char *);
Huffeader * getHeader(); 
Forest * growForest();
Tree * huffTree(Forest *);
bool assignBits(Tree *);
bool traverse(Tree *, int);

int main(int argc, char * argv[]) {
    
    //ensure proper usage
    if (argc != 3) {
        printf("Usage: %s infile outfile",argv[0]);
        return 1;
    }

    //fills freq[] with infile character frequencies
    if(!getFreq(argv[1])) {
        printf("Could not open %s\n",argv[1]);
        return 2;
    }
    //make forest
    Forest * f = growForest();
    if (f == NULL) {
        printf("Could not grow forest\n");
        return 3;
    }
    //creates huffman binary tree out of forest
    Tree * root = huffTree(f);
    if (root == NULL) {
        printf("Could not make binary tree\n");
        return 4;
    }
    //assigns bits according to tree to characters
    if (!assignBits(root)) {
        printf("Tree is empty\n");
        rmtree(root);
        for (int i = 0; i < SYMBOLS; i++)
            free(bits[i]);
        return 5;
    }
    
    //create huffile
    Huffile * huffout = hfopen(argv[2], "w");
    if (huffout == NULL) {
        printf("Could not open %s\n",argv[2]);
        rmtree(root);
        for (int i = 0; i < SYMBOLS; i++)
            free(bits[i]);
        return 4;
    }

    //creates header
    Huffeader * header = getHeader();

    //write header to huffile
    if (!hwrite(header, huffout)) {
        printf("Could not write out header\n");
        free(header);
        rmtree(root);
        for (int i = 0; i < SYMBOLS; i++)
            free(bits[i]);
        return 6;
    }
    free(header);
    //write letters to file
    FILE * infile = fopen(argv[1], "r");
    unsigned char c; 
    while (TRUE) {
        c=(unsigned char) fgetc(infile);
        if (feof(infile))
            break;
        for (int i = 0; bits[(int)c][i] != '\0'; i++)
            bwrite(bits[(int)c][i] - 48, huffout);
    }
    hfclose(huffout);
    fclose(infile);
    rmtree(root);
    for (int i = 0; i < SYMBOLS; i++)
        free(bits[i]);
    return 0;
}

//counts frequencies of characters in file
bool getFreq(char * filename) {
    
    //attempt opening file to be huffed
    FILE * infile = fopen(filename, "r");
    if (infile == NULL) {
        printf("Could not open file %s",filename);
        return FALSE;
    }

    //fill array of frequencies
    unsigned char c;
    for (int i = 0; i < SYMBOLS; i++)
        freq[i] = 0;
    while (TRUE) {
        c=(unsigned char) fgetc(infile);
        if (feof(infile))
            break;
        freq[(int)c]++;
        sum++;
    }
    fclose(infile);
    return TRUE;
}

//creates header of huffile with given frequencies
Huffeader * getHeader() {

    Huffeader * header =(Huffeader *) malloc(sizeof(Huffeader));
    if (header == NULL)
        return NULL;
    header->magic = MAGIC;
    for (int i = 0; i < SYMBOLS; i++)
        header->frequencies[i] = freq[i];
    header->checksum = sum;
    return header;
}

//creates a forest using file frequencies
Forest * growForest() {

    Forest * f = mkforest();
    if (f == NULL) 
        return NULL;
    for (int i = 0; i < SYMBOLS; i++) {
        Tree * temp = mktree();
        if (temp == NULL) {
            rmforest(f);
            return NULL;
        }
        temp->symbol = i;
        temp->frequency = freq[i];
        if(!plant(f, temp))
            free(temp);
    }
    return f;
}

//returns root of binary huffman tree
Tree * huffTree(Forest * f) {
    Tree * first;
    Tree * second;
    while (TRUE) {
        first = pick(f);
        if (first == NULL) {
            rmforest(f);
            return NULL;
        }
        second = pick(f);
        if (second == NULL) {
            rmforest(f);
            return first;
        }
        Tree * parent = mktree();
        if (parent == NULL) {
            rmforest(f);
            return NULL;
        }
        parent->left = (Tree *) first;
        parent->right =(Tree *) second;
        parent->frequency = first->frequency + second->frequency;
        plant(f, parent);
    }
}
//assigns bits to all chars in tree
bool assignBits(Tree * t) {
    return traverse(t, 0);
}

//traverses tree and copies a buffer string of bits to 
//the corresponding character
bool traverse(Tree * t, int l) {
    unsigned char c;
    if (t == NULL)
        return FALSE;
    if (t->left == NULL && t->right == NULL) {
        c = (unsigned char)t->symbol;
        buff[l] = '\0';
        bits[(int)c] = (unsigned char *) malloc(sizeof(char) * (l+1));
        for (int i = 0; i <= l; i++)
            bits[(int)c][i] = buff[i];
    }
    buff[l] = '0';
    traverse(t->left, l+1);
    buff[l] = '1';
    traverse(t->right, l+1);
    return TRUE;
}
