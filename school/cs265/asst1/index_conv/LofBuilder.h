/* 
 * File:   LofBuilder.h
 * Author: loren
 *
 * Created on September 19, 2010, 2:31 AM
 */

#ifndef _LOFBUILDER_H
#define	_LOFBUILDER_H
#include <string>

#include "LofSa.h"
#include "LofNode.h"
#include "LoArray.h"

using namespace std;
void saveTrie(LofNode * node, ostream * out);
LofNode * loadTrie(istream in);

class LofBuilder {
public:
    LofBuilder(LoArrayIterator * lo_it, string text_file, string trie_file, string lof_file);
    virtual ~LofBuilder();
    LofSa * build();
private:
    void gotoEntry(lo_entry entry, int offset = 0);
    void createLof(LofNode * root, ofstream * out, int * i);
    LofNode * getTrie();
    void readCharsToFringe(char * fringe);
    int handleEntry(lo_entry entry, LofNode* node, uint counter);
    LofNode * buildOntoNode();
    char readTextChar();
    ifstream * text_;
    LoArrayIterator * lo_it_;
    string lof_filename_;
    string trie_filename_;

};

#endif	/* _LOFBUILDER_H */

