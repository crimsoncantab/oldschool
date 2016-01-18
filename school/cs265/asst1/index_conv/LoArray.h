/* 
 * File:   LOArray.h
 * Author: loren
 *
 * Created on September 19, 2010, 1:37 AM
 */

#ifndef _LOARRAY_H
#define	_LOARRAY_H

#include <cstdlib>

struct lo_entry {
    uint i;
    uint lcp;
};

using namespace std;

class LoArrayIterator {
public:
    LoArrayIterator(string saindex_filename);
    bool hasNext();
    lo_entry getNext();
    virtual ~LoArrayIterator();
    void reset();
private:
    ifstream * saindex_;
    lo_entry next_;

};

#endif	/* _LOARRAY_H */

