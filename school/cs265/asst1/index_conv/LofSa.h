/* 
 * File:   LofSa.h
 * Author: loren
 *
 * Created on September 19, 2010, 2:35 AM
 */

#ifndef _LOFSA_H
#define	_LOFSA_H
#include <iostream>
#include <string>
#include "LofNode.h"
#define F 4

using namespace std;

struct lof {
    uint lcp;
    uint i;
    char fr[F];
};

class LofSa {
public:
    LofSa(LofNode * trie_root, string lof_file);
    virtual ~LofSa();
private:
    LofNode * trie_root_;
    ifstream * lof_;

};

#endif	/* _LOFSA_H */

