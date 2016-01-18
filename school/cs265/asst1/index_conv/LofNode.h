/* 
 * File:   LofTrie.h
 * Author: loren
 *
 * Created on September 19, 2010, 2:40 AM
 */

#ifndef _LOFTRIE_H
#define	_LOFTRIE_H

#include <iostream>
#include <cstdlib>
using namespace std;

#define SIGMA_SIZE 5



class LofNode {
public:
    LofNode(uint start_i, uint level, char c, LofNode * parent);
    ~LofNode();
    LofNode * getChild(char c);
    LofNode ** getChildren();
    LofNode * getParent();
    void pruneChildren();
    bool isLeaf();
    void addChild(char c, LofNode * child);
    const uint start_i_;
    uint end_i_;
    const char c_;
    const uint level_;

private:
    void nullifyChildren();
    bool is_leaf_;
    LofNode * parent_;
    LofNode * children_[SIGMA_SIZE];

};

//class LofTrie {
//public:
//    LofTrie();
//    virtual ~LofTrie();
//    LofNode * getRoot();
//private:
//    LofNode * root;
//};

#endif	/* _LOFTRIE_H */

