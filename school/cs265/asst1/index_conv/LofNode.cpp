/* 
 * File:   LofTrie.cpp
 * Author: loren
 * 
 * Created on September 19, 2010, 2:40 AM
 */

#include "LofNode.h"

uint i_from_char(char c) {
    switch (c) {
        case 'A':
            return 0;
        case 'C':
            return 1;
        case 'G':
            return 2;
        case 'T':
            return 3;
        case '$':
            return 4;
    }
    cerr << "Unhandled character in trie: " << ((int) c) << "!!" << endl;
    return 0;
}

void LofNode::nullifyChildren() {
    for (int i = 0; i < SIGMA_SIZE; i++) {
        children_[i] = NULL;
    }
    is_leaf_ = true;
}

bool LofNode::isLeaf() {
    return is_leaf_;
}

void LofNode::pruneChildren() {
    for (int i = 0; i < SIGMA_SIZE; i++) {
        if (children_[i] != NULL) {
            delete children_[i];
            children_[i] = NULL;
        }
    }
    is_leaf_ = true;
}

void LofNode::addChild(char c, LofNode* child) {
    children_[i_from_char(c)] = child;
    is_leaf_ = false;
}

LofNode * LofNode::getChild(char c) {
    return children_[i_from_char(c)];
}

LofNode ** LofNode::getChildren() {
    return children_;
}

LofNode * LofNode::getParent() {
    return parent_;
}

LofNode::LofNode(uint start_i, uint level, char c, LofNode* parent) : start_i_(start_i), c_(c), parent_(parent), level_(level) {
    is_leaf_ = true;
    nullifyChildren();
}

LofNode::~LofNode() {
    pruneChildren();
}

