/* 
 * File:   LofSa.cpp
 * Author: loren
 * 
 * Created on September 19, 2010, 2:35 AM
 */

#include "LofSa.h"
#include <iostream>
#include <fstream>

LofSa::LofSa(LofNode * trie_root, string lof_file) {
    trie_root_ = trie_root;
    lof_ = new ifstream(lof_file.c_str());
}

LofSa::~LofSa() {
    delete trie_root_;
    lof_->close();
    delete lof_;
}

