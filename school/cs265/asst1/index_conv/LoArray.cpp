/* 
 * File:   LOArray.cpp
 * Author: loren
 * 
 * Created on September 19, 2010, 1:37 AM
 */

#include <fstream>


#include <iosfwd>
#include <string>
#include <iostream>

#include "LoArray.h"

//LoArrayIterator::LoArrayIterator(string datafile, string lo_index_file) {
LoArrayIterator::LoArrayIterator(string lo_index_file) {
//    text = new ifstream(datafile.c_str());
    saindex_ = new ifstream(lo_index_file.c_str());
    saindex_->read(reinterpret_cast<char *> (&next_), sizeof (next_));
}

bool LoArrayIterator::hasNext() {
    return saindex_->good();
}

lo_entry LoArrayIterator::getNext() {
    lo_entry ret = next_;
    saindex_->read(reinterpret_cast<char *> (&next_), sizeof (next_));
    return ret;
}

void LoArrayIterator::reset() {
    saindex_->clear();
    saindex_->seekg(0, ios::beg);
    saindex_->read(reinterpret_cast<char *> (&next_), sizeof (next_));
}

LoArrayIterator::~LoArrayIterator() {
    saindex_->close();
    delete saindex_;
}

