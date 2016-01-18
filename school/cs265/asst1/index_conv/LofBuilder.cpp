/* 
 * File:   LofBuilder.cpp
 * Author: loren
 * 
 * Created on September 19, 2010, 2:31 AM
 */

#include <fstream>

#include "LofBuilder.h"

#define B 2

struct lof_node {
    uint start_i;
    uint end_i;
    uint level;
    char c;
    uint num_child;

};

void saveTrie(LofNode * node, ostream * out) {
    lof_node serial;
    serial.start_i = node->start_i_;
    serial.end_i = node->end_i_;
    serial.level = node->level_;
    serial.c = node->c_;
    serial.num_child = 0;
    LofNode ** children = node->getChildren();
    for (int i = 0; i < SIGMA_SIZE; i++) {
        if (children[i] != NULL) serial.num_child++;
    }
    out->write(reinterpret_cast<char*> (&serial), sizeof (serial));
    for (int i = 0; i < SIGMA_SIZE; i++) {
        if (children[i] != NULL) saveTrie(children[i], out);
    }
}

LofNode * loadNode(LofNode* parent, istream * in) {
    lof_node serial;
    in->read(reinterpret_cast<char *> (&serial), sizeof (serial));
    LofNode * newNode = new LofNode(serial.start_i, serial.level, serial.c, parent);
    if (parent != NULL) parent->addChild(serial.c, newNode);
    for (int i = 0; i < serial.num_child; i++) {
        loadNode(newNode, in);
    }
    return newNode;
}

LofNode * loadTrie(istream * in) {
    return loadNode(NULL, in);
}

LofBuilder::LofBuilder(LoArrayIterator * lo_it, string text_file, string trie_file, string lof_file) {
    text_ = new ifstream(text_file.c_str());
    lo_it_ = lo_it;
    lof_filename_ = lof_file;
    trie_filename_ = trie_file;
}

void LofBuilder::gotoEntry(lo_entry entry, int offset) {
    text_->clear();
    text_->seekg(entry.i + offset, ios::beg);
}

char LofBuilder::readTextChar() {
    char c;
    text_->get(c);
    return c;
}

LofNode * LofBuilder::getTrie() {
    FILE * trie_file = fopen(trie_filename_.c_str(), "r");
    if (trie_file == NULL) {// no trie available, gotta build it
        cout << "Building trie" << endl;
        buildOntoNode(root, entry);
        ofstream * out = new ofstream(trie_filename_.c_str());
        saveTrie(root, out);
        out->close();
        delete out;
        return root;
    } else {
        cout << "On disk trie found" << endl;
        fclose(trie_file);
        ifstream * in = new ifstream(trie_filename_.c_str());
        LofNode * root = loadTrie(in);
        in->close();
        delete in;
        return root;
    }
}

LofSa * LofBuilder::build() {
    LofNode * root = getTrie();
    int c = 0;
    FILE * lof_file = fopen(lof_filename_.c_str(), "r");
    if (lof_file == NULL) {
        cout << "Building index" << endl;
        ofstream * lof = new ofstream(lof_filename_.c_str());
        lo_it_->reset();
        createLof(root, lof, &c);
        lof->close();
        delete lof;
    } else {
        cout << "Index found" << endl;
        fclose(lof_file);
    }

    return new LofSa(root, lof_filename_);
}

void LofBuilder::readCharsToFringe(char * fringe) {
    for (int i = 0; i < F; i++) {
        char c = readTextChar();
        if (!text_->eof()) {
            fringe[i] = c;
        } else {
            fringe[i] = (char) 0;
        }
    }
}

void LofBuilder::createLof(LofNode * node, ofstream * out, int * i) {
    if (node->isLeaf()) {
        if (*i != node->start_i_) {
            cout << "BAD!!" << *i << "," << node->start_i_ << endl;
            //            exit(EXIT_FAILURE);
        }
        for (int j = *i; j < node->end_i_; j++) {
            lo_entry e = lo_it_->getNext();
            lof l;
            l.i = e.i;
            l.lcp = e.lcp;
            gotoEntry(e, e.lcp);
            readCharsToFringe(l.fr);
            out->write(reinterpret_cast<char*> (&l), sizeof (l));
        }
        *i = node->end_i_;
    } else {
        LofNode ** children = node->getChildren();
        for (int j = 0; j < SIGMA_SIZE; j++) {
            if (children[j] != NULL) createLof(children[j], out, i);
        }
    }
}

LofNode * LofBuilder::buildOntoNode() {
    LofNode * root = new LofNode(0, 0, 0, NULL);
    lo_entry entry = lo_it_->getNext();
    gotoEntry(entry);
    int counter = 0;
    LofNode * node = root;
    while (lo_it_->hasNext()) {
        //        cout << "Node: " << node->level_ << "," << node->c_ << endl; //<< ".  Child: " << child->level_ << "," << child->c_ << endl;
        //        cout << "Entry: " << entry.i << "," << entry.lcp << endl;
        if (entry.lcp < node->level_) { // the parent must handle entry, child is finished
            node->end_i_ = counter;
            if (node->end_i_ - node->start_i_ <= B) { //we don't need the children
                node->pruneChildren();
            }
            //            cout << "Moving to parent" << endl;
            node = node->getParent();
            continue;
        }
        char c = readTextChar();
        if (text_->eof()) {
            c = '$';
            text_->clear();
        }
        if (node->getChild(c) == NULL) { //create the child if it doesn't exist
            LofNode * child = new LofNode(counter, node->level_ + 1, c, node);
            node->addChild(c, child);
        }

        if (entry.lcp > node->level_) //add child, then we're done
        { //now let the child handle this entry
            //                cout << "Moving to child" << endl;
            node = node->getChild(c);
            continue;
        }
        counter++;
        entry = lo_it_->getNext();
        gotoEntry(entry);
    }
    //handle last step to close off the remaining intervals
    //    counter++;
    while (node != NULL) {
        node->end_i_ = counter;
        if (node->end_i_ - node->start_i_ <= B) {
            node->pruneChildren();
        }
        node = node->getParent();
    }
    return root;
}
//LofNode * LofBuilder::buildOntoNode() {
//    LofNode * root = new LofNode(0, 0, 0, NULL);
////    root.end_i_ = 0;
//    uint counter = 0;
//    while (lo_it_->hasNext()) {
//        lo_entry = lo_it_->getNext();
//        //if the entry lcp is the same as the level, we children until
//        //if the entry lcp is
//    }
//}

LofBuilder::~LofBuilder() {
    text_->close();
    delete text_;
}

