// DictList.cpp v2.0

#include<string>
#include<iostream>
#include<cassert>
#include "DictList.h"

using namespace std;

//
// Implementation for LLNode
//

LLNode::LLNode(string f, string s) {
    first = f;
    second = s;
    next = NULL;
}

LLNode::~LLNode() {
    // delete does not break if it is passed NULL
    delete next;
}

//
// Implementation for DictList
//

DictList::DictList() {
    head = NULL;
}

DictList::~DictList() {
    // delete does not break if it is passed NULL
    delete head;
}
    
void DictList::extend(string key, string value) { 
    // If we find the key in the list, replace its value with the new one
    // and return
    LLNode *prev, *trav;
    prev = trav = NULL;
    for(trav = head; trav != NULL; trav = trav->next) {
        if(trav->first == key) {
            trav->second = value;
            return;
        }
        prev = trav;
    }
    
    if(prev == NULL) {
        // List was empty
        head = new LLNode(key, value);
    } else {
        // Key wasn't found in the list
        prev->next = new LLNode(key, value);
    }
}

string DictList::lookup(string key) {
    // Try to find the key
    for(LLNode *trav = head; trav != NULL; trav = trav->next) {
        if(trav->first == key) {
            return trav->second;
        }
    }

    // If we didn't find anything, return "" to signify this
    return "";
}

string DictList::keys() {
    string ret;
    LLNode *trav;
    int i;
    for(trav = head; trav != NULL; trav = trav->next) {
        ret += trav->first + "*";
    }
    return ret;
}

