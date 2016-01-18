// main.cpp v2.0

#include<string>
#include<iostream>
#include<cassert>
#include "DictTree.h"
using namespace std;

// This simply initializes a dictionary and runs some testing suites.
int main() {
    cout << "Running DictTree tests..." << endl;
    
    //
    // Test DictTree.extend() and DictTree.lookup()
    //

    // Lookup on an empty dictionary
    DictTree dt;
    assert(dt.lookup("hello") == "");

    // Lookup on a one element dictionary
    dt.extend("hello","world");
    assert(dt.lookup("hello") == "world");

    // Lookup on a multi element dictionary
    dt.extend("goodbye","earth");
    dt.extend("foo","bar");
    assert(dt.lookup("hello") == "world");
    assert(dt.lookup("goodbye") == "earth");
    assert(dt.lookup("foo") == "bar");
    
    // Lookup something that isn't there
    assert(dt.lookup("bar") == "");

    // Replace the root key
    dt.extend("hello", "notworld");
    assert(dt.lookup("hello") == "notworld");

    // Replace a key at a leaf in the tree
    dt.extend("foo","baz");
    assert(dt.lookup("foo") == "baz");
    
    //
    // Test DictTree.keys()
    //

    // Keys on an empty dictionary
    DictTree dt_keys;
    assert(dt_keys.keys() == "");

    // Keys on a one element dictionary
    dt_keys.extend("orange","a nice color");
    assert(dt_keys.keys() == "orange*"); 
    
    // Keys on a multi-element dictionary
    // They should print in alphabetical order
    dt_keys.extend("cs51","a nice class");
    dt_keys.extend("thomas","a nice person");
    assert(dt_keys.keys() == "cs51*orange*thomas*");

    // If we've made it this far without failing an assert, then we must
    // have passed.
    cout << "All tests passed!" << endl;
}
