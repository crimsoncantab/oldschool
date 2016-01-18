// main.cpp v2.0

#include<string>
#include<iostream>
#include<cassert>
#include "DictList.h"
using namespace std;

// This simply initializes a dictionary and runs some testing suites.
int main() {
    cout << "Running DictList tests..." << endl;
    
    //
    // Test DictList.extend() and DictList.lookup()
    //

    // Lookup on an empty dictionary
    DictList dl;
    assert(dl.lookup("hello") == "");

    // Lookup on a one element dictionary
    dl.extend("hello","world");
    assert(dl.lookup("hello") == "world");

    // Lookup on a multi element dictionary
    dl.extend("goodbye","earth");
    dl.extend("foo","bar");
    assert(dl.lookup("hello") == "world");
    assert(dl.lookup("goodbye") == "earth");
    assert(dl.lookup("foo") == "bar");
    
    // Lookup something that isn't there
    assert(dl.lookup("bar") == "");

    // Replace a key in the middle of the list
    dl.extend("hello", "notworld");
    assert(dl.lookup("hello") == "notworld");

    // Replace a key at the end of the list
    dl.extend("foo","baz");
    assert(dl.lookup("foo") == "baz");
    
    //
    // Test DictList.keys()
    //

    // Keys on an empty dictionary
    DictList dl_keys;
    assert(dl_keys.keys() == "");

    // Keys on a one element dictionary
    dl_keys.extend("orange","a nice color");
    assert(dl_keys.keys() == "orange*"); 
    
    // Keys on a multi-element dictionary
    // They should print in the order they were added
    dl_keys.extend("cs51","a nice class");
    dl_keys.extend("thomas","a nice person");
    assert(dl_keys.keys() == "orange*cs51*thomas*");

    // If we've made it this far without failing an assert, then we must
    // have passed.
    cout << "All tests passed!" << endl;
}
