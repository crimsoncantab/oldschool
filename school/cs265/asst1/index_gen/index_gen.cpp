/* 
 * File:   main.cpp
 * Author: loren
 *
 * Created on September 18, 2010, 2:37 PM
 */

#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
extern "C" {
#include "interface.h"
}

using namespace std;

string load_from_file(const char * filename) {
    string buf;
    string line;
    ifstream in(filename);
    while (std::getline(in, line))
        buf += line;
    cout << "read: " << buf << "\n";
    return buf;
}

int main(int argc, char** argv) {

    void * index;
    string data = load_from_file("data");

    build_index((unsigned char *) data.c_str(), data.length(), NULL, &index);
    save_index(index, "data");

    return (EXIT_SUCCESS);
}

