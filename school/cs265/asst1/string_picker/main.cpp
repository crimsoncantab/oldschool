/* 
 * File:   main.cpp
 * Author: loren
 *
 * Created on September 19, 2010, 12:47 PM
 */

#include <stdlib.h>
#include <fstream>
#include <string>
#include <ctime>
#include <iostream>
extern "C" {
    #include "mt19937-64.h"
}

using namespace std;

/*
 * 
 */
int main(int argc, char** argv) {
    string text(argv[1]);
    uint text_len = atoi(argv[2]);
    uint string_len = atoi(argv[3]);
    uint num_strings = atoi(argv[4]);

    ifstream in(text.c_str());

    init_genrand64(time(NULL));
    for (int i = 0; i < num_strings; i++) {

        in.seekg((int)(genrand64_int64() % (text_len - string_len)), ios::beg);
        for (int j = 0; j < string_len; j++) {
            char c;
            in >> c;
            cout << c;
        }
        cout << endl;

    }
    return (EXIT_SUCCESS);
}

