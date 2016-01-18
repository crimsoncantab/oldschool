/* 
 * File:   main.cpp
 * Author: loren
 *
 * Created on September 19, 2010, 6:36 PM
 */

#include <stdlib.h>
#include <iostream>

using namespace std;
/*
 * 
 */
int main(int argc, char** argv) {

    int count = 0;
    while (cin.good()) {
        int n;
        string s;
        cin >> n >> s;
        count += n;
    }

    cout << count << endl;

    return (EXIT_SUCCESS);
}

