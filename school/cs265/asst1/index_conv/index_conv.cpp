/* 
 * File:   main.cpp
 * Author: loren
 *
 * Created on September 18, 2010, 11:02 PM
 */

#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include "LoArray.h"
#include "LofBuilder.h"
#include "LofNode.h"

using namespace std;

//string load_from_file(const char * filename) {
//    string buf;
//    string line;
//    ifstream in(filename);
//    while (std::getline(in, line))
//        buf += line;
//    //    cout << "read: " << buf << "\n";
//    return buf;
//}

//uint calc_lcp(ifstream & text, uint i1, uint i2) {

uint calc_lcp(ifstream & text1, ifstream & text2, uint i1, uint i2) {
    int lcp = 0;
    text1.seekg(i1, ios::beg);
    text2.seekg(i2, ios::beg);
    char a, b;
    text1.get(a);
    text2.get(b);
    //    cout << "a" << a << endl;
    //    cout << "b" << b << endl;
    while (text1.good() && text2.good() && a == b) {
        lcp++;
        //        i1++;
        //        i2++;
        text1.get(a);
        text2.get(b);
        //        cout << "a" << a << endl;
        //        cout << "b" << b << endl;

    }
    text1.clear();
    text2.clear();
    return lcp;
}

int INT_little_endian_TO_big_endian(int i) {
    return ((i & 0xff) << 24)+((i & 0xff00) << 8)+((i & 0xff0000) >> 8)+((i >> 24)&0xff);
}

void make_lo_array(string sa_file, string text_file, string lo_file) {
    ifstream * text = new ifstream(text_file.c_str());
    ifstream * text2 = new ifstream(text_file.c_str());

    ifstream in(sa_file.c_str());
    uint count = 0;
    uint temp;
    while (!in.eof()) {
        in.read(reinterpret_cast<char *> (&temp), sizeof (temp));
        temp++;
    }
    in.clear();
    in.seekg(0, ios::beg);

    ofstream out(lo_file.c_str());
    lo_entry entry;
    entry.i = count;
    entry.lcp = 0;
    out.write(reinterpret_cast<char*> (&entry), sizeof (entry));

    uint prev;
    bool has_prev = false;

    int m;
    in.read(reinterpret_cast<char *> (&m), sizeof (m));
    entry.i = INT_little_endian_TO_big_endian(m);
    while (!in.eof()) {
        //        cout << "read: " << m << endl;
        entry.lcp = (has_prev) ? calc_lcp(*text, *text2, prev, entry.i) : 0;
        //        cout << "lcp: " << entry.lcp << endl;
        out.write(reinterpret_cast<char*> (&entry), sizeof (entry));
        prev = entry.i;
        has_prev = true;
        in.read(reinterpret_cast<char *> (&m), sizeof (m));
        entry.i = INT_little_endian_TO_big_endian(m);
    }
    out.close();
    in.close();
    text->close();
    text2->close();
    delete text;
    delete text2;
}

int main(int argc, char** argv) {
    string text_filename(argv[1]);
    string sa_filename = text_filename + ".ary";
    string lo_filename = text_filename + ".lo";
    string lof_filename = text_filename + ".lof";
    string trie_filename = text_filename + ".trie";
    make_lo_array(sa_filename, text_filename, lo_filename);

    LofSa * lofSa;
    LoArrayIterator * lo_it = new LoArrayIterator(lo_filename);
    LofBuilder * builder = new LofBuilder(lo_it, text_filename, trie_filename, lof_filename);
    lofSa = builder->build();
    delete builder;
    delete lo_it;
    delete lofSa;
    return 0;
}

