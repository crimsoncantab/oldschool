#ifndef SAXPY_UTILS_H
#define SAXPY_UTILS_H

#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <vector>
#include <iostream>

/* Reads in a space-delimited text file of floating point numbers */
int ReadFile(const char *fname, float **data, int *n){
    std::vector<float> datain;
    std::fstream fh(fname,std::fstream::in);
    if(!fh.good())
        return 0;
    float val;
    while(fh.good()){
        fh >> val;
        datain.push_back(val);
    }
    datain.pop_back();
    fh.close();

    *data=(float*)malloc(sizeof(float)*datain.size());
    *n=(int)datain.size();
    memcpy(*data,&datain.front(),sizeof(float)*datain.size());
    return 1;
}

/* Writes a space-delimited text file of floating point numbers */
int WriteFile(const char *fname, float *data, int n){
    std::fstream fh(fname,std::fstream::out);
    if(!fh.good())
        return 0;
    for(int i=0; i<n; i++)
        fh << data[i] << ' ';
    if(!fh.good())
        return 0;
    fh << std::endl;
    fh.close();  
    return 1;
}

#endif
