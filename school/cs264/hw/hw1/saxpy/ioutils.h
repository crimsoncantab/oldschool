#ifndef SAXPY_UTILS_H
#define SAXPY_UTILS_H


/* Reads in a space-delimited text file of floating point numbers */
int ReadFile(const char *fname, float **data, int *n);

/* Writes a space-delimited text file of floating point numbers */
int WriteFile(const char *fname, float *data, int n);


#endif
