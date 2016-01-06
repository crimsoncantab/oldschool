/****************************************************************************
 * whodunit.c
 *
 * Computer Science 50
 * Problem Set 4
 *
 ***************************************************************************/
       
#include <cs50.h>
#include <stdio.h>
#include <stdlib.h>


#include "whodunit.h"


int
main(int argc, char * argv[])
{
    // ensure proper usage
    if (argc != 3)
    {
        printf("Usage: %s infile outfile\n", argv[0]);
        return 1;
    }

    // remember filenames
    char * infile = argv[1];
    char * outfile = argv[2];

    // try to open input file 
    FILE * inptr = fopen(infile, "r");
    if (inptr == NULL)
    {
        printf("Could not open %s.\n", infile);
        return 1;
    }

    // try to open output file
    FILE * outptr = fopen(outfile, "wb");
    if (outptr == NULL)
    {
        printf("Could not open %s for writing.\n", outfile);
        return 2;
    }

    // read in file's BITMAPFILEHEADER
    BITMAPFILEHEADER header;
    fread(&header, sizeof(BITMAPFILEHEADER), 1, inptr);

    // write out file's BITMAPFILEHEADER
    fwrite(&header, sizeof(BITMAPFILEHEADER), 1, outptr);

    // read in file's BITMAPINFOHEADER
    BITMAPINFOHEADER info;
    fread(&info, sizeof(BITMAPINFOHEADER), 1, inptr);

    // write out file's BITMAPINFOHEADER
    fwrite(&info, sizeof(BITMAPINFOHEADER), 1, outptr);

    // temporary storage
    BYTE byte;

    // iterate over bitmap's bytes
    while (!feof(inptr))
    {
        if (fread(&byte, sizeof(BYTE), 1, inptr)) {
        	//changes all occurences of ff in file to 00
        	//This will make any red or white pixels black
        	if (byte == 0xff)
        		byte = 0x00;
        	fwrite(&byte, sizeof(BYTE), 1, outptr);
        }
    }

    // close files
    fclose(inptr);
    fclose(outptr);

    // that's all folks
    return 0;
}

