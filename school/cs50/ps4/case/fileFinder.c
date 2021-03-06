/****************************************************************************
 * recover.c
 * Loren
 * Chris
 * Computer Science 50
 * Problem Set 4
 *
 ***************************************************************************/
       
#include <cs50.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SECTIONSIZE 512

typedef char BYTE;
typedef short TWOBYTES;

short int gterminating = 0;

int checkForZipEnd(FILE *inptr);
int checkForZipTag(FILE *inptr);
int countSections(FILE *inptr);
int toNextSection(FILE *inptr);
//void createZip(FILE *, int, TWOBYTES);
void createStartOfZip(FILE *inptr);
void createZip(FILE *inptr);


int main(int argc, char * argv[])
{
    int  nOfSections = 0;
    int nOfZipHeaders = 0;
    int section = 0;

    // ensure proper usage
    if (argc != 2)
    {
        printf("Usage: %s infile\n", argv[0]);
        return 1;
    }

    // remember filename
    char *infile = argv[1];

    // try to open input file 
    FILE *inptr = fopen(infile, "r");
    if (inptr == NULL)
    {
        printf("Could not open %s.\n", infile);
        return 1;
    }
       
    nOfSections = countSections(inptr);
    char taglist[nOfSections];
    
    printf("%d sections\n", nOfSections);

    /*
     * search beginning of each section for a tag
     * copy that section to a file if a tag is found.
     */

    while (section <= nOfSections && !gterminating)
    {
        if (checkForZipTag(inptr)) 
        {
            nOfZipHeaders++;
            taglist[section] = 'z';
            //createZip(inptr);
            createStartOfZip(inptr);
        }
        toNextSection(inptr);
        section++;
    }
    
    int zipSections[nOfZipHeaders];
    int zIndex = 0;
    
    for (int i = 0; i < nOfSections; i++)
    {
        char tag = taglist[i];
        
        switch (tag)
        {
            case 'z':
                zipSections[zIndex] = i;
                zIndex++;
                break;
        }
    }

    printf("Found zip tags at tops of sections ");
    for (int i = 0; i < nOfZipHeaders; i++)
        printf("%d, ", zipSections[i]);
   
    printf("\n");

    // close file
    fclose(inptr);
    
    return 0;
}

// copies a section that begins with a zip tag to its own file
void createStartOfZip(FILE *inptr)
{    
    BYTE temp;
	static char outFile[] = "zipStart00.zip";
   
   FILE *outptr = fopen(outFile, "wb");
	for (int i = 0; i < SECTIONSIZE / sizeof(BYTE); i++)
	{
		fread(&temp, sizeof(BYTE), 1, inptr);
		fwrite(&temp, sizeof(BYTE), 1, outptr);
	}
	
    fclose(outptr);
	fseek(inptr, -SECTIONSIZE, SEEK_CUR);

	if (outFile[9] == '9')
	{
		outFile[8] = outFile[8] + 1;
		outFile[9] = '0';
	}
	else
		outFile[9] = outFile[9] + 1;
}
/*
void createZip(FILE *inptr)
{    
    BYTE temp;
    int foundLastByte = 0;
	static char outFile[] = "zipStart00.zip";
   
   FILE *outptr = fopen(outFile, "wb");
	while (!foundLastByte)
	{
	    foundLastByte = checkForZipEnd(inptr);
		fread(&temp, sizeof(BYTE), 1, inptr);
        fwrite(&temp, sizeof(BYTE), 1, outptr);
    }
		fread(&temp, sizeof(BYTE), 1, inptr);
        fwrite(&temp, sizeof(BYTE), 1, outptr);
	
    fclose(outptr);
	fseek(inptr, -SECTIONSIZE, SEEK_CUR);

	if (outFile[9] == '9')
	{
		outFile[8] = outFile[8] + 1;
		outFile[9] = '0';
	}
	else
		outFile[9] = outFile[9] + 1;
}
*/
/*
void createZip2(FILE *inptr)
{    
    BYTE temp;
    int foundLastSector = 0;
	static char outFile[] = "zipStart00.zip";
   
    FILE *outptr = fopen(outFile, "wb");
	while (!foundLastSector && )
	{
	    foundLastSector = checkForZipEnd(inptr);
		fread(&temp, sizeof(BYTE), 1, inptr);
        fwrite(&temp, sizeof(BYTE), 1, outptr);
    }
		fread(&temp, sizeof(BYTE), 1, inptr);
        fwrite(&temp, sizeof(BYTE), 1, outptr);
	
    fclose(outptr);
	fseek(inptr, -SECTIONSIZE, SEEK_CUR);

	if (outFile[9] == '9')
	{
		outFile[8] = outFile[8] + 1;
		outFile[9] = '0';
	}
	else
		outFile[9] = outFile[9] + 1;
}
*/

/* void createZip(FILE * inptr, int sector, TWOBYTES sentinel) {
    BYTE temp;
    int atEnd = 0;
    int counter = 0;
    fseek(inptr, 512 * sector, SEEK_SET);
    FILE * outptr = fopen("test.zip", "wb");
    do {
        if (fread(&temp, sizeof(BYTE), 1, inptr)) {
            fwrite(&temp, sizeof(BYTE), 1, outptr);
        }
        counter++;
        if (temp == sentinel)
            atEnd++;
    } while (!(atEnd == 2 && (counter-1) % (512 / sizeof(TWOBYTES)) == 0));
}
*/

// tells whether or not the current section begins with a zip tag
int checkForZipTag(FILE *inptr)
{
    BYTE temp[2];
    if (fread(temp, sizeof(BYTE), 2, inptr)) 
    {
        fseek(inptr, -sizeof(BYTE) * 2, SEEK_CUR);
        if (temp[0] == 'P' && temp[1] == 'K')
            return 1;
        else
            return 0;
    }
    else 
    {
        gterminating = 1;
        return 0;
    }
}


int checkForZipEnd(FILE *inptr)
{
    BYTE temp[2];
    if (fread(temp, sizeof(BYTE), 2, inptr)) 
    {
        fseek(inptr, -sizeof(BYTE) * 2, SEEK_CUR);
        if (temp[0] == 0x05 && temp[1] == 0x06)
            return 1;
        else
            return 0;
    }
    else 
    {
        gterminating = 1;
        return 0;
    }
}
//moves to the next section of the drive
int toNextSection(FILE *inptr)
{
   return fseek(inptr, SECTIONSIZE, SEEK_CUR);
}

// counts how many sections there are in the file
int countSections(FILE *inptr)
{
    
    int n = 0;
    BYTE temp;
    fread(&temp, sizeof(BYTE), 1, inptr);
    
    while (!feof(inptr))
    {
         fseek(inptr, -sizeof(BYTE), SEEK_CUR);
         toNextSection(inptr);      
         fread(&temp, sizeof(BYTE), 1, inptr);
         n++;
    }

    rewind(inptr);
    return n;
}
