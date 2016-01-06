#include <stdio.h>

typedef char BYTE;

// program name, original file, output file
int main(int argc, char *argv[])
{
    int nOfSegments = 0;
    int start = 0;
    int end = 0;
    BYTE temp;
    FILE *inptr = fopen(argv[1], "r");
    FILE *outptr = fopen(argv[2], "wb");

    printf("How many segments?\n");
    scanf("%i", &nOfSegments);

    for (int i = 0; i < nOfSegments; i++)
    {
        printf("Start (inclusive) of segment %i? ", i + 1);
        scanf("%i", &start);
    
        printf("End (exclusive) of segment %i? ", i + 1);
        scanf("%i", &end);
        
        fseek(inptr, start, SEEK_SET);
        for (int j = start; j < end; j++)
        {
            fread(&temp, sizeof(BYTE), 1, inptr);
            fwrite(&temp, sizeof(BYTE), 1, outptr);
        }
    }

    return 0;
}
