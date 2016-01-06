#include <cs50.h>
#include <stdio.h>

int main(int argc, char * argv[])
{
    //Initialization
    int height;
    int i, j, k;
    //Asks user for input until valid value is given
    do
    {
        printf("Please type an integer between 1 and 23:\n");
        height = GetInt();
    } while (height < 1 || height > 23);
    //iterates through 23 lines, providing break at end of each one
    for (i = 23; i > 0; i--)
    {
        //checks for lines that will contain pyramid
        if (i <= height)
        {
            //outputs spaces until pyramid is reached
            for (j = 80; j > height - i + 2; j--)
            {
                printf(" ");
            }
            //finishes 80-length line with * to represent blocks of pyramid
            for (k = j; k > 0; k--)
            {
                printf("*");
            }
        }
        printf("\n");
    }
    return 0;
}
