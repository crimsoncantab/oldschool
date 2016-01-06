#include <stdio.h>
#include <cs50.h>
//initializes functions
void PrintSmallSquare(char a, int size);
void PrintSquare(char a, char b, int size);
int main(int argc, char * argv[])
{
    int size;
    char a, b;
    //Asks for the size of square and checks input
    do
    {
    printf("Please type a number between 1 and 23\n");
    size = GetInt();
    } while (size < 1 || size > 23);
    //Asks user for symbol
    printf("Please type in a symbol:\n");
    a = GetChar();
    //Makes a simple square with one kind of symbol if size is small enough
    if (size < 3)
        PrintSmallSquare(a, size);
    else
    {
        /*Asks user for another symbol for larger squares, ensures 
        new symbol is different, then creates square */
        
        do
        {
            printf("Please type in a different symbol:\n");
            b = GetChar();
        } while (b == a);
        PrintSquare(a, b, size);
    }
    return 0;
}
//Prints sqares with only one character (smaller than 3 in size)
void PrintSmallSquare(char a, int size)
{
    //iterates down the square, providing breaks
    for(int i = 0; i < size; i++)
    {
        //iterates accross the square, printing the symbol
        for(int j = 0; j < size; j++)
            printf("%c ", a);
        printf("\n");
    }
}
//Prints large square with 2 characters
void PrintSquare(char a, char b, int size)
{
    //initialize array that will hold one line of characters at a time
    char line[size];
    //iterates through array from both ends of line[]
    for(int j = 0, k = size; j < size; j++, k--)
    {
        //prints spaces to center square in an 80 width window
        for(int i = 0; i < (80 - size * 2) / 2; i++)
            printf(" ");
        //checks when iteration reaches middle of square
        if (j > k)
        {
        /*switches middle characters in line[] for
        bottom half of square*/
        for(int i = k; i < j; i++)
            {
                if(line[i] == a)
                    line[i] = b;
                else
                    line[i] = a;
            }
        }
        /*switches middle characters in line[] for 
        top half of square */
        else
        {
        for(int i = j; i < k; i++)
            {
                if(line[i] == a)
                    line[i] = b;
                else
                    line[i] = a;
            }
        }
        //iterates along array and prints every character, then returns
        for (int i = 0; i < size; i++)
            printf("%c ", line[i]);
        printf("\n");

    }
}
