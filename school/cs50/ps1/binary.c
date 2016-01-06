#include <cs50.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char * argv[])
{
    int num;
    //asks for a number and checks user input
    do
    {
    printf("Please type a number between 0 and 2147483647\n");
    num = GetInt();
    } while (num < 0 || num > 2147483647);
    //iterates through every binary digit of num
    for(int i = 31; i >= 0; i--)
    {
        /*checks if num has a 0 value for a power of 2
        prints 0 if it does, 1 if it does not*/
        if ( (num & ( 1 << i ) ) == 0)
            printf("0");
        else
            printf("1");
    }
    printf("\n");
    return 0;
}
