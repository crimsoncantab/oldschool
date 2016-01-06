#include <stdio.h>
#include <cs50.h>

int main(int argc, char * argv[])
{
    //Initialization
    string name;
    //Asks for user input
    printf("What is your name?\n");
    name = GetString();
    //Says Hello to the user
    printf("Hello %s\n", name);
    return 0;
}
