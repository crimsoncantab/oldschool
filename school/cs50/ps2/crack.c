#include <stdio.h>
#include <cs50.h>
#include <string.h>
#define _XOPEN_SOURCE
#include <unistd.h>
char salt[2];
char code[13];
void crack(int,int);
int main(int argc, char * argv[])
{
    //checks for correct number of command line arguments
    if (argc != 2)
    {
        printf("Please use only one command-argument\n");
        return 1;
    }
    strcpy(code, argv[1]); //moves the encryption to code
    strcpy(salt, ""); //initializes salt as empty
    strncpy(salt, code, 2); //extracts salt from encryption
    crack(97, 122); //cracks password
    return 0;
}

void crack(int lo, int hi)
{
    char password[8] = ""; //initializes an empty test password
    hi++;
    /*the following eight loops change each of the eight
    characters of the password systematically */
    int a, b, c, d, e, f, g, h;
    for (h = lo; h <= hi; h++)
    {
        for (g = lo; g <= hi; g++)
        {
            for (f = lo; f <= hi; f++)
            {
                for (e = lo; e <= hi; e++)
                {
                    for (d = lo; d <= hi; d++)
                    {
                        for (c = lo; c <= hi; c++)
                        {
                            for (b = lo; b <= hi; b++)
                            {
                                for (a = lo; a <= hi; a++)
                                {
                                    password[0] = (char) a;
        //checks if encryption of password equals command line value
                                if (strcmp((string) crypt(password,salt), code) 
                                == 0)
                                        {
                                            //prints password when found
                                            printf("%s",password);
                                            return;
                                        }
                                }
                                password[1] = (char) b;
                            }
                            password[2] = (char) c;
                        }
                        password[3] = (char) d;
                    }
                    password[4] = (char) e;
                }
                password[5] = (char) f;
            }
            password[6] = (char) g;
        }
        password[7] = (char) h;
    }
    return;
}
