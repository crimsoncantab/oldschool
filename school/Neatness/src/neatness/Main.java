/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package neatness;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 *
 * @author Loren
 */
public class Main {

    private static class Pointer {
        int i;
        int j;

        public Pointer(int i, int j) {
            this.i = i;
            this.j = j;
        }
    }

    public static void main(String[] args) throws FileNotFoundException, IOException {
        int M = 72;
        File f = new File("firefly");
        String[] words = new BufferedReader(new InputStreamReader(new FileInputStream(f))).readLine().split(" ");
        //initialize the sum of lengths to make lookup of f() constant
        int[][] L = new int[words.length][words.length];
        for (int i = 0; i < words.length; i++) {
            for (int j = i; j < words.length; j++) {
                if (i==j) L[i][j]= words[i].length();
                else L[i][j] = L[i][j-1] + words[j].length();
            }
        }
        //N finds the minimum penalty over all (i,w)
        int[][] N = new int[words.length][words.length];
        //If a line goes past M, we want to mark it to be ingnored
        boolean [][] overflows = new boolean[words.length][words.length];
        //breadcrumbs
        Pointer [][] prev = new Pointer[words.length][words.length];

        //init T(1,w) for all w (the penalty over the first line)
        for (int i = 0; i < words.length; i++) {
            int extraSpace = f(L, 0, i, M);
            if (extraSpace < 0) overflows[0][i] = true;
            else {
                overflows[0][i] = false;
                N[0][i] = extraSpace * extraSpace * extraSpace;
            }
        }

        //using dyn. programming, find
        for (int i = 1; i < words.length; i++) {
            for (int w = 0; w < words.length; w++) {
                N[i][w] = -1;
                for (int l = 1; l <= w; l++) {
                    //ignore l if we have overflow in prev lines
                    if (overflows[i-1][l-1]) continue;
                    int cumulative = N[i-1][l-1]; //is positive
                    int extraSpace = f(L, l, w, M);
                    if (extraSpace < 0) continue; //too long, bail
                    int penalty;
                    //don't add extra space penalty for last line
                    if (w == words.length-1) {
                        penalty = cumulative;
                    } else penalty = cumulative + 
                            extraSpace * extraSpace * extraSpace;
                    //find the minimum penalty, update pointer
                    if (N[i][w] == -1 || N[i][w] > penalty) {
                        N[i][w] = penalty;
                        prev[i][w] = new Pointer(i-1, l-1);
                    }
                }
                //we couldn't find any legal line split
                if (N[i][w] == -1) overflows[i][w]=true;
            }
        }
        //find the line  ilength with the smallest N(i, n)
        Pointer optimum = null;
        for (int i = 0; i < words.length; i++) {
             if (!overflows[i][words.length-1]) {
                 if (optimum == null || N[i][words.length-1] < N[optimum.i][optimum.j]){
                     optimum = new Pointer(i, words.length-1);
                 }
             }
        }
        System.out.println("Minimum penalty: " + N[optimum.i][optimum.j]);
        //follow pointers to find all the splits
        int[] splits = new int[optimum.i + 1];
        int numLines = optimum.i+1;
        for (int i = 0; i < numLines; i++) {
            splits[numLines -i -1] = optimum.j;
            optimum = prev[optimum.i][optimum.j];
        }
        //print out the text, with splits in correct places
        int line = 0;
        for (int i = 0; i < words.length; i++) {
            System.out.print(words[i] + " ");
            if (i == splits[line]) {
                System.out.println();
                line++;
            }
        }
        System.out.println();
    }
    //extra space
    public static int f(int[][] L, int i, int j, int M) {
        return M-j+i-L[i][j];
    }

}
