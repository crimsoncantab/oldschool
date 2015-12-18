/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package birthday;

import java.math.BigInteger;
import java.security.SecureRandom;
import java.util.Random;

/**
 *
 * @author Loren
 */
public class Main {

    static final BigInteger TWO = BigInteger.valueOf(2);

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        BigInteger n = new BigInteger("46947848749720430529628739081");
        BigInteger e = new BigInteger("37267486263679235062064536973");
        BigInteger message = new BigInteger("010001110110100101110110011001010010000001101101011001010010000001100001011011100010000001000001", 2);
        System.out.println(message.toString());
        System.out.println(message.modPow(e, n));
//        BigInteger e_message =
//        BigInteger n3 = BigInteger.valueOf(101);
//        testN(n2);
//        testN(n);
//        testN(n3);

//        BigInteger test = BigInteger.valueOf(202771);
//        BigInteger u = BigInteger.valueOf(36801);
//        System.out.println(test.modPow(BigInteger.valueOf(1).multiply(u), n).toString());
//        System.out.println(test.modPow(BigInteger.valueOf(2).multiply(u), n).toString());
//        System.out.println(test.modPow(BigInteger.valueOf(4).multiply(u), n).toString());
//        test = BigInteger.valueOf(422541);
//        u = BigInteger.valueOf(318063);
//        System.out.println(test.modPow(BigInteger.valueOf(1).multiply(u), n2).toString());

    }

//    private static BigInteger testN(final BigInteger n) {
//        System.out.println("Testing " +n.toString()+ " for primality.");
//        Random rnd = new SecureRandom();
//        int count = 0, num_trials = 10;
//        BigInteger a;
//        boolean isProbablyPrime = true;
//        do {
//            count++;
//            do {
//                a = new BigInteger(n.bitLength(), rnd);
//            } while (a.compareTo(BigInteger.ONE) <= 0 || a.compareTo(n) >= 0);
//            isProbablyPrime = isProbablyPrime && passesMillerRabin(n, a);
//        } while (isProbablyPrime && count < num_trials);
////        System.out.println("a: " + a.toString());
//        System.out.print(n.toString());
//        if (isProbablyPrime) {
//            System.out.println(" is probably prime.");
//        } else {
//            System.out.println(" is composite.");
//        }
//        return a;
//    }
//
//    private static boolean passesMillerRabin(final BigInteger n, final BigInteger a) {
//        System.out.println("Randomly picked a: " + a.toString());
//        // Find a and m such that m is odd and this == 1 + 2**a * m
//        BigInteger nMinusOne = n.subtract(BigInteger.ONE);
//        BigInteger u = nMinusOne;
//        int t = u.getLowestSetBit();
//        u = u.shiftRight(t);
//
//        // Do the tests
//        // Generate a uniform random on (1, this)
//
//        int i = 0;
//        BigInteger z = a.modPow(u, n);
//        System.out.println("u= " + u.toString());
//        System.out.println("t= " + t);
//        while (!((i == 0 && z.equals(BigInteger.ONE)) || z.equals(nMinusOne))) {
//            System.out.println("i= " + i);
//            System.out.println("a^(2^iu) mod " + n.toString() + "= " + z.toString());
//            if (i > 0 && z.equals(BigInteger.ONE) || ++i == t+1) {
////                System.out.println("!= 1");
//                return false;
//            }
////            System.out.println();
//            z = z.modPow(TWO, n);
//        }
//        System.out.println("i= " + i);
//        System.out.println("a^(2^iu) mod " + n.toString() + "= " + z.toString());
//        return true;
//    }
}
