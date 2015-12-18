package org.lsm.finances.v2;

import java.awt.EventQueue;

/**
 * Hello world!
 *
 */
public class App {

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        EventQueue.invokeLater(new Runnable() {

            @Override
            public void run() {
                new MainFrame().setVisible(true);
            }
        });
    }
}
