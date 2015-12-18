/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.lsm.finances.v2.object;

/**
 *
 * @author Loren
 */
public interface ProfileChangeListener {

    public void notifyAccountChange();

    public void notifyTransactionChange();

    public void notifyCategoryChange();

    public void notifyChange();

    public void reload(FinanceProfile profile);
}
