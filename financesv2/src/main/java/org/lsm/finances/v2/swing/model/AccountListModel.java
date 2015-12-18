/* @author Loren
 *
 * Created on: Oct 6, 2008, at 7:11:58 PM
 *
 *
 */
package org.lsm.finances.v2.swing.model;

import java.util.List;
import javax.swing.AbstractListModel;
import javax.swing.MutableComboBoxModel;
import org.lsm.finances.v2.object.Account;
import org.lsm.finances.v2.object.FinanceProfile;
import org.lsm.finances.v2.object.ProfileChangeListener;

public class AccountListModel extends AbstractListModel implements MutableComboBoxModel, ProfileChangeListener {

    private List<Account> data;
    private Account selected;

    public AccountListModel(FinanceProfile profile) {
        this.data = profile.getAccounts();
        profile.addListener(this);
    }

    @Override
    public int getSize() {
        return data.size();
    }

    @Override
    public Object getElementAt(int index) {
        return data.get(index);
    }

    @Override
    public void addElement(Object obj) {
        data.add((Account) obj);
        fireIntervalAdded(this, data.indexOf(obj), data.indexOf(obj));
    }

    @Override
    public void removeElement(Object obj) {
        data.remove(obj);
        fireIntervalRemoved(this, data.indexOf(obj), data.indexOf(obj));
    }

    @Override
    public void insertElementAt(Object obj, int index) {
        data.add(index, (Account) obj);
        fireIntervalAdded(this, index, index);
    }

    @Override
    public void removeElementAt(int index) {
        data.remove(index);
        fireIntervalRemoved(this, index, index);
    }

    @Override
    public void setSelectedItem(Object anItem) {
        selected = (Account) anItem;
    }

    @Override
    public Object getSelectedItem() {
        return selected;
    }

    @Override
    public void notifyAccountChange() {
        fireContentsChanged(this, 0, getSize());
    }

    @Override
    public void notifyTransactionChange() {
        //does nothing
    }

    @Override
    public void notifyCategoryChange() {
        //does nothing
    }

    @Override
    public void notifyChange() {
        notifyAccountChange();
    }

    @Override
    public void reload(FinanceProfile profile) {
        data = profile.getAccounts();
        notifyAccountChange();
    }
}