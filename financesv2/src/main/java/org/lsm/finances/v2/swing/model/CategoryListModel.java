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
import org.lsm.finances.v2.object.Category;
import org.lsm.finances.v2.object.FinanceProfile;
import org.lsm.finances.v2.object.ProfileChangeListener;

public class CategoryListModel extends AbstractListModel implements MutableComboBoxModel, ProfileChangeListener {

    private List<Category> data;
    private Category selected;

    public CategoryListModel(FinanceProfile profile) {
        this.data = profile.getCategories();
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
        data.add((Category) obj);
        fireIntervalAdded(this, data.indexOf(obj), data.indexOf(obj));
    }

    @Override
    public void removeElement(Object obj) {
        data.remove(obj);
        fireIntervalRemoved(this, data.indexOf(obj), data.indexOf(obj));
    }

    @Override
    public void insertElementAt(Object obj, int index) {
        data.add(index, (Category) obj);
        fireIntervalAdded(this, index, index);
    }

    @Override
    public void removeElementAt(int index) {
        data.remove(index);
        fireIntervalRemoved(this, index, index);
    }

    @Override
    public void setSelectedItem(Object anItem) {
        selected = (Category) anItem;
    }

    @Override
    public Object getSelectedItem() {
        return selected;
    }

    @Override
    public void notifyAccountChange() {
        //does nothing
    }

    @Override
    public void notifyTransactionChange() {
        //does nothing
    }

    @Override
    public void notifyCategoryChange() {
        fireContentsChanged(this, 0, getSize());
    }

    @Override
    public void notifyChange() {
        notifyCategoryChange();
    }

    @Override
    public void reload(FinanceProfile profile) {
        data = profile.getCategories();
        notifyCategoryChange();
    }
}
