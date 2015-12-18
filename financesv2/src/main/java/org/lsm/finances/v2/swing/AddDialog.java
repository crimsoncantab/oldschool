/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.lsm.finances.v2.swing;

import java.awt.Color;
import java.awt.Component;
import java.awt.Frame;
import java.util.Map;
import java.util.Map.Entry;
import javax.swing.JDialog;
import org.lsm.finances.v2.object.FinanceProfile;
import org.lsm.finances.v2.object.ProfileChangeListener;

/**
 *
 * @author Loren
 */
public abstract class AddDialog<T> extends JDialog implements ProfileChangeListener {

    protected FinanceProfile profile;

    public AddDialog(Frame owner, FinanceProfile profile) {
        super(owner, true);
        this.profile = profile;
        profile.addListener(this);
    }

    protected void highlightErrors(Map<Component, Boolean> invalidFields) {

        for (Entry<Component, Boolean> field : invalidFields.entrySet()) {
            if (field.getValue()) {
                field.getKey().setForeground(Color.red);
            } else {
                field.getKey().setForeground(Color.black);
            }
        }
    }

    protected void close() {
        setVisible(false);
    }

    @Override
    public void notifyAccountChange() {
    }

    @Override
    public void notifyTransactionChange() {
    }

    @Override
    public void notifyCategoryChange() {
    }

    @Override
    public void notifyChange() {
    }

    @Override
    public void reload(FinanceProfile profile) {
        this.profile = profile;
    }

    protected abstract T getNew();
}
