/* @author Loren
 *
 * Created on: Sep 17, 2008, at 9:25:36 PM
 *
 *
 */
package org.lsm.finances.v2.swing.model;

import java.awt.Component;
import javax.swing.DefaultListCellRenderer;
import javax.swing.JList;
import org.lsm.finances.v2.object.Account;
import org.lsm.finances.v2.object.Category;

public class Renderers {

    public static class CatListRenderer extends DefaultListCellRenderer {

        @Override
        public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
            super.getListCellRendererComponent(list, (value == null) ? null : ((Category) value).getName(), index, isSelected, cellHasFocus);
            return this;
        }
    }

    public static class AccountListRenderer extends DefaultListCellRenderer {

        @Override
        public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
            return super.getListCellRendererComponent(list, (value == null) ? null : ((Account) value).getName(), index, isSelected, cellHasFocus);
        }
    }
}