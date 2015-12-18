/* @author Loren
 *
 * Created on: Nov 5, 2008, at 6:51:08 PM
 *
 *
 */
package org.lsm.finances.v2.swing.model;

import java.awt.Component;
import javax.swing.DefaultCellEditor;
import javax.swing.JTable;
import javax.swing.JTextField;

public class Editors {

    public static class StringEditor extends DefaultCellEditor {

        public StringEditor() {
            super(new JTextField());
        }

        @Override
        public Component getTableCellEditorComponent(JTable table, Object value, boolean isSelected, int row, int column) {
            return super.getTableCellEditorComponent(table, ((TableCell) value).value, isSelected, row, column);
        }
    }
}
