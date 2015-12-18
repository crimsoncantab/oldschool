/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.lsm.finances.v2.swing.model;

import java.awt.Component;
import javax.swing.JTable;
import javax.swing.table.DefaultTableCellRenderer;

/**
 *
 * @author Loren
 */
public class MyRenderer extends DefaultTableCellRenderer {

    @Override
    public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected, boolean hasFocus, int row, int column) {
        TableCell cell = (TableCell) value;
        Component ret = super.getTableCellRendererComponent(table, cell.value, isSelected, hasFocus, row, column);
        ret.setForeground(cell.color);
        return ret;
    }
}
