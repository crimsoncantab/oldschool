package org.lsm.finances.v2.swing.model;

import java.awt.Color;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.table.AbstractTableModel;
import org.lsm.finances.v2.object.Account;
import org.lsm.finances.v2.object.FinanceProfile;
import org.lsm.finances.v2.object.ProfileChangeListener;

public class AccountsTableModel extends AbstractTableModel implements ProfileChangeListener {

    public static enum Column {

        NAME("Name", String.class, true, true),
        DESCRIPTION("Description", String.class, true, true),
        BALANCE("Balance", Double.class, true, false);
        private static Map<Integer, Column> indices = new HashMap<Integer, Column>();


        static {
            repopulateIndices();
        }
        private String name;
        private Class klass;
        private boolean visible;
        private boolean editable;

        public boolean isEditable() {
            return editable;
        }

        public boolean isVisible() {
            return visible;
        }

        public void setVisible(boolean visible) {
            this.visible = visible;
            repopulateIndices();
        }

        private Column(String name, Class klass, boolean visible, boolean editable) {
            this.name = name;
            this.klass = klass;
            this.visible = visible;
            this.editable = editable;
        }

        public Class getKlass() {
            return klass;
        }

        public String getName() {
            return name;
        }

        public static Column colFromIndex(int index) {
            return indices.get(index);
        }

        public static void repopulateIndices() {
            for (Column c : Column.values()) {
                indices.put(c.ordinal(), c);
            }
        }
    }
    private List<Account> data;
    private FinanceProfile profile;

    public AccountsTableModel(FinanceProfile profile) {
        this.profile = profile;
        profile.addListener(this);
        this.data = profile.getAccounts();
    }

    @Override
    public void notifyAccountChange() {
        fireTableDataChanged();
    }

    @Override
    public void notifyTransactionChange() {
        for (int i = 0; i < data.size(); i++) {
            fireTableCellUpdated(i, Column.BALANCE.ordinal());
        }
    }

    @Override
    public void notifyCategoryChange() {
        //do nothing
    }

    @Override
    public void notifyChange() {
        fireTableDataChanged();
    }

    @Override
    public void reload(FinanceProfile profile) {
        this.profile = profile;
        data = profile.getAccounts();
        fireTableDataChanged();
    }

    @Override
    public int getRowCount() {
        return data.size();
    }

    @Override
    public int getColumnCount() {
        return Column.values().length;
    }

    @Override
    public Object getValueAt(int rowIndex, int columnIndex) {
        Object value;
        Account row = data.get(rowIndex);
        double balance = row.getBalance().getBalance();
        Color color = null;


        switch (Column.colFromIndex(columnIndex)) {
            case BALANCE:
                value = "$" + Account.accountNumberFormat.format(row.getBalance().getBalance());
                if (balance < 0) {
                    color = Color.RED;
                } else {
                    color = Color.GREEN;
                }
                break;
            case DESCRIPTION:
                value = row.getDescription();
                break;
            case NAME:
                value = row.getName();
                break;
            default:
                throw new AssertionError("Unhandled enum value");
        }

        return new TableCell(value, color);
    }

    @Override
    public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
        if (aValue == null) {
            return;
        }
        Account row = data.get(rowIndex);
        switch (Column.colFromIndex(columnIndex)) {
            case BALANCE:
                throw new IllegalArgumentException("Column is not editable");
            case DESCRIPTION:
                row.setDescription((String) aValue);
                break;
            case NAME:
                row.setName((String) aValue);
                break;
            default:
                throw new AssertionError("Unhandled enum value");
        }
        profile.accountModified(row);
    }

    @Override
    public Class<?> getColumnClass(int columnIndex) {
        return Column.colFromIndex(columnIndex).getKlass();
    }

    @Override
    public String getColumnName(int column) {
        return Column.colFromIndex(column).getName();
    }

    @Override
    public boolean isCellEditable(int rowIndex, int columnIndex) {
        return Column.colFromIndex(columnIndex).editable;
    }
}
