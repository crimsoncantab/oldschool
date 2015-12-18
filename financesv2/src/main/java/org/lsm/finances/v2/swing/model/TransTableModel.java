package org.lsm.finances.v2.swing.model;

import java.awt.Color;
import java.text.ParseException;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.table.AbstractTableModel;
import org.lsm.finances.v2.object.Account;
import org.lsm.finances.v2.object.Category;
import org.lsm.finances.v2.object.FinanceProfile;
import org.lsm.finances.v2.object.ProfileChangeListener;
import org.lsm.finances.v2.object.Transaction;

public class TransTableModel extends AbstractTableModel implements ProfileChangeListener {

    public enum Column {

//		ID("ID", Integer.class, false, false),
        ACCOUNT("Account", Account.class, true, true),
        DESCRIPTION("Description", String.class, true, true),
        DATE("Date", Date.class, true, true),
        AMOUNT("Amount", Double.class, true, true),
        CATEGORY("Category", Category.class, true, true),
        ACTIVE("Active", Boolean.class, true, true);
        private static Map<Integer, Column> indices = new HashMap<Integer, Column>();


        static {
            for (Column c : Column.values()) {
                indices.put(c.ordinal(), c);
            }
        }
        private String name;
        private Class klass;
        private boolean visible;
        private boolean editable;

        public boolean isVisible() {
            return visible;
        }

        public void setVisible(boolean visible) {
            this.visible = visible;
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
    }
    private List<Transaction> data;
    private FinanceProfile profile;

    public TransTableModel(FinanceProfile profile) {
        this.profile = profile;
        profile.addListener(this);
        this.data = profile.getTransactions();
    }

    @Override
    public void notifyAccountChange() {
        fireTableDataChanged();
    }

    @Override
    public void notifyTransactionChange() {
        fireTableDataChanged();
    }

    @Override
    public void notifyCategoryChange() {
        for (int i = 0; i < data.size(); i++) {
            fireTableCellUpdated(i, Column.CATEGORY.ordinal());
        }
    }

    @Override
    public void notifyChange() {
        fireTableDataChanged();
    }

    @Override
    public void reload(FinanceProfile profile) {
        data = profile.getTransactions();
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
        Transaction row = data.get(rowIndex);
        Color color;
        if (row.isActive()) {
            color = null;
        } else {
            color = Color.GRAY;
        }


        switch (Column.colFromIndex(columnIndex)) {
            case AMOUNT:
                double amount = row.getAmount();
                if (color == null) {
                    if (amount < 0) {
                        color = Color.RED;
                    } else {
                        color = Color.GREEN;
                    }
                }
                value = "$" + Transaction.transAmountFormat.format(amount);
                break;
            case DESCRIPTION:
                value = row.getDescription();
                break;
            case ACCOUNT:
                value = row.getAccount().getName();
                break;
            case ACTIVE:
                return row.isActive();
            case CATEGORY:
                value = row.getCategory().getName();
                break;
            case DATE:
                value = Transaction.transDateFormat.format(row.getDate());
                break;
            default:
                throw new AssertionError("Unhandled enum value");
        }

        return new TableCell(value, color);
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

    @Override
    public void setValueAt(Object aValue, int rowIndex, int columnIndex) {
        Transaction row = data.get(rowIndex);
        switch (Column.colFromIndex(columnIndex)) {
            case ACTIVE:
                if (aValue == null) {
                    aValue = true;
                }
                row.setActive((Boolean) aValue);
                break;
            case DESCRIPTION:
                if (aValue == null) {
                    return;
                }
                row.setDescription((String) aValue);
                break;
            case ACCOUNT:
                if (aValue == null) {
                    return;
                }
                row.setAccount((Account) aValue);
                break;
            case CATEGORY:
                if (aValue == null) {
                    aValue = Category.NO_CAT;
                }
                row.setCategory((Category) aValue);
                break;
            case DATE:
                try {
                    row.setDate(Transaction.transDateFormat.parse((String) aValue));
                } catch (ParseException ex) {
                    //give up
                }
                break;
            case AMOUNT:
                try {
                    row.setAmount(Transaction.transAmountFormat.parse((String) aValue).doubleValue());
                } catch (ParseException ex) {
                    //give up
                }
                break;
            default:
                throw new AssertionError("Unhandled enum value");
        }
        profile.transactionModified(row);
    }
}
