/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.lsm.finances.v2.object;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.ParseException;
import java.util.LinkedList;
import java.util.List;
import org.lsm.finances.v2.dao.CsvProfileDao;

/**
 *
 * @author Loren
 */
public class FinanceProfile {

    public static DecimalFormat moneyFormat = new DecimalFormat("$#,##0.00;-$#,##0.00");
    private static CsvProfileDao dao = new CsvProfileDao();
    private List<Transaction> transactions;
    private List<Account> accounts;
    private List<Category> categories;
    private File profileFile;
    private List<ProfileChangeListener> listeners;
    private int transId = 0;
    private int accountId = 0;
    private int catId = 0;
    private boolean needsSaving = false;

    public FinanceProfile(List<Transaction> transactions, List<Account> accounts, List<Category> categories, File profileFile, List<ProfileChangeListener> listeners) {
        this.transactions = transactions;
        this.accounts = accounts;
        this.categories = categories;
        this.profileFile = profileFile;
        this.listeners = listeners;
        for (Transaction t : transactions) {
            if (t.getId() > transId) {
                transId = t.getId();
            }
        }
        transId++;
        for (Account a : accounts) {
            if (a.getId() > accountId) {
                accountId = a.getId();
            }
        }
        accountId++;
        for (Category c : categories) {
            if (c.getId() > catId) {
                catId = c.getId();
            }
        }
        catId++;

    }

    public FinanceProfile(List<Transaction> transactions, List<Account> accounts, List<Category> categories, List<ProfileChangeListener> listeners) {
        this(transactions, accounts, categories, null, listeners);
    }

    public FinanceProfile(List<Transaction> transactions, List<Account> accounts, List<Category> categories, File profileFile) {
        this(transactions, accounts, categories, profileFile, new LinkedList<ProfileChangeListener>());
    }

    public FinanceProfile(File profileFile) {
        this(new LinkedList<Transaction>(), new LinkedList<Account>(), new LinkedList<Category>(), profileFile);
    }

    public FinanceProfile() {
        this(null);
    }

    public void addTransaction(Transaction t) {
        if (!accounts.contains(t.getAccount())) {
            accounts.add(t.getAccount());
        }
        t.getAccount().addTransaction(t);
        transactions.add(t);
        notifyTransactionChange();
    }

    public void removeTransaction(Transaction t) {
        transactions.remove(t);
        t.getAccount().removeTransaction(t);
        notifyTransactionChange();
    }

    public List<Transaction> getTransactions() {
        return transactions;
    }

    public void addAccount(Account a) {
        accounts.add(a);
        transactions.addAll(a.getTransactions());
        notifyAccountChange();
    }

    public void removeAccount(Account a) {
        accounts.remove(a);
        List<Transaction> toRemove = new LinkedList<Transaction>();
        for (Transaction t : transactions) {
            if (t.getAccount().equals(a)) {
                toRemove.add(t);
            }
        }
        transactions.removeAll(toRemove);
        notifyChange();
    }

    public List<Account> getAccounts() {
        return accounts;
    }

    public void addCategory(Category c) {
        categories.add(c);
        notifyCategoryChange();
    }

    public void removeCategory(Category c) {
        categories.remove(c);
        for (Transaction t : transactions) {
            if (t.getCategory().equals(c)) {
                t.setCategory(Category.NO_CAT);
            }
        }
        notifyTransactionChange();
        notifyCategoryChange();
    }

    public List<Category> getCategories() {
        return categories;
    }

    public File getFile() {
        return profileFile;
    }

    public void save() throws IOException {
        save(profileFile);
    }

    public void save(File file) throws IOException {
        if (file == null) {
            throw new IllegalArgumentException("No file specified");
        } else {
            dao.save(file, this);
        }
        profileFile = file;
        needsSaving = false;
    }

    public boolean isNew() {
        return profileFile == null;
    }

    public boolean needsSaving() {
        return needsSaving;
    }

    public static FinanceProfile load(File file, FinanceProfile old) throws FileNotFoundException, IOException, ParseException {
        FinanceProfile profile = dao.load(file, old.listeners);
        for (ProfileChangeListener listener : profile.listeners) {
            listener.reload(profile);
        }
        return profile;
    }

    public int getNewTransId() {
        int ret = transId;
        transId++;
        return ret;
    }

    public int getNewAccountId() {
        int ret = accountId;
        accountId++;
        return ret;

    }

    public int getNewCatId() {
        int ret = catId;
        catId++;
        return ret;
    }

    public void addListener(ProfileChangeListener listener) {
        listeners.add(listener);
    }

    public void accountModified(Account a) {
        notifyAccountChange();
    }

    public void transactionModified(Transaction t) {
        for (Account account : accounts) {
            account.calcBalance();
        }
        notifyTransactionChange();
    }

    private void notifyAccountChange() {
        needsSaving = true;
        for (ProfileChangeListener listener : listeners) {
            listener.notifyAccountChange();
        }
    }

    private void notifyTransactionChange() {
        needsSaving = true;
        for (ProfileChangeListener listener : listeners) {
            listener.notifyTransactionChange();
        }
    }

    private void notifyCategoryChange() {
        needsSaving = true;
        for (ProfileChangeListener listener : listeners) {
            listener.notifyCategoryChange();
        }
    }

    private void notifyChange() {
        needsSaving = true;
        for (ProfileChangeListener listener : listeners) {
            listener.notifyChange();
        }
    }
}
