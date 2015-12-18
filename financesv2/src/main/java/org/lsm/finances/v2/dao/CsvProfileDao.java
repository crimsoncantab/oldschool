/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.lsm.finances.v2.dao;

import au.com.bytecode.opencsv.CSVReader;
import au.com.bytecode.opencsv.CSVWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import org.lsm.finances.v2.object.Account;
import org.lsm.finances.v2.object.Category;
import org.lsm.finances.v2.object.FinanceProfile;
import org.lsm.finances.v2.object.ProfileChangeListener;
import org.lsm.finances.v2.object.Transaction;

/**
 *
 * @author Loren
 */
public class CsvProfileDao {

    public CsvProfileDao() {
    }

    public void save(File file, FinanceProfile profile) throws IOException {
        CSVWriter writer = new CSVWriter(new FileWriter(file));

        for (Category category : profile.getCategories()) {
            System.out.println("Saving category " + category.getName());
            String[] catRow = {"C", category.getId() + "", category.getName()};
            writer.writeNext(catRow);
        }
        for (Account account : profile.getAccounts()) {
            System.out.println("Saving account " + account.getName());
            String[] accountRow = {"A", account.getId() + "", account.getName(), account.getDescription()};
            writer.writeNext(accountRow);
            saveTransactions(writer, account);
        }
        writer.close();
    }

    private void saveTransactions(CSVWriter writer, Account account) {
        for (Transaction transaction : account.getTransactions()) {
            String transDate = Transaction.transDateFormat.format(transaction.getDate());
            String transAmount = Transaction.transAmountFormat.format(transaction.getAmount());
            String transCat = transaction.getCategory().getId() + "";
            String[] transRow = {"T", transaction.getId() + "", transaction.getDescription(), transDate, transAmount, transCat};
            writer.writeNext(transRow);
        }
    }

    public FinanceProfile load(File file) throws FileNotFoundException, IOException, ParseException {
        return load(file, null);
    }

    public FinanceProfile load(File file, List<ProfileChangeListener> listeners) throws FileNotFoundException, IOException, ParseException {
        CSVReader reader = new CSVReader(new FileReader(file));
        String[] curRow;
        List<Account> accounts = new LinkedList<Account>();
        List<Transaction> transactions = new LinkedList<Transaction>();
        Map<Integer, Category> categoriesMap = new HashMap<Integer, Category>();
        Account tempAccount = null;
        Integer categoryId = null;
        while ((curRow = reader.readNext()) != null) {
            if (curRow[0] != null && !curRow[0].isEmpty()) {
                switch (curRow[0].charAt(0)) {
                    case 'A':
                        tempAccount = new Account(Integer.parseInt(curRow[1]), curRow[2], curRow[3]);
                        accounts.add(tempAccount);
                        break;
                    case 'C':
                        //we hope to get all of these before any transactions
                        categoryId = Integer.parseInt(curRow[1]);
                        categoriesMap.put(categoryId, new Category(curRow[2], categoryId));
                        break;
                    case 'T':
                        categoryId = Integer.parseInt(curRow[5]);
                        Category c = (categoryId == Category.NO_CAT_ID) ? Category.NO_CAT : categoriesMap.get(categoryId);

                        Transaction trans = new Transaction(Integer.parseInt(curRow[1]), curRow[2], Transaction.transDateFormat.parse(curRow[3]), tempAccount, Transaction.transAmountFormat.parse(curRow[4]).doubleValue(), c);
                        transactions.add(trans);
                        tempAccount.addTransaction(trans);
                        break;
                    default:
                    //this shouldn't happen, but for now we'll ignore it.
                    //might be an empty row or something
                }
            }
        }
        List<Category> categories = new LinkedList<Category>();
        categories.addAll(categoriesMap.values());

        FinanceProfile profile;
        if (listeners == null) {
            profile = new FinanceProfile(transactions, accounts, categories, file);
        } else {
            profile = new FinanceProfile(transactions, accounts, categories, file, listeners);
            for (ProfileChangeListener listener : listeners) {
                listener.reload(profile);
            }
        }
        reader.close();
        return profile;
    }
}
