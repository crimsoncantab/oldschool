/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.lsm.finances.v2.object;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.LinkedList;
import java.util.List;

/**
 *
 * @author Loren
 */
public class Account {

    public static final NumberFormat accountNumberFormat = new DecimalFormat("#,##0.00;-#");

    public static class AccountBalance {

        private double income;
        private double expense;

        private AccountBalance() {
            income = 0;
            expense = 0;
        }

        private void calcBalance(List<Transaction> transactions) {
            income = 0;
            expense = 0;
            for (Transaction transaction : transactions) {
                if (transaction.isActive()) {
                    addToBalance(transaction.getAmount());
                }
            }
        }

        private void addToBalance(double amount) {
            if (amount < 0) {
                expense += amount;
            } else {
                income += amount;
            }
        }

        public double getBalance() {
            return income + expense;
        }

        public double getExpense() {
            return expense;
        }

        public double getIncome() {
            return income;
        }
    }
    private final int id;
    private final AccountBalance balance;
    private final List<Transaction> transactions;
    private String name;
    private String description;

    public Account(int id, List<Transaction> transactions, String name, String description) {
        this.id = id;
        this.transactions = transactions;
        this.name = name;
        this.description = description;
        this.balance = new AccountBalance();
        balance.calcBalance(transactions);

    }

    public Account(int id, String name, String description) {
        this(id, new LinkedList<Transaction>(), name, description);
    }

    public String getDescription() {
        return description;
    }

    public String getName() {
        return name;
    }

    public List<Transaction> getTransactions() {
        return transactions;
    }

    public AccountBalance getBalance() {
        return balance;
    }

    public int getId() {
        return id;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void removeTransaction(Transaction transaction) {
        transactions.remove(transaction);
        balance.calcBalance(transactions);
    }

    public void addTransaction(Transaction transaction) {
        transactions.add(transaction);
        balance.calcBalance(transactions);
    }

    public void calcBalance() {
        balance.calcBalance(transactions);
    }
}
