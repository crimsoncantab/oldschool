/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.lsm.finances.v2.object;

import java.text.DateFormat;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 *
 * @author Loren
 */
public class Transaction {

    public static final DateFormat transDateFormat = new SimpleDateFormat("MM/dd/yyyy");
    public static final NumberFormat transAmountFormat = new DecimalFormat("#,##0.00;-#");
    private final int id;
    private String description;
    private Date date;
    private Account account;
    private Category category;
    private double amount;
    private boolean active = true;

    public Transaction(int id, String description, Date date, Account account, double amount, Category category) {
        this.id = id;
        this.description = description;
        this.date = date;
        this.account = account;
        this.category = category;
        this.amount = amount;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
        this.account.calcBalance();
    }

    public Account getAccount() {
        return account;
    }

    public void setAccount(Account account) {
        this.account.removeTransaction(this);
        this.account = account;
        account.addTransaction(this);
    }

    public double getAmount() {
        return amount;
    }

    public void setAmount(double amount) {
        this.amount = amount;
        this.account.calcBalance();
    }

    public Category getCategory() {
        return category;
    }

    public void setCategory(Category category) {
        this.category = category;
    }

    public Date getDate() {
        return date;
    }

    public void setDate(Date date) {
        this.date = date;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public int getId() {
        return id;
    }
}
