/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package org.lsm.finances.v2.object;

/**
 *
 * @author Loren
 */
public class Category {

    private String name;
    private final int id;
    public static final int NO_CAT_ID = -1;
    public static final Category NO_CAT = new Category("None", NO_CAT_ID);

    public Category(String name, int id) {
        this.name = name;
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

}
