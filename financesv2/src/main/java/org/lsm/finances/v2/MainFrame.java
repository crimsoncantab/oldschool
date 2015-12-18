/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * MainFrame.java
 *
 * Created on May 31, 2009, 11:30:57 AM
 */
package org.lsm.finances.v2;

import java.awt.Frame;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.ParseException;
import javax.swing.DefaultCellEditor;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JTable;
import javax.swing.table.TableCellEditor;
import javax.swing.table.TableCellRenderer;
import org.lsm.finances.v2.object.FinanceProfile;
import org.lsm.finances.v2.swing.AccountDialog;
import org.lsm.finances.v2.swing.CatDialog;
import org.lsm.finances.v2.swing.TransDialog;
import org.lsm.finances.v2.swing.model.AccountListModel;
import org.lsm.finances.v2.swing.model.AccountsTableModel;
import org.lsm.finances.v2.swing.model.CategoryListModel;
import org.lsm.finances.v2.swing.model.Editors;
import org.lsm.finances.v2.swing.model.MyRenderer;
import org.lsm.finances.v2.swing.model.Renderers;
import org.lsm.finances.v2.swing.model.TransTableModel;

/**
 *
 * @author Loren
 */
public class MainFrame extends JFrame {

    private final TransDialog transDialog;
    private final AccountDialog accountDialog;
    private final CatDialog catDialog;
    private FinanceProfile profile;

    public MainFrame() {
        profile = new FinanceProfile();
        transDialog = new TransDialog(this, profile);
        transDialog.setLocationRelativeTo(this);
        accountDialog = new AccountDialog(this, profile);
        accountDialog.setLocationRelativeTo(this);
        catDialog = new CatDialog(this, profile);
        catDialog.setLocationRelativeTo(this);
        initComponents();
        setExtendedState(Frame.MAXIMIZED_BOTH);
        MyRenderer renderer = new MyRenderer();
        Editors.StringEditor editor = new Editors.StringEditor();
        JComboBox accountEditor = new JComboBox(new AccountListModel(profile));
        accountEditor.setRenderer(new Renderers.AccountListRenderer());
        JComboBox catEditor = new JComboBox(new CategoryListModel(profile));
        catEditor.setRenderer(new Renderers.CatListRenderer());
        for (AccountsTableModel.Column c : AccountsTableModel.Column.values()) {
            initRenderer(accountTable, c.ordinal(), renderer);
            if (c != AccountsTableModel.Column.BALANCE) {
                initEditor(accountTable, c.ordinal(), editor);
            }
        }
        for (TransTableModel.Column c : TransTableModel.Column.values()) {
            if (c != TransTableModel.Column.ACTIVE) {
                initRenderer(transTable, c.ordinal(), renderer);
                if (c == TransTableModel.Column.ACCOUNT) {
                    initEditor(transTable, c.ordinal(), new DefaultCellEditor(accountEditor));
                } else if (c == TransTableModel.Column.CATEGORY) {
                    initEditor(transTable, c.ordinal(), new DefaultCellEditor(catEditor));
                } else {
                    initEditor(transTable, c.ordinal(), editor);

                }
            }
        }
        filterButton.setEnabled(false);
    }

    private void initRenderer(JTable table, int c, TableCellRenderer r) {
        table.getColumnModel().getColumn(c).setCellRenderer(r);
    }

    private void initEditor(JTable table, int c, TableCellEditor e) {
        table.getColumnModel().getColumn(c).setCellEditor(e);
    }

    public boolean save() {
        try {
            if (profile.isNew()) {
                saveAs();
            } else {
                profile.save();
            }
            return true;
        } catch (IOException ex) {
            System.err.println("Could not save file");
            return false;
        }
    }

    private boolean saveAs() {
        if (profile.getFile() != null) {
            fileChooser.setSelectedFile(profile.getFile());
        }
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                profile.save(fileChooser.getSelectedFile());
            } catch (Exception ex) {
                System.err.println("Could not save file");
                return false;
            }
            return true;
        }
        return false;
    }

    public void load() {
        int val = fileChooser.showOpenDialog(this);
        if (val == JFileChooser.APPROVE_OPTION) {
            try {
                profile = FinanceProfile.load(fileChooser.getSelectedFile(), profile);
            } catch (FileNotFoundException ex) {
                System.err.println("File could not be found.");
            } catch (IOException ex) {
                System.err.println("Could not load file");
            } catch (ParseException ex) {
                System.err.println("File formatted incorrectly");
            }
        }
    }

    /** This method is called from within the constructor to
     * initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is
     * always regenerated by the Form Editor.
     */
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        fileChooser = new javax.swing.JFileChooser();
        savePrompt = new javax.swing.JDialog();
        promptLabel = new javax.swing.JLabel();
        promptButtonPanel = new javax.swing.JPanel();
        promptSave = new javax.swing.JButton();
        promptQuit = new javax.swing.JButton();
        promptCancel = new javax.swing.JButton();
        buttonPane = new javax.swing.JPanel();
        accountAdd = new javax.swing.JButton();
        transAdd = new javax.swing.JButton();
        catButton = new javax.swing.JButton();
        filterButton = new javax.swing.JButton();
        mainPanel = new javax.swing.JPanel();
        accountScroll = new javax.swing.JScrollPane();
        accountTable = new javax.swing.JTable();
        transScroll = new javax.swing.JScrollPane();
        transTable = new javax.swing.JTable();
        jMenuBar1 = new javax.swing.JMenuBar();
        menuFile = new javax.swing.JMenu();
        menuFileSave = new javax.swing.JMenuItem();
        menuFileSaveAs = new javax.swing.JMenuItem();
        menuFileLoad = new javax.swing.JMenuItem();
        menuFileSeparator = new javax.swing.JSeparator();
        menuFileQuit = new javax.swing.JMenuItem();

        savePrompt.setMinimumSize(new java.awt.Dimension(195, 65));
        savePrompt.setModal(true);
        savePrompt.setResizable(false);

        promptLabel.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        promptLabel.setText("Would you like to save before quitting?");
        promptLabel.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        savePrompt.getContentPane().add(promptLabel, java.awt.BorderLayout.CENTER);

        promptSave.setText("Save");
        promptSave.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                promptSave(evt);
            }
        });
        promptButtonPanel.add(promptSave);

        promptQuit.setText("Quit");
        promptQuit.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                promptQuitquit(evt);
            }
        });
        promptButtonPanel.add(promptQuit);

        promptCancel.setText("Cancel");
        promptCancel.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                promptCancelcancelQuit(evt);
            }
        });
        promptButtonPanel.add(promptCancel);

        savePrompt.getContentPane().add(promptButtonPanel, java.awt.BorderLayout.PAGE_END);

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        addWindowListener(new java.awt.event.WindowAdapter() {
            public void windowClosing(java.awt.event.WindowEvent evt) {
                formWindowClosing(evt);
            }
        });

        accountAdd.setText("Add Account...");
        accountAdd.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                accountAddActionPerformed(evt);
            }
        });
        buttonPane.add(accountAdd);

        transAdd.setText("Add Transaction...");
        transAdd.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                transAddActionPerformed(evt);
            }
        });
        buttonPane.add(transAdd);

        catButton.setText("Categories...");
        catButton.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                catButtonActionPerformed(evt);
            }
        });
        buttonPane.add(catButton);

        filterButton.setText("Filter...");
        buttonPane.add(filterButton);

        getContentPane().add(buttonPane, java.awt.BorderLayout.PAGE_END);

        mainPanel.setLayout(new java.awt.GridLayout(2, 1));

        accountTable.setModel(new AccountsTableModel(profile));
        accountScroll.setViewportView(accountTable);

        mainPanel.add(accountScroll);

        transTable.setModel(new TransTableModel(profile));
        transScroll.setViewportView(transTable);

        mainPanel.add(transScroll);

        getContentPane().add(mainPanel, java.awt.BorderLayout.CENTER);

        menuFile.setText("File");

        menuFileSave.setAccelerator(javax.swing.KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_S, java.awt.event.InputEvent.CTRL_MASK));
        menuFileSave.setText("Save");
        menuFileSave.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                menuFileSaveActionPerformed(evt);
            }
        });
        menuFile.add(menuFileSave);

        menuFileSaveAs.setAccelerator(javax.swing.KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_S, java.awt.event.InputEvent.SHIFT_MASK | java.awt.event.InputEvent.CTRL_MASK));
        menuFileSaveAs.setText("Save As...");
        menuFileSaveAs.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                menuFileSaveAsActionPerformed(evt);
            }
        });
        menuFile.add(menuFileSaveAs);

        menuFileLoad.setAccelerator(javax.swing.KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_L, java.awt.event.InputEvent.CTRL_MASK));
        menuFileLoad.setText("Load...");
        menuFileLoad.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                menuFileLoadActionPerformed(evt);
            }
        });
        menuFile.add(menuFileLoad);
        menuFile.add(menuFileSeparator);

        menuFileQuit.setAccelerator(javax.swing.KeyStroke.getKeyStroke(java.awt.event.KeyEvent.VK_Q, java.awt.event.InputEvent.CTRL_MASK));
        menuFileQuit.setText("Quit");
        menuFileQuit.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                menuFileQuitActionPerformed(evt);
            }
        });
        menuFile.add(menuFileQuit);

        jMenuBar1.add(menuFile);

        setJMenuBar(jMenuBar1);

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void accountAddActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_accountAddActionPerformed
        accountDialog.setVisible(true);
    }//GEN-LAST:event_accountAddActionPerformed

    private void transAddActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_transAddActionPerformed
        transDialog.setVisible(true);
    }//GEN-LAST:event_transAddActionPerformed

    private void catButtonActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_catButtonActionPerformed
        catDialog.setVisible(true);
    }//GEN-LAST:event_catButtonActionPerformed

    private void menuFileSaveActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_menuFileSaveActionPerformed
        save();
    }//GEN-LAST:event_menuFileSaveActionPerformed

    private void menuFileSaveAsActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_menuFileSaveAsActionPerformed
        saveAs();
    }//GEN-LAST:event_menuFileSaveAsActionPerformed

    private void promptSave(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_promptSave
        savePrompt.setVisible(false);
        save();
}//GEN-LAST:event_promptSave

    private void promptQuitquit(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_promptQuitquit
        System.exit(0);
}//GEN-LAST:event_promptQuitquit

    private void promptCancelcancelQuit(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_promptCancelcancelQuit
        savePrompt.setVisible(false);
}//GEN-LAST:event_promptCancelcancelQuit

    private void menuFileQuitActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_menuFileQuitActionPerformed
        if (profile.needsSaving()) {
            savePrompt.setLocationRelativeTo(this);
            savePrompt.pack();
            savePrompt.setVisible(true);
        } else {
            promptQuitquit(evt);
        }
    }//GEN-LAST:event_menuFileQuitActionPerformed

    private void menuFileLoadActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_menuFileLoadActionPerformed
        if (!profile.needsSaving()) {
            load();
        }
    }//GEN-LAST:event_menuFileLoadActionPerformed

    private void formWindowClosing(java.awt.event.WindowEvent evt) {//GEN-FIRST:event_formWindowClosing
        menuFileQuitActionPerformed(null);
    }//GEN-LAST:event_formWindowClosing

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton accountAdd;
    private javax.swing.JScrollPane accountScroll;
    private javax.swing.JTable accountTable;
    private javax.swing.JPanel buttonPane;
    private javax.swing.JButton catButton;
    private javax.swing.JFileChooser fileChooser;
    private javax.swing.JButton filterButton;
    private javax.swing.JMenuBar jMenuBar1;
    private javax.swing.JPanel mainPanel;
    private javax.swing.JMenu menuFile;
    private javax.swing.JMenuItem menuFileLoad;
    private javax.swing.JMenuItem menuFileQuit;
    private javax.swing.JMenuItem menuFileSave;
    private javax.swing.JMenuItem menuFileSaveAs;
    private javax.swing.JSeparator menuFileSeparator;
    private javax.swing.JPanel promptButtonPanel;
    private javax.swing.JButton promptCancel;
    private javax.swing.JLabel promptLabel;
    private javax.swing.JButton promptQuit;
    private javax.swing.JButton promptSave;
    private javax.swing.JDialog savePrompt;
    private javax.swing.JButton transAdd;
    private javax.swing.JScrollPane transScroll;
    private javax.swing.JTable transTable;
    // End of variables declaration//GEN-END:variables
}
