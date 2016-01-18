// DictTree.cpp v2.0

#include<string>
#include<iostream>
#include<cassert>
#include "DictTree.h"

using namespace std;

//
// Implementation for TreeNode
//

TreeNode::TreeNode(string f, string s) {
    first = f;
    second = s;
    left = NULL;
    right = NULL;
}

TreeNode::~TreeNode() {
    // delete does not break if it is passed NULL
    delete left;
    delete right;
}

//
// Implementation for DictTree
//

DictTree::DictTree() {
    root = NULL;
}

DictTree::~DictTree() {
    // delete does not break if it is passed NULL
    delete root;
}
    
void DictTree::extend(string key, string value) { 
    // If we find the key in the list, replace its value with the new one
    // and return
    TreeNode *prev, *trav;
    prev = trav = root;
    while (trav != NULL) {
        if(trav->first == key) {
            trav->second = value;
            return;
        }
        prev = trav;
        if (trav->first < key) {
            trav = trav->right;
            if (trav == NULL)
                prev->right = new TreeNode(key, value);
        }
        else {
            trav = trav->left;
            if (trav == NULL)
                prev->left = new TreeNode(key, value);
        }
    }

    if(prev == NULL) {
        // List was empty
        root = new TreeNode(key, value);
    }
}

string DictTree::lookup(string key) {
    // Try to find the key
    TreeNode *trav = root;
    while (trav != NULL) {
        if (trav->first == key) 
            return trav->second;
        else if (trav->first < key)
            trav = trav->right;
        else
            trav = trav->left;
    }

    // If we didn't find anything, return "" to signify this
    return "";
}

string DictTree::keys() {
    return traverse(root);
}

string DictTree::traverse(TreeNode *node) {
    if (node == NULL)
        return "";
    return (traverse(node->left) + node->first + "*" + traverse(node->right));
}

