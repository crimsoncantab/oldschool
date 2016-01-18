// DictList.h v2.0

#include<string>

using namespace std;

struct TreeNode {
    string first;
    string second;
    TreeNode *left;
    TreeNode *right;
    
    TreeNode(string f, string s);

    ~TreeNode();
};

class DictTree {
private:
    TreeNode *root;
    string traverse(TreeNode *node);

public:
    DictTree();
    ~DictTree();
    void extend(string key, string value);
    string lookup(string key);
    string keys();
};
