// DictList.h v2.0

#include<string>

using namespace std;

struct LLNode {
    string first;
    string second;
    LLNode *next;
    
    LLNode(string f, string s);

    ~LLNode();
};

class DictList {
private:
    LLNode *head;

public:
    DictList();
    ~DictList();
    void extend(string key, string value);
    string lookup(string key);
    string keys();
};
