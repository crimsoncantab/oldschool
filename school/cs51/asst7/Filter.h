#ifndef _FILTER_
#define _FILTER_

#include "Mailbox.h"
#include "PatternRule.h"
#include "Rule.h"
#include <vector>
#include <string>

class Filter
{
    private:
        vector<Rule *> rulesList;
        Mailbox *inBox;
        Mailbox *outBox;
        
    public:
        Filter();
        Filter(Mailbox *in,const char * rules_file);
        ~Filter();
        void clean();
        Mailbox * getOutbox();
};

#endif //_FILTER_
