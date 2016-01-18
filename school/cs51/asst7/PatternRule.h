#ifndef _PATTERNRULE_
#define _PATTERNRULE_

#include <string>

#include "MailMessage.h"
#include "Rule.h"

using namespace std;

class PatternRule : public Rule
{
    private:
        string pattern;
        string field;

    public:
        PatternRule();
        PatternRule(string &f, string &p);
        bool Apply(MailMessage *m);
};

#endif //_PATTERNRULE_
