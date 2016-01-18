#include "PatternRule.h"

PatternRule::PatternRule()
{
}
PatternRule::PatternRule(string &f, string &p)
{
    field = f;
    pattern = p;
}
bool PatternRule::Apply(MailMessage *m)
{
    string s = m->getField(field);
    string::size_type loc = s.find(pattern,0);
    return (loc == string::npos);
}
