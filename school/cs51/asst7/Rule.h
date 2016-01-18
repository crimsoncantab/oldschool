#ifndef _RULE_
#define _RULE_

#include "MailMessage.h"

class Rule
{
    private:

    public:
        Rule();
        virtual ~Rule() { };

        //returns false if the message is spam
        virtual bool Apply(MailMessage *m) = 0;

};

#endif //_RULE_
