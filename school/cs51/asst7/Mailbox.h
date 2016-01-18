#ifndef _MAILBOX_
#define _MAILBOX_

#include "MailMessage.h"
#include "mbox.h"
#include <vector>
#include <string>

class Mailbox
{
    private:
        vector<MailMessage*> messages;
        int count;
        void createMessage(mbox *m, mbox::iterator &i);

    public:
        Mailbox();
        Mailbox(string &fileName);
        ~Mailbox();
        void addMessage(MailMessage *m);

        void parse(string &fileName);
        
        int getMessageCount();
        MailMessage *nthMessage(int n);

        void writeToFile(char * fileName);
};

#endif //_MAILBOX_
