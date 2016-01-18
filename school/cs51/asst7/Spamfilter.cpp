#include <iostream>
#include <string>

#include "Rule.h"
#include "MailMessage.h"
#include "PatternRule.h"
#include "Filter.h"
#include "Mailbox.h"

using namespace std;


int main(int argc, char* argv[])
{
    // Make sure only 3 arguments
    if (argc != 4)
    {
        cout << "Usage: spamfilter mbox_in mbox_out rules_file" << endl;
        return -1;
    }
    
    string mbox_in = (string) argv[1];
    
    Mailbox * mailboxIn = new Mailbox(mbox_in);

    // Using rules, filter out bad messages, and create new mbox file
    Filter * filterer = new Filter(mailboxIn, argv[3]);
    Mailbox * mailboxOut = new Mailbox();
    filterer->clean();
    mailboxOut = filterer->getOutbox();
    
    // Write this filtered mbox file back out as mbox_out
    mailboxOut->writeToFile(argv[2]);

    return 0;
}
