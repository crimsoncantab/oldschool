#include <iostream>
#include <fstream>
#include <string>

#include "Mailbox.h"
#include "MailMessage.h"
#include "mbox.h"

using namespace std;

Mailbox::Mailbox()
{
    count = 0;
}

Mailbox::Mailbox(string &fileName)
{
    count = 0;
    parse(fileName);
}

Mailbox::~Mailbox()
{
    for (int i = 0; i < count; i++)
        delete messages[i];
}

void Mailbox::addMessage(MailMessage *m)
{
    messages.push_back(m);
    count++;
}

void Mailbox::parse(string &fileName)
{
    mbox *m = new mbox(fileName);

    mbox::iterator cur = m->begin();
    
    //iterate through mailbox
    while (cur != m->end())
    {
        //finds and adds every new message
        if (cur->tag == mbox::token::FromLine)
        {
            createMessage(m, cur);
        }
        cur++;
    }
}
void Mailbox::createMessage(mbox *m, mbox::iterator &i)
{
    //initialize temp strings and new message
    string message = "", curHeader = "";
    MailMessage * newM = new MailMessage();
    
    //add from field
    newM->addField("From", i->text);
    message += "From " + i->text + "\n";
    //iterate to next token
    i++;
    
    //iterate until next from or EOF
    while (i != m->end() && i->tag != mbox::token::FromLine)
    {
        //handle header
        if (i->tag == mbox::token::HdrLabel)
        {
            curHeader == i->text;
            message += i->text + ": ";
        }
        //append header and its value to message's fields
        else if (i->tag == mbox::token::HdrValue)
        {
            message += i->text + "\n";
            newM->extendField(curHeader, i->text);
        }
        //handle body lines
        else
        {
            message += "\n" + i->text;
            newM->extendField("Body", i->text);
        }
        
        //iterate to next token
        i++;
    }
    
    //set message and add to list of Mailbox
    newM->setMessage(message);
    addMessage(newM);
    return;
}


int Mailbox::getMessageCount()
{
    return count;
}

MailMessage* Mailbox::nthMessage(int n)
{
    return messages[n];
}

void Mailbox::writeToFile(char * filename)
{
    ofstream fh;
    fh.open(filename);
    for (int i=0; i<getMessageCount(); i++)
        fh << nthMessage(i)->getMessage() << endl;
    fh.close();
}
