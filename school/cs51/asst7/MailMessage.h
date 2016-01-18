#ifndef _MAILMESSAGE_
#define _MAILMESSAGE_

#include <string>
#include <map>

using namespace std;

class MailMessage
{
    private:
        string message;
        map<string, string> fields;

    public:
        MailMessage();
        MailMessage(string &m);
        const string &getField(const string &field);
        void addField(const string &field, const string &text);
        void extendField(const string &field, const string &text);
        void setMessage(string &newMessage);
        string getMessage();

};

#endif //_MAILMESSAGE_
