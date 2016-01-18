#include "MailMessage.h"

MailMessage::MailMessage()
{
}

MailMessage::MailMessage(string &m)
{
    message = m;
}

const string& MailMessage::getField(const string& field)
{
    return fields[field];
}

void MailMessage::addField(const string &field, const string &text)
{
    fields[field] = text;
}

void MailMessage::extendField(const string &field, const string &text)
{
	fields[field] += text;
}

void MailMessage::setMessage(string &newMessage)
{
    message = newMessage;
}

string MailMessage::getMessage()
{
    return message;
}
