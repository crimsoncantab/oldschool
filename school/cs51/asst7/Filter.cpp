#include "Filter.h"
#include <iostream>
#include <fstream>
#include <string>

Filter::Filter()
{
}

Filter::Filter(Mailbox * in, const char * rules_file)
{   
    // This constructor has to populate "rulesList", and "inBox";
    inBox = in;
    outBox = new Mailbox();
    
    // Read in the rules file into the ruleList vector
    ifstream ifs (rules_file);
    string tempLine;
    
    while (getline(ifs, tempLine))
    {
        // Split each line, create a PatternRule, and push onto array
        int space_pos = tempLine.find_first_of(' ', 0);
        
        string f = tempLine.substr(0, space_pos);
        string p = tempLine.substr(space_pos+1);
        
        Rule * tmpRule = new PatternRule(f, p);
        rulesList.push_back(tmpRule);
    }
    ifs.close();
}

Filter::~Filter()
{
    for (unsigned int i=0; i<rulesList.size(); i++)
        delete rulesList[i];
    
    delete outBox;
}

void Filter::clean()
{
    // Go through each mailMessage, while checking each rule if needed on it
    for (int i=0; i < (inBox->getMessageCount()); i++)
    {   
        unsigned int counter = 0;
        for (unsigned int j=0; j<rulesList.size(); j++)
        {
            // if message is fine, add to new mailbox
            if (rulesList[j]->Apply( inBox->nthMessage(i) ))
                counter++;
        }
        if (counter == rulesList.size())
            outBox->addMessage( inBox->nthMessage(i) );
    }
}

Mailbox * Filter::getOutbox()
{
    return outBox;
}
