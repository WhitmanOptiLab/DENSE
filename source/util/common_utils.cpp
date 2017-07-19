#include "common_utils.hpp"

string file_add_num(string prFileName, const string& pcfFillPrefix,
        const char& pcfFillWith, const int& pcfFillMe,
        const int& pcfFillLen, const string& pcfFillAt)
{
    prFileName = prFileName.substr(0,
        prFileName.find_last_of(pcfFillAt)) + pcfFillPrefix +
        cfill(to_string(pcfFillMe), pcfFillWith, pcfFillLen) +
        prFileName.substr(prFileName.find_last_of(pcfFillAt));
    return prFileName;
}

string cfill(string prFileName, const char& pcfFillWith, const int& pcfFillLen)
{
    while (prFileName.length() < pcfFillLen)
    {
        prFileName = pcfFillWith + prFileName;
    }

    return prFileName;
}

