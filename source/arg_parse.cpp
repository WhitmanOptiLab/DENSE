#include "arg_parse.hpp"
#include "color.hpp"

#include "reaction.hpp" // for "typedef float RATETYPE;"

#include <cstring>
#include <iostream>
#include <exception>
#include <vector>
using namespace std;



namespace arg_parse
{
    // anonymous namespace
    namespace
    {
        // For storing a copy of *argv[]
        vector<string> iArgVec;
        
        // Get index of (or index after) pcOpt if it exists in iArgVec. Return -1 if not found.
        const int getIndex(std::string pcOptShort, std::string pcOptLong, const bool& pcNext)
        {
            pcOptShort = "-" + pcOptShort;
            pcOptLong = "--" + pcOptLong;
            
            for ( int i=0; i<iArgVec.size(); i++)
            {
                if (iArgVec[i] == pcOptShort || iArgVec[i] == pcOptLong)
                {
                    if (pcNext)
                    {
                        if (i + 1 >= iArgVec.size())
                        {
                            cout << color::set(color::RED) << "Command line argument search failed. No argument provided after flag \'" << pcOptShort << "\' or \'" << pcOptLong << "\'." << color::set(color::RESET) << endl;
                        }
                        else
                        {
                            return i + 1;
                        }
                    }
                    else
                    {
                        return i;
                    }
                }
            }
            
            return -1;
        }
        
    }; // end anonymous namespace


    
    // See usage documentation in header
    void init(const int& pcArgc, char* pcArgv[])
    {
        // Skip argv[0] because that's just the file name
        for ( int i=1; i<pcArgc; i++)
        {
            char pushStr[strlen(pcArgv[i])];
            strcpy(pushStr, pcArgv[i]);
            iArgVec.push_back(string(pushStr));
        }
    }



    // See usage documentation in header
    template<typename T>
    const T get(const std::string& pcOptShort, const T& pcDefault, const std::string& pcOptLong)
    {
        cout << color::set(color::RED) << "Command line argument search failed. Invalid typename for flag \'" << pcOptShort << "\' or \'" << pcOptLong << "\'." << color::set(color::RESET) << endl;
        return nullptr;
    }



    // Template specializations
    template<>
    const string get<string>(const std::string& pcOptShort, const string& pcDefault, const std::string& pcOptLong)
    {
        int index = getIndex(pcOptShort, pcOptLong, true);
        if (index != -1)
            return iArgVec[index];
        else
            return pcDefault;
    }
    
    template<>
    const int get<int>(const std::string& pcOptShort, const int& pcDefault, const std::string& pcOptLong)
    {
        int index = getIndex(pcOptShort, pcOptLong, true);
        if (index != -1)
        {
            try
            {
                return stoi(iArgVec[index]);
            }
            catch (exception ex)
            {
                cout << color::set(color::RED) << "Command line argument parsing failed. Argument \'" << iArgVec[index] << "\' cannot be converted to integer." << color::set(color::RESET) << endl;
                return pcDefault;
            }
        }
        else
            return pcDefault;
    }
    
    template<>
    const RATETYPE get<RATETYPE>(const std::string& pcOptShort, const RATETYPE& pcDefault, const std::string& pcOptLong)
    {
        int index = getIndex(pcOptShort, pcOptLong, true);
        if (index != -1)
        {
            try
            {
                return stold(iArgVec[index]);
            }
            catch (exception ex)
            {
                cout << color::set(color::RED) << "Command line argument parsing failed. Argument \'" << iArgVec[index] << "\' cannot be converted to RATETYPE." << color::set(color::RESET) << endl;
                return pcDefault;
            }
        }
        else
            return pcDefault;
    }
    
    template<>
    const bool get<bool>(const std::string& pcOptShort, const bool& pcDefault, const std::string& pcOptLong)
    {
        if (getIndex(pcOptShort, pcOptLong, false) != -1) // If found
            return !pcDefault;
        else
            return pcDefault;
    }
}
