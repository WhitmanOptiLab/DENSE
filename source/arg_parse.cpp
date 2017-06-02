#include "arg_parse.hpp"
#include "color.hpp"

#include "reaction.hpp" // For "typedef float RATETYPE;"

#include <cfloat> // For FLT_MIN
#include <climits>
#include <cstring> // For INT_MIN
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
        
        // Stops obligatory message
        bool suppressObligatory = false;
        
        
        // Get index of (or index after) pcFlag if it exists in iArgVec. Return -1 if not found.
        const int getIndex(std::string pcFlagShort, std::string pcFlagLong, const bool& pcNext)
        {
            pcFlagShort = "-" + pcFlagShort;
            pcFlagLong = "--" + pcFlagLong;
            
            for ( int i=0; i<iArgVec.size(); i++)
            {
                if (iArgVec[i] == pcFlagShort || iArgVec[i] == pcFlagLong)
                {
                    if (pcNext)
                    {
                        if (i + 1 >= iArgVec.size())
                        {
                            cout << color::set(color::RED) << "Command line argument search failed. No argument provided after flag \'-" << pcFlagShort << "\' or \'--" << pcFlagLong << "\'." << color::clear() << endl;
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
        
        
        // Prints message warning that flag is required
        void warnObligatory(std::string pcFlagShort, std::string pcFlagLong)
        {
            if (!suppressObligatory)
            {
                cout << color::set(color::RED) << "Command line argument search failed. Flag \'-" << pcFlagShort << "\' or \'--" << pcFlagLong << "\' is required in order for the program to execute." << color::clear() << endl;
            }
        }
        
        
        template<typename T>
        const T getSuppressObligatory(const std::string& pcFlagShort, const std::string& pcFlagLong)
        {
            suppressObligatory = true;
            const T rval = get<T>(pcFlagShort, pcFlagLong);
            suppressObligatory = false;
            return rval;
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
    const T get(const std::string& pcFlagShort, const std::string& pcFlagLong)
    {
        cout << color::set(color::RED) << "Command line argument search failed. Invalid typename for flag \'-" << pcFlagShort << "\' or \'--" << pcFlagLong << "\'." << color::clear() << endl;
        return nullptr;
    }
    
    template<typename T>
    const T get(const std::string& pcFlagShort, const std::string& pcFlagLong, const T& pcDefault)
    {
        return get<T>(pcFlagShort, pcFlagLong);
    }



    // Template specializations
    template<>
    const string get<string>(const std::string& pcFlagShort, const std::string& pcFlagLong)
    {
        int index = getIndex(pcFlagShort, pcFlagLong, true);
        if (index != -1)
        {
            return iArgVec[index];
        }
        else
        {
            warnObligatory(pcFlagShort, pcFlagLong);
            return "";
        }
    }
    
    template<>
    const string get<string>(const std::string& pcFlagShort, const std::string& pcFlagLong, const string& pcDefault)
    {
        string rval = getSuppressObligatory<string>(pcFlagShort, pcFlagLong);
        return rval != "" ? rval : pcDefault;
    }
    
    
    
    template<>
    const int get<int>(const std::string& pcFlagShort, const std::string& pcFlagLong)
    {
        int index = getIndex(pcFlagShort, pcFlagLong, true);
        if (index != -1)
        {
            try
            {
                return stoi(iArgVec[index]);
            }
            catch (exception ex)
            {
                cout << color::set(color::RED) << "Command line argument parsing failed. Argument \'" << iArgVec[index] << "\' cannot be converted to integer." << color::clear() << endl;
            }
        }
        
        warnObligatory(pcFlagShort, pcFlagLong);
        return INT_MIN;
    }
    
    template<>
    const int get<int>(const std::string& pcFlagShort, const std::string& pcFlagLong, const int& pcDefault)
    {
        int rval = getSuppressObligatory<int>(pcFlagShort, pcFlagLong);
        return rval != INT_MIN ? rval : pcDefault;
    }
    
    
    
    template<>
    const RATETYPE get<RATETYPE>(const std::string& pcFlagShort, const std::string& pcFlagLong)
    {
        int index = getIndex(pcFlagShort, pcFlagLong, true);
        if (index != -1)
        {
            try
            {
                return stold(iArgVec[index]);
            }
            catch (exception ex)
            {
                cout << color::set(color::RED) << "Command line argument parsing failed. Argument \'" << iArgVec[index] << "\' cannot be converted to RATETYPE." << color::clear() << endl;
            }
        }
        
        warnObligatory(pcFlagShort, pcFlagLong);
        return FLT_MIN;
    }
    
    template<>
    const RATETYPE get<RATETYPE>(const std::string& pcFlagShort, const std::string& pcFlagLong, const RATETYPE& pcDefault)
    {
        RATETYPE rval = getSuppressObligatory<RATETYPE>(pcFlagShort, pcFlagLong);

	    return rval != FLT_MIN ? rval : pcDefault;
    }
    
    
    
    template<>
    const bool get<bool>(const std::string& pcFlagShort, const std::string& pcFlagLong)
    {
        if (getIndex(pcFlagShort, pcFlagLong, false) != -1) // If found
            return true;
        else
            return false;
    }
    
    template<>
    const bool get<bool>(const std::string& pcFlagShort, const std::string& pcFlagLong, const bool& pcDefault)
    {
        if (getIndex(pcFlagShort, pcFlagLong, false) != -1) // If found
            return !pcDefault;
        else
            return pcDefault;
    }
}
