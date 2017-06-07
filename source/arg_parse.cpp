#include "arg_parse.hpp"
#include "color.hpp"

#include "reaction.hpp" // For "typedef float RATETYPE;"

#include <cfloat> // For FLT_MAX
#include <climits> // For INT_MAX
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
        
        // Stops obligatory message
        bool iSuppressObligatory = false;
        
        
        // Get index of (or index after) pcFlag if it exists in iArgVec. Return -1 if not found.
        const int getIndex(std::string pcfFlagShort, std::string pcfFlagLong, const bool& pcfNext)
        {
            pcfFlagShort = "-" + pcfFlagShort;
            pcfFlagLong = "--" + pcfFlagLong;
            
            for ( int i=0; i<iArgVec.size(); i++)
            {
                if (iArgVec[i] == pcfFlagShort || iArgVec[i] == pcfFlagLong)
                {
                    if (pcfNext)
                    {
                        if (i + 1 >= iArgVec.size())
                        {
                            cout << color::set(color::RED) << "Command line argument search failed. No argument provided after flag \'-" << pcfFlagShort << "\' or \'--" << pcfFlagLong << "\'." << color::clear() << endl;
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
        void warnObligatory(std::string pcfFlagShort, std::string pcfFlagLong)
        {
            if (!iSuppressObligatory)
            {
                cout << color::set(color::RED) << "Command line argument search failed. Flag \'-" << pcfFlagShort << "\' or \'--" << pcfFlagLong << "\' is required in order for all program behaviors to function properly." << color::clear() << endl;
            }
        }
        
        
        template<typename T>
        const T getSuppressObligatory(const std::string& pcfFlagShort, const std::string& pcfFlagLong)
        {
            iSuppressObligatory = true;
            const T rval = get<T>(pcfFlagShort, pcfFlagLong);
            iSuppressObligatory = false;
            return rval;
        }
        
    }; // end anonymous namespace





    
    // See usage documentation in header
    void init(const int& pcfArgc, char* pcfArgv[])
    {
        // Skip argv[0] because that's just the file name
        for ( int i=1; i<pcfArgc; i++)
        {
            char hStr[strlen(pcfArgv[i])];
            strcpy(hStr, pcfArgv[i]);
            iArgVec.push_back(string(hStr));
        }
    }






    // See usage documentation in header
    template<typename T>
    const T get(const std::string& pcfFlagShort, const std::string& pcfFlagLong)
    {
        cout << color::set(color::RED) << "Command line argument search failed. Invalid typename for flag \'-" << pcfFlagShort << "\' or \'--" << pcfFlagLong << "\'." << color::clear() << endl;
        return nullptr;
    }
    
    template<typename T>
    const T get(const std::string& pcfFlagShort, const std::string& pcfFlagLong, const T& pcfDefault)
    {
        return get<T>(pcfFlagShort, pcfFlagLong);
    }



    // Template specializations
    template<>
    const string get<string>(const std::string& pcfFlagShort, const std::string& pcfFlagLong)
    {
        int index = getIndex(pcfFlagShort, pcfFlagLong, true);
        if (index != -1)
        {
            return iArgVec[index];
        }
        else
        {
            warnObligatory(pcfFlagShort, pcfFlagLong);
            return "";
        }
    }
    
    template<>
    const string get<string>(const std::string& pcfFlagShort, const std::string& pcfFlagLong, const string& pcfDefault)
    {
        string rval = getSuppressObligatory<string>(pcfFlagShort, pcfFlagLong);
        return rval != "" ? rval : pcfDefault;
    }
    
    
    
    template<>
    const int get<int>(const std::string& pcfFlagShort, const std::string& pcfFlagLong)
    {
        int index = getIndex(pcfFlagShort, pcfFlagLong, true);
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
        
        warnObligatory(pcfFlagShort, pcfFlagLong);
        return INT_MAX;
    }
    
    template<>
    const int get<int>(const std::string& pcfFlagShort, const std::string& pcfFlagLong, const int& pcfDefault)
    {
        int rval = getSuppressObligatory<int>(pcfFlagShort, pcfFlagLong);
        return rval != INT_MAX ? rval : pcfDefault;
    }
    
    
    
    template<>
    const RATETYPE get<RATETYPE>(const std::string& pcfFlagShort, const std::string& pcfFlagLong)
    {
        int index = getIndex(pcfFlagShort, pcfFlagLong, true);
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
        
        warnObligatory(pcfFlagShort, pcfFlagLong);
        return FLT_MAX;
    }
    
    template<>
    const RATETYPE get<RATETYPE>(const std::string& pcfFlagShort, const std::string& pcfFlagLong, const RATETYPE& pcfDefault)
    {
        RATETYPE rval = getSuppressObligatory<RATETYPE>(pcfFlagShort, pcfFlagLong);

        return rval != FLT_MAX ? rval : pcfDefault;
    }
    
    
    
    template<>
    const bool get<bool>(const std::string& pcfFlagShort, const std::string& pcfFlagLong)
    {
        if (getIndex(pcfFlagShort, pcfFlagLong, false) != -1) // If found
            return true;
        else
            return false;
    }
    
    template<>
    const bool get<bool>(const std::string& pcfFlagShort, const std::string& pcfFlagLong, const bool& pcfDefault)
    {
        if (getIndex(pcfFlagShort, pcfFlagLong, false) != -1) // If found
            return !pcfDefault;
        else
            return pcfDefault;
    }
}
