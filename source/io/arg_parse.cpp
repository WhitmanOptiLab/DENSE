#include "arg_parse.hpp"
#include "util/color.hpp"

#include "core/specie.hpp" // For specie_vec specialization
#include "core/reaction.hpp" // For "typedef float RATETYPE;"

#include <algorithm> // For remove
#include <cfloat> // For FLT_MAX
#include <climits> // For INT_MAX
#include <cstring> // For strcpy in init
#include <iostream>
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
        
        
        // Get index of (or index after) pcFlag if it exists in iArgVec.
        // Return false if not found.
        const bool getIndex(string pcfFlagShort, string pcfFlagLong,
                int* pnIndex, const bool& pcfNext)
        {
            pcfFlagShort = "-" + pcfFlagShort;
            pcfFlagLong = "--" + pcfFlagLong;

            for (int i=0; i<iArgVec.size(); i++)
            {
                if (iArgVec[i] == pcfFlagShort || iArgVec[i] == pcfFlagLong)
                {
                    if (pcfNext)
                    {
                        if (i + 1 >= iArgVec.size())
                        {
                            cout << color::set(color::RED) << "Command line argument "
                                "search failed. No argument provided after flag [" <<
                                pcfFlagShort << " | " << pcfFlagLong << "]." <<
                                color::clear() << endl;
                        }
                        else
                        {
                            if (pnIndex)
                                *pnIndex = i + 1;
                            return true;
                        }
                    }
                    else
                    {
                        if (pnIndex)
                            *pnIndex = i;
                        return true;
                    }
                }
            }
            
            return false;
        }
        
        
        // Prints message warning that flag is required
        void warnObligatory(string pcfFlagShort, string pcfFlagLong)
        {
            if (!iSuppressObligatory)
            {
                cout << color::set(color::RED) << "Command line argument search "
                    "failed. Flag [-" << pcfFlagShort << " | --" <<
                    pcfFlagLong << "] is required in order for all program "
                    "behaviors to function properly." << color::clear() << endl;
            }
        }
        
        
        template<typename T>
        const T getNotOblig(const string& pcfFlagShort, const string& pcfFlagLong)
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
    
    
    
    
    template<>
    bool get<string>(const std::string& pcfFlagShort, const std::string& pcfFlagLong,
            string* pnPushTo, const bool& pcfObligatory)
    {
        int index;
        if (getIndex(pcfFlagShort, pcfFlagLong, &index, true))
        {
            if (pnPushTo)
                *pnPushTo = iArgVec[index];
            return true;
        }
        else
        {
            if (pcfObligatory)
                warnObligatory(pcfFlagShort, pcfFlagLong);
            return false;
        }
    }
    
    // The default is a vec filled with all specie ids
    template<>
    bool get<specie_vec>(const std::string& pcfFlagShort, const std::string& pcfFlagLong,
            specie_vec* pnPushTo, const bool& pcfObligatory)
    {
        string tArg = get<string>(pcfFlagShort, pcfFlagLong, "");
        tArg = "," + tArg + ",";
        tArg.erase(remove(tArg.begin(), tArg.end(), ' '), tArg.end());

        specie_vec rVec;
        rVec.reserve(NUM_SPECIES);
        for (unsigned int i=0; i<NUM_SPECIES; i++)
        {
            if (tArg.find(","+specie_str[i]+",") != string::npos || tArg == ",,")
            {
                rVec.push_back((specie_id) i);
            }
        }

        if (rVec.size() > 0)
        {
            if (pnPushTo)
                *pnPushTo = rVec;
            return true;
        }
        else
        {
            if (pcfObligatory)
                warnObligatory(pcfFlagShort, pcfFlagLong);
            return false;
        }
    }
    
    template<>
    bool get<int>(const string& pcfFlagShort, const string& pcfFlagLong,
            int* pnPushTo, const bool& pcfObligatory)
    {
        bool success = true;
        int index;
        if (getIndex(pcfFlagShort, pcfFlagLong, &index, true))
        {
            char* tInvalidAt;
            int tPushTo = strtol(iArgVec[index].c_str(), &tInvalidAt, 0);

            if (*tInvalidAt)
            {
                success = false;
                cout << color::set(color::RED) << "Command line argument parsing "
                    "failed. Argument \"" << iArgVec[index] << "\" cannot be "
                    "converted to integer." << color::clear() << endl;
            }
            else
            {
                *pnPushTo = tPushTo;
            }
        }
        else
        {
            if (pcfObligatory)
                warnObligatory(pcfFlagShort, pcfFlagLong);
            success = false;
        }
        
        return success;
    }
    
    template<>
    bool get<RATETYPE>(const string& pcfFlagShort, const string& pcfFlagLong,
            RATETYPE* pnPushTo, const bool& pcfObligatory)
    {
        bool success = true;
        int index;
        if (getIndex(pcfFlagShort, pcfFlagLong, &index, true))
        {
            char* tInvalidAt;
            RATETYPE tPushTo = strtold(iArgVec[index].c_str(), &tInvalidAt);

            if (*tInvalidAt)
            {
                success = false;
                cout << color::set(color::RED) << "Command line argument parsing "
                    "failed. Argument \"" << iArgVec[index] << "\" cannot be "
                    "converted to RATETYPE." << color::clear() << endl;
            }
            else
            {
                *pnPushTo = tPushTo;
            }
        }
        else
        {
            if (pcfObligatory)
                warnObligatory(pcfFlagShort, pcfFlagLong);
            success = false;
        }
        
        return success;
    }
    
    template<>
    bool get<bool>(const string& pcfFlagShort, const string& pcfFlagLong,
            bool* pnPushTo, const bool& pcfObligatory)
    {
        // true if found, false if not
        bool found = getIndex(pcfFlagShort, pcfFlagLong, 0, false);
        
        if (pnPushTo)
            *pnPushTo = found;
        return found;
    }
    
    
    
    /*
    template<>
    bool get<bool>(const string& pcfFlagShort, const string& pcfFlagLong,
            const bool& pcfDefault)
    {
        if (getIndex(pcfFlagShort, pcfFlagLong, 0, false)) // If found
            return !pcfDefault;
        else
            return pcfDefault;
    }
    */
}
