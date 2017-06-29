#include "csvw_param.hpp"
#include "color.hpp"
#include "specie.hpp"

#include <iostream>
#include <string>
using namespace std;

/*
// Gets the length of an array of type T
// Based on https://stackoverflow.com/a/3368894
template<typename T, const unsigned int rcSize>
const unsigned int len(const T(&)[rcSize])
{
    return rcSize;
}
*/

csvw_param::csvw_param(const string& pcfFileName) :
    csvw::csvw(pcfFileName, true, "# IMPORTANT: Rename this file to \"param_list.csv\" or similar once data has been entered!\n")
{
    const string cReaction[] = {
        #define REACTION(name) #name, 
        #define DELAY_REACTION(name) #name, 
        #include "reactions_list.hpp"
        #undef REACTION
        #undef DELAY_REACTION
    };
    
    const string cDelay[] = {
        #define REACTION(name) 
        #define DELAY_REACTION(name) #name, 
        #include "reactions_list.hpp"
        #undef REACTION
        #undef DELAY_REACTION
    };
    
    const string cCritical[] = {
        #define SPECIE(name)
        #define CRITICAL_SPECIE(name) #name, 
        #include "specie_list.hpp"
        #undef SPECIE
        #undef CRITICAL_SPECIE
    };
    

    // Write column headers
    for (unsigned int i=0; i<NUM_REACTIONS; i++)
    {
        csvw::add_div(cReaction[i] + ", ");
    }
    
    csvw::add_div(", ");
    
    for (unsigned int i=0; i<NUM_DELAY_REACTIONS; i++)
    {
        csvw::add_div("dreact_" + cDelay[i] + ", ");
    }
    
    csvw::add_div(", ");
    
    for (unsigned int i=0; i<NUM_CRITICAL_SPECIES; i++)
    {
        csvw::add_div("rcrit_" + cCritical[i] + ", ");
    }
    
    
    // Victory message
    cout << color::set(color::GREEN) << "CSV parameter list column header generation successful. See \'" << pcfFileName << "\'." << color::clear() << endl;
}


csvw_param::~csvw_param()
{
}
