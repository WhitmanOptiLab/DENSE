#include "csvw_sim.hpp"
#include "specie.hpp"

#include <string>
using namespace std;


/*
csvw_sim::csvw_sim(const string& pcfFileName) :
    csvw::csvw(pcfFileName, true)
{
    const string STR_ALL_SPECIES[NUM_SPECIES] = {
        #define SPECIE(name) #name, 
        #include "specie_list.hpp"
        #undef SPECIE
    };

    for (int i=0; i<NUM_SPECIES; i++)
    {
        csvw::add_div(STR_ALL_SPECIES[i] + ", ");
    }
    csvw::add_div("\n");
}


csvw_sim::~csvw_sim()
{
}


void csvw_sim::update(ContextBase& pfStart)
{
    while (pfStart.isValid())
    {
        for (int i=0; i<NUM_SPECIES; i++)
        {
            csvw::add_data(pfStart.getCon(i));
        }
        
        csvw::add_div("\n");
        pfStart.advance();
    }
}
*/
