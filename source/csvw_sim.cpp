#include "csvw_sim.hpp"
#include "specie.hpp"

#include <string>
using namespace std;



csvw_sim::csvw_sim(const string& pcfFileName, Observable *pnObl) :
    Observer(pnObl), csvw(pcfFileName, true, "# This file can be used as a template for user-created data inputs under this particular model. It is recommended that \n")
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


void csvw_sim::finalize(ContextBase& pfStart)
{
    // Anything even need to be in here?
}


void csvw_sim::update(ContextBase& pfStart)
{
    if (pfStart.isValid())
    {
        for (int i=0; i<NUM_SPECIES; i++)
        {
            csvw::add_data(pfStart.getCon((specie_id) i));
        }
        
        csvw::add_div("\n");
    }
}
