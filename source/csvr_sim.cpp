#include "csvr_sim.hpp"

#include <iostream>
using namespace std;




csvr_sim::mini_ct::mini_ct() :
    iIter(0)
{

}


RATETYPE csvr_sim::mini_ct::getCon(specie_id sp) const
{
    return iRate[sp];
}


void csvr_sim::mini_ct::advance()
{
    iIter++;
}


bool csvr_sim::mini_ct::isValid() const
{
    return iIter < NUM_SPECIES;
}


void csvr_sim::mini_ct::reset()
{
    iIter = 0;
}




csvr_sim::csvr_sim(const string& pcfFileName) :
    csvr(pcfFileName)
{

}


csvr_sim::~csvr_sim()
{

}


void csvr_sim::run()
{
    int lID = 0;
    csvr_sim::mini_ct hMCT;
    while (csvr::get_next(&hMCT.iRate[lID++]))
    {
        if (lID >= NUM_SPECIES)
        {
            lID = 0;
            notify(hMCT);
        }
    }
}



