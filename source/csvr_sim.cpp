#include "csvr_sim.hpp"

#include <iostream>
using namespace std;


csvr_sim::csvr_sim(const string& pcfFileName) :
    csvr(pcfFileName)
{

}


csvr_sim::~csvr_sim()
{

}


void csvr_sim::run()
{
    specie_id lID = 0;
    csvr_sim::mini_ct hMCT;
    while (csvr::get_next(hMCT.iRate[lID++]))
    {
        cout << "[DEBUG:csvr_sim.cpp] get_next " << hMCT.iRate[lID-1] << endl;
        if (lID >= NUM_SPECIES)
        {
            cout << "[DEBUG:csvr_sim.cpp] notifying" << endl;
            lID = 0;
            notify(hMCT);
        }
    }
}


csvr_sim::mini_ct() :
    iIter(0)
{

}


RATETYPE csvr_sim::mini_ct::getCon(specie_id sp)
{
    return iRate[sp];
}


RATETYPE csvr_sim::mini_ct::advance()
{
    iIter++;
}


bool csvr_sim::mini_ct::isValid()
{
    return iIter < NUM_SPECIES;
}
