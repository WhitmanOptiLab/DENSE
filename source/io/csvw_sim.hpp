#ifndef IO_CSVW_SIM_HPP
#define IO_CSVW_SIM_HPP

#include "csvw.hpp"
#include "core/context.hpp"
#include "core/observable.hpp"
#include "core/specie.hpp"


class csvw_sim : public csvw, public Observer
{
public:
    csvw_sim(const std::string& pcfFileName, const RATETYPE& pcfTimeInterval, const unsigned int& pcfCellTotal, const specie_vec& pcfSpecieVec, Observable *pnObl);
    virtual ~csvw_sim();
    
    virtual void finalize(ContextBase& pfStart);
    virtual void update(ContextBase& pfStart);

private:
    specie_vec oSpecieVec;
    unsigned int iTimeCount;
    const unsigned int icCellTotal;
    const RATETYPE icTimeInterval;
};

#endif
