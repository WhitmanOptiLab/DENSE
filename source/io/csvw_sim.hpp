#ifndef IO_CSVW_SIM_HPP
#define IO_CSVW_SIM_HPP

#include "csvw.hpp"
#include "core/context.hpp"
#include "core/observable.hpp"
#include "core/specie.hpp"


class csvw_sim : public csvw, public Observer
{
public:
    csvw_sim(const std::string& pcfFileName, const RATETYPE& pcfTimeInterval,
            const RATETYPE& pcfTimeStart, const RATETYPE& pcfTimeEnd,
            const bool& pcfTimeColumn, const unsigned int& pcfCellTotal,
            const unsigned int& pcfCellStart, const unsigned int& pcfCellEnd,
            const specie_vec& pcfSpecieOption, Observable *pnObl);
    virtual ~csvw_sim();
    
    virtual void finalize(ContextBase& pfStart);
    virtual void update(ContextBase& pfStart);

private:
    unsigned int ilCell;
    RATETYPE ilTime;
    const bool icTimeColumn;
    const RATETYPE icTimeInterval, icTimeStart, icTimeEnd;
    const specie_vec icSpecieOption;
    const unsigned int icCellTotal, icCellStart, icCellEnd;
};

#endif
