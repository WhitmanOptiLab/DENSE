#ifndef IO_CSVW_SIM_HPP
#define IO_CSVW_SIM_HPP

#include "csvw.hpp"
#include "sim/base.hpp"
#include "core/observable.hpp"
#include "core/specie.hpp"


class csvw_sim : private csvw, public Observer
{
public:
    csvw_sim(std::string const& pcfFileName, Real const& pcfTimeInterval,
            Real const& pcfTimeStart, Real const& pcfTimeEnd,
            bool const& pcfTimeColumn, const unsigned int& pcfCellTotal,
            const unsigned int& pcfCellStart, const unsigned int& pcfCellEnd,
            specie_vec const& pcfSpecieOption, Observable & observable);

    void update(dense::Context<> pfStart);

  protected:
    void when_updated_by(Observable & observable) override;

private:
    specie_vec const icSpecieOption;
    Real const icTimeInterval, icTimeStart, icTimeEnd;
    Real ilTime;
    unsigned const icCellTotal, icCellStart, icCellEnd;
    unsigned ilCell;
    bool const icTimeColumn;
};

#endif
