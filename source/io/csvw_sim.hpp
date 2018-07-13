#ifndef IO_CSVW_SIM_HPP
#define IO_CSVW_SIM_HPP

#include "csvw.hpp"
#include "sim/base.hpp"
#include "core/observable.hpp"
#include "core/specie.hpp"

#include <limits>

namespace dense {

class csvw_sim : private csvw, public Observer
<Simulation>
{
public:
    csvw_sim(std::string const& pcfFileName,
            bool pcfTimeColumn,
            specie_vec const& pcfSpecieOption, Simulation & observable);

    void update(dense::Context<> pfStart);

  protected:
    void when_updated_by(Simulation & observable) override;

private:
    specie_vec const icSpecieOption;
    Real const icTimeStart = 0, icTimeEnd = std::numeric_limits<Real>::infinity();
    unsigned const icCellTotal, icCellStart = 0, icCellEnd;
    unsigned ilCell = icCellStart;
    bool const icTimeColumn;
};

#endif

}
