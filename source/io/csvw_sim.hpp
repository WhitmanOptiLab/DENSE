#ifndef IO_CSVW_SIM_HPP
#define IO_CSVW_SIM_HPP

#include "csvw.hpp"
#include "sim/base.hpp"
#include "measurement/base.hpp"
#include "core/specie.hpp"

#include <limits>

namespace dense {

template <typename Simulation>
class csvw_sim : public Analysis<Simulation> {

  public:
    csvw_sim(Natural cell_total,
            bool pcfTimeColumn,
            std::vector<Species> const& pcfSpecieOption);

    csvw_sim(csvw_sim const& other)
    : Analysis<Simulation>(other)
    , icTimeColumn{other.icTimeColumn} {
    }

    void update(Simulation& simulation, std::ostream& log) override;

    void finalize() override {};

    csvw_sim* clone() const override {
      return new auto(*this);
    }

private:
    unsigned ilCell = this->min;
    bool icTimeColumn = true;
    bool print_header_ = true;
};

}

#include "csvw_sim.ipp"

#endif
