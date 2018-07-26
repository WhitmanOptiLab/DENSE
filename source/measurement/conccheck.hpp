#ifndef ANLYS_CONCCHECK_HPP
#define ANLYS_CONCCHECK_HPP

#include "base.hpp"
#include "bad_simulation_error.hpp"

class ConcentrationCheck : public Analysis {

  private:
    Real lower_bound, upper_bound;
    specie_id target_specie;

  public:
    ConcentrationCheck (
      unsigned min_cell, unsigned max_cell,
      Real lowerB, Real upperB,
      Real start_time, Real end_time,
      specie_id t_specie = static_cast<specie_id>(-1)
    ) :
      Analysis(min_cell, max_cell, start_time, end_time),
      lower_bound(lowerB), upper_bound(upperB),
      target_specie(t_specie) {
    };

    void update(Simulation & simulation, std::ostream&) override {
      for (unsigned c = min; c < max; ++c) {
        if (target_specie > -1) {
          check(simulation, c, target_specie);
        } else {
          for (Natural s = 0; s < NUM_SPECIES; ++s) {
            check(simulation, c, static_cast<Species>(s));
          }
        }
      }
    }

    void finalize() override {};

    ConcentrationCheck* clone() const override {
      return new auto(*this);
    }

  private:

    void check(Simulation & simulation, Natural cell, Species species) const {
      Real concentration = simulation.get_concentration(cell, species);
      if (concentration < lower_bound || concentration > upper_bound) {
        throw dense::Bad_Simulation_Error("Concentration out of bounds: [" +
          specie_str[species] + "] = " + std::to_string(concentration), simulation);
      }
    }

};
#endif
