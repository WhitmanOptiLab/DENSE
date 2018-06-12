#ifndef ANLYS_CONCCHECK_HPP
#define ANLYS_CONCCHECK_HPP

#include "base.hpp"
#include "core/context.hpp"

class ConcentrationCheck : public Analysis {

  private:
    RATETYPE lower_bound, upper_bound;
    specie_id target_specie;

  public:
    ConcentrationCheck (
      Observable & observable,
      unsigned min_cell, unsigned max_cell,
      Real lowerB, Real upperB,
      Real start_time, Real end_time,
      specie_id t_specie = static_cast<specie_id>(-1)
    ) :
      Analysis(min_cell, max_cell, start_time, end_time),
      lower_bound(lowerB), upper_bound(upperB),
      target_specie(t_specie) {
      subscribe_to(observable);
    };

    void update (ContextBase & start) override {
        for (unsigned c = min; c < max; ++c) {
            if (target_specie > -1) {
                Real concentration = start.getCon(target_specie);
                if (con < lower_bound || con > upper_bound) {
                    subject->abort();
                }
            } else {
                for (unsigned s = 0; s < NUM_SPECIES; ++s){
                    Real con = start.getCon((specie_id) s);
                    if (con<lower_bound || con>upper_bound) {
                        subject->abort();
                    }
                }
            }
        }
    }

    void finalize() override {};
};
#endif
