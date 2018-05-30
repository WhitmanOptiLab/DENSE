#ifndef ANLYS_BASIC_HPP
#define ANLYS_BASIC_HPP

#include "base.hpp"
#include <vector>

/*
* Subclass of Analysis superclass
* - records overall mins and maxs for each specie
* - records mins and maxs for each specie per cell
* - records overall specie averages
* - records specie averages per cell
*/
class BasicAnalysis : public Analysis {



public:
    BasicAnalysis (
      Observable * log,
      specie_vec const& species_vector,
      csvw * csv_writer,
      int min, int max,
      Real start_time, Real end_time
    );

    /*
     * Update: repeatedly called by observable to notify that there is more data
     * - arg ContextBase& start: reference to iterator over concentrations
     * - precondition: start.isValid() is true.
     * - postcondition: start.isValid() is false.
     * - update is overloaded virtual function of Observer
     */
    void update (ContextBase & begin) override;

    /* Finalize: overloaded virtual function of observer
       - must be called to produce correct average values
     */
    void finalize () override;

  private:

    std::vector<Real> mins, maxs, means;

    std::vector<std::vector<Real>> mins_by_context, maxs_by_context, means_by_context;

};

#endif
