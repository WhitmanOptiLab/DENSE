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
template <typename Simulation>
class BasicAnalysis : public Analysis<Simulation> {

  public:

    BasicAnalysis (
      std::vector<Species> const& species_vector,
      std::pair<dense::Natural, dense::Natural> cell_range,
      std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() }
    );

    /*
     * Update: repeatedly called by observable to notify that there is more data
     * - arg Context& start: reference to iterator over concentrations
     * - precondition: start.isValid() is true.
     * - postcondition: start.isValid() is false.
     * - update is overloaded virtual function of Analysis
     */
    void update (Simulation& simulation, std::ostream& log) override;

    /* Finalize: overloaded virtual function of Analysis
       - must be called to produce correct average values
     */
    void finalize () override;

    void show (csvw * = nullptr) override;

    BasicAnalysis* clone() const override {
      return new auto(*this);
    }

  private:

    std::vector<Real> mins, maxs, means;

    std::vector<std::vector<Real>> mins_by_context, maxs_by_context, means_by_context;

};

#include "basic.ipp"

#endif
