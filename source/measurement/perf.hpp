#ifndef ANLYS_PERF_HPP
#define ANLYS_PERF_HPP

#include "base.hpp"

#include <vector>
#include "details.hpp"
#include <stdexcept>
#include "../runtimecheck.hpp"

/*
* Subclass of Analysis superclass
* - records overall mins and maxs for each specie
* - records mins and maxs for each specie per cell
* - records overall specie averages
* - records specie averages per cell
*/
template <typename Simulation>
class PerfAnalysis : public Analysis<Simulation>{

  public:

    PerfAnalysis(std::vector<Species> const& observed_species,
  std::pair<dense::Natural, dense::Natural> cell_range,
  std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity()});

    PerfAnalysis() {t->set_begin();};
    PerfAnalysis (PerfAnalysis const& obj);

    /* Finalize: overloaded virtual function of Analysis
       - must be called to produce correct average values
     */

    void update(Simulation & simulation, std::ostream& log) override; 
    void finalize () override;
    void show (csvw * = nullptr) override;
 
    Details get_details() override ;
	//~PerfAnalysis();
		 
    PerfAnalysis* clone() const override {
      return new auto(*this);
    }

  private:

    double duration;
    bool finalized;
    Details detail;
    std::vector<Real> perf_vector;
    runtimecheck* t;
    std::vector<Species> ob;
    std::pair<dense::Natural, dense::Natural> cr;
    std::pair<Real, Real> tr;


};

#include "perf.ipp"

#endif
