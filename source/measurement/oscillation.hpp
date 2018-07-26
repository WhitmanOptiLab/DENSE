#ifndef ANLYS_OSCILLATION_HPP
#define ANLYS_OSCILLATION_HPP

#include "base.hpp"

#include <vector>
#include <set>
#include <string>
#include "core/queue.hpp"

/*
* OscillationAnalysis: Subclass of Analysis superclass
* - identifies time and concentration of peaks and troughs of a given specie
* - calculates average amplitude of oscillations per cell
* - calculates average period of oscillations per cell
*/
template <typename Simulation>
class OscillationAnalysis : public Analysis<Simulation> {

private:
	struct crit_point {
		Real time;
		Real conc;
		bool is_peak;
	};

	bool vectors_assigned;

    // Outer-most vector is "for each specie in observed_species_"

	std::vector<std::vector<Queue<Real>>> windows;

	std::vector<std::vector<std::vector<crit_point>>> peaksAndTroughs;

	int range_steps;
	Real analysis_interval;

	std::vector<std::vector<std::multiset<Real>>> bst;

	std::vector<std::vector<Real>> amplitudes;
	std::vector<std::vector<Real>> periods;

    // s: std::vector<Species> index
	void addCritPoint(int s, int context, crit_point crit);
	void get_peaks_and_troughs(Simulation const& simulation, int c);
	void calcAmpsAndPers(int s, int c);
	void checkCritPoint(int s, int c);

public:
	/*
	* Constructor: creates an oscillation analysis for a specific specie
	* arg *dLog: observable to collected data from
	* interval: frequency that OscillationAnalysis is updated, in minutes.
	* range: required local range of a peak or trough in minutes.
	* specieID: specie to analyze.
	*/
	OscillationAnalysis(Real interval,
                        Real range, std::vector<Species> const& pcfSpecieOption,
                        std::pair<dense::Natural, dense::Natural> cell_range,
                        std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() }) :
            Analysis<Simulation>(pcfSpecieOption, cell_range, time_range),
            range_steps(range/interval), analysis_interval(interval)
    {
        auto cell_count = Analysis<>::max - Analysis<>::min;
        for (std::size_t i = 0; i < Analysis<>::observed_species_.size(); ++i)
        {
            windows.emplace_back(cell_count, Queue<Real>(range_steps));
            peaksAndTroughs.emplace_back(cell_count);
            bst.emplace_back(cell_count);
            amplitudes.emplace_back(cell_count, 0.0);
            periods.emplace_back(cell_count, 0.0);
        }
	}

	virtual ~OscillationAnalysis() {}


	/*
	* Update: repeatedly called by observable to notify that there is more data
	* - arg Context& start: reference to iterator over concentrations
	* - precondition: start.isValid() is true.
	* - postcondition: start.isValid() is false.
	* - update is overloaded virtual function of Observer
	*/
	void update (Simulation& simulation, std::ostream& log) override;

	//Finalize: called by observable to signal end of data
	// - generates peaks and troughs in final slice of data.
	void finalize () override;

  void show (csvw * = nullptr) override;

  OscillationAnalysis* clone() const override {
    return new auto(*this);
  }

};

template <typename Simulation>
class CorrelationAnalysis : public Analysis<Simulation> {

  public:

    CorrelationAnalysis(
      std::vector<Species> const& pcfSpecieOption,
  std::pair<dense::Natural, dense::Natural> cell_range,
  std::pair<Real, Real> time_range = { 0, std::numeric_limits<Real>::infinity() })
    : Analysis<Simulation>(pcfSpecieOption, cell_range, time_range)
    {
    }

    void update(Simulation& simulation, std::ostream&) override {
    }

    bool pearson_correlate();
};

#include "oscillation.ipp"

#endif
