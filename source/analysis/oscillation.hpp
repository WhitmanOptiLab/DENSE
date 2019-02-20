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
class OscillationAnalysis : public Analysis {

private:
	struct crit_point {
		Real time;
		Real conc;
		bool is_peak;
	};

	bool vectors_assigned;

    // Outer-most vector is "for each specie in observed_species_"

	std::vector<std::vector<Queue>> windows;

	std::vector<std::vector<std::vector<crit_point>>> peaksAndTroughs;

	int range_steps;
	Real analysis_interval;

	std::vector<std::vector<std::multiset<Real>>> bst;

	std::vector<std::vector<Real>> amplitudes;
	std::vector<std::vector<Real>> periods;

    // s: specie_vec index
	void addCritPoint(int s, int context, bool isPeak, Real minute, Real concentration);
	void get_peaks_and_troughs(dense::Context const& start, int c);
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
	OscillationAnalysis(Observable & observable, Real interval,
                        Real range, specie_vec const& pcfSpecieOption,
                        csvw* pnFileOut, unsigned min_cell, unsigned max_cell,
                        Real start_time, Real end_time) :
            Analysis(pcfSpecieOption,pnFileOut,min_cell,max_cell,start_time,end_time),
            range_steps(range/interval), analysis_interval(interval)
    {
      subscribe_to(observable);
        for (std::size_t i = 0; i < observed_species_.size(); ++i)
        {
            windows.emplace_back();
            peaksAndTroughs.emplace_back();
            bst.emplace_back();
            amplitudes.emplace_back();
            periods.emplace_back();
            for (unsigned c = min; c < max; ++c){
                Queue q(range_steps);
                std::vector<crit_point> v;
                std::multiset<Real> BST;

                windows[i].push_back(q);
                peaksAndTroughs[i].push_back(v);
                bst[i].push_back(BST);

                amplitudes[i].push_back(0);
                periods[i].push_back(0);
            }
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
	void update (dense::Context &) override;

	//Finalize: called by observable to signal end of data
	// - generates peaks and troughs in final slice of data.
	void finalize () override;

  void show () override;
};

class CorrelationAnalysis : public Analysis {

  public:

    CorrelationAnalysis(Observable *dLog,specie_vec const& pcfSpecieOption,
            unsigned min_cell, unsigned max_cell, Real start_time, Real end_time) :
        Analysis(pcfSpecieOption, 0, min_cell, max_cell, start_time, end_time)
    {
      subscribe_to(*dLog);
    }

    void update(dense::Context & start) {
    }

    bool pearsonCorrelate();
};

#endif
