#ifndef ANLYS_OSCILLATION_HPP
#define ANLYS_OSCILLATION_HPP

#include "base.hpp"
#include <vector>
#include <set>
#include <string>
#include "core/queue.hpp"
using namespace std;


/*
* OscillationAnalysis: Subclass of Analysis superclass
* - identifies time and concentration of peaks and troughs of a given specie
* - calculates average amplitude of oscillations per cell
* - calculates average period of oscillations per cell
*/
class OscillationAnalysis : public Analysis {

private:
	struct crit_point {
		bool is_peak;
		RATETYPE time;
		RATETYPE conc;
	};

	bool vectors_assigned;

    // Outer-most vector is "for each specie in ucSpecieOption"

	vector<  vector<Queue>  > windows;

	vector<  vector<vector<crit_point> >  > peaksAndTroughs;

	int range_steps;
	RATETYPE analysis_interval;

	vector<  vector<multiset<RATETYPE> >  > bst;

	vector<  vector<RATETYPE>  > amplitudes;
	vector<  vector<RATETYPE>  > periods;

    // s: specie_vec index
	void addCritPoint(int s, int context, bool isPeak, RATETYPE minute, RATETYPE concentration);
	void get_peaks_and_troughs(const ContextBase& start,int c);
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
	OscillationAnalysis(Observable *dLog, RATETYPE interval,
                        RATETYPE range, const specie_vec& pcfSpecieOption,
                        csvw* pnFileOut, int min_cell, int max_cell,
                        RATETYPE startT, RATETYPE endT) : 
            Analysis(dLog,pcfSpecieOption,pnFileOut,min_cell,max_cell,startT,endT),
            range_steps(range/interval), analysis_interval(interval)
    {
        for (int i=0; i<ucSpecieOption.size(); i++)
        {
            windows.emplace_back();
            peaksAndTroughs.emplace_back();
            bst.emplace_back();
            amplitudes.emplace_back();
            periods.emplace_back();
            for (int c=min; c<max; c++){ 
                Queue q(range_steps);
                vector<crit_point> v;
                multiset<RATETYPE> BST;

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
	* - arg ContextBase& start: reference to iterator over concentrations
	* - precondition: start.isValid() is true.
	* - postcondition: start.isValid() is false.
	* - update is overloaded virtual function of Observer
	*/
	void update(ContextBase& start);

	//Finalize: called by observable to signal end of data
	// - generates peaks and troughs in final slice of data.
	void finalize();
};

class CorrelationAnalysis : public Analysis {

	CorrelationAnalysis(Observable *dLog,const specie_vec& pcfSpecieOption,
            int mn, int mx, RATETYPE startT, RATETYPE endT) :
        Analysis(dLog,pcfSpecieOption, 0, mn,mx,startT,endT) {
	}

	void update(ContextBase& start){
	}

	bool pearsonCorrelate();
};

#endif
