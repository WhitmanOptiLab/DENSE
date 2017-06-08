#ifndef ANALYSIS_HPP
#define ANALYSIS_HPP

#include "datalogger.hpp"
#include "observable.hpp"
#include <vector>
#include <set>
#include <string>
#include "queue.hpp"

using namespace std;


class Analysis : public Observer {
protected:
	DataLogger *dl;
	int time;
public:
	Analysis(DataLogger *dLog) : Observer(dLog) {
		dl = dLog;

		time = 0;
	}

};


class BasicAnalysis : public Analysis {

	RATETYPE *averages;
	RATETYPE *mins;
	RATETYPE *maxs;
	RATETYPE **avgs_by_context;
	RATETYPE **mins_by_context;
	RATETYPE **maxs_by_context;


	BasicAnalysis(DataLogger *dLog) : Analysis(dLog) {
	
		averages = new RATETYPE[NUM_SPECIES];
		mins = new RATETYPE[NUM_SPECIES];
		maxs = new RATETYPE[NUM_SPECIES];
		avgs_by_context = new RATETYPE*[dl->sim->_cells_total];
		mins_by_context = new RATETYPE*[dl->sim->_cells_total];
		maxs_by_context = new RATETYPE*[dl->sim->_cells_total];	
	
		for (int c=0; c<dl->contexts; c++){
			
			avgs_by_context[c] = new RATETYPE[NUM_SPECIES];
			mins_by_context[c] = new RATETYPE[NUM_SPECIES];
			maxs_by_context[c] = new RATETYPE[NUM_SPECIES];

			for (int s=0; s<NUM_SPECIES; s++){
				mins_by_context[c][s] = 9999;
			}
		}
		for (int s=0; s<NUM_SPECIES; s++){
			mins[s] = 9999;
		}
		
		time = 0;

	}

	void update(){
		
		update_averages();
		update_minmax();
	/*		
		if (dl->sim_done){
		}
	*/
	}
	
	void update_averages();
	void update_minmax();
};

class OscillationAnalysis : public Analysis {

	struct crit_point {
		bool is_peak;
		RATETYPE time;
		RATETYPE conc;
	};

	vector<Queue> windows;

	vector<vector<crit_point> > peaksAndTroughs;

	RATETYPE local_range;
	int range_steps;
	specie_id s;

	vector<set<RATETYPE> > bst;

	void addCritPoint(int context, bool isPeak, RATETYPE minute, RATETYPE concentration);

	void initialize();

public:	
	OscillationAnalysis(DataLogger *dLog, RATETYPE loc_Range, specie_id specieID) : Analysis(dLog) {
		local_range = loc_Range;
		range_steps = local_range/dl->analysis_interval;
		s = specieID;
	}

	void testQueue(){
		cout<<s<<endl;
		for (int c=0; c<dl->contexts; c++){
			cout<<"CELL "<<c<<endl;
			for (int t=0; t<peaksAndTroughs[c].size(); t++){
				string text = "Trough: ";
				if (peaksAndTroughs[c][t].is_peak){
					text = "Peak: ";
				}
				cout<<text<<peaksAndTroughs[c][t].conc<<" at time "<<peaksAndTroughs[c][t].time<<endl;
			}
		}			
	}

	void update();

	void get_peaks_and_troughs();
	
	void calc_amplitudes();

	void calc_periods();

};

class CorrelationAnalysis : public Analysis {

	CorrelationAnalysis(DataLogger *dLog) : Analysis(dLog) {
	}

	void update(){
	}

	bool pearsonCorrelate();
};

#endif
