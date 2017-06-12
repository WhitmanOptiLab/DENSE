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

	int crit_tracker;
	vector<int> numPeaks,numTroughs,cycles;
	vector<RATETYPE> peakSum,troughSum,cycleSum;

	vector<multiset<RATETYPE> > bst;

	vector<RATETYPE> amplitudes;
	vector<RATETYPE> periods;

	void addCritPoint(int context, bool isPeak, RATETYPE minute, RATETYPE concentration);
	void get_peaks_and_troughs();
	void calcAmpsAndPers();

public:	
	OscillationAnalysis(DataLogger *dLog, RATETYPE loc_Range, specie_id specieID) : Analysis(dLog) {
		local_range = loc_Range;
		range_steps = local_range/dl->analysis_interval;
		s = specieID;
		crit_tracker = 0;
		for (int c=0; c<dl->contexts; c++){
			Queue q(range_steps);
			vector<crit_point> v;
			multiset<RATETYPE> BST;

			windows.push_back(q);
			peaksAndTroughs.push_back(v);
			bst.push_back(BST);
				
			amplitudes.push_back(0);
			periods.push_back(0);
			numPeaks.push_back(0);
			numTroughs.push_back(0);
			cycles.push_back(0);
			peakSum.push_back(0);
			troughSum.push_back(0);
			cycleSum.push_back(0);
		}
	}

	void testQueue(){
		for (int c=0; c<dl->contexts; c++){
			cout<<"CELL "<<c<<endl;
			cout<<"Amplitude = "<<amplitudes[c]<<"   Period = "<<periods[c]<<"min"<<endl;
		}			
	}

	void update();
};

class CorrelationAnalysis : public Analysis {

	CorrelationAnalysis(DataLogger *dLog) : Analysis(dLog) {
	}

	void update(){
	}

	bool pearsonCorrelate();
};

#endif
