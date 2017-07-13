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

	vector<Queue> windows;

	vector<vector<crit_point> > peaksAndTroughs;

	int range_steps;
	RATETYPE analysis_interval;
	specie_id s;

	vector<multiset<RATETYPE> > bst;

	vector<RATETYPE> amplitudes;
	vector<RATETYPE> periods;

	void addCritPoint(int context, bool isPeak, RATETYPE minute, RATETYPE concentration);
	void get_peaks_and_troughs(const ContextBase& start,int c);
	void calcAmpsAndPers(int c);
	void checkCritPoint(int c);

public:	
	/*
	* Constructor: creates an oscillation analysis for a specific specie
	* arg *dLog: observable to collected data from
	* interval: frequency that OscillationAnalysis is updated, in minutes.
	* range: required local range of a peak or trough in minutes.
	* specieID: specie to analyze.
	*/
	OscillationAnalysis(Observable *dLog, RATETYPE interval,
                        RATETYPE range, specie_id specieID,
                        int min_cell, int max_cell,
                        RATETYPE startT, RATETYPE endT) : 
            Analysis(dLog,min_cell,max_cell,startT,endT), range_steps(range/interval), 
            analysis_interval(interval), s(specieID){
        
            for (int c=min; c<max; c++){ 
                 Queue q(range_steps);
		    	vector<crit_point> v;
		    	multiset<RATETYPE> BST;

    			windows.push_back(q);
	    		peaksAndTroughs.push_back(v);
	    		bst.push_back(BST);
				
	    		amplitudes.push_back(0);
	    		periods.push_back(0);
            }
	}

	//Test: print output.
	void test(){
		/*	
		for (int c=min; c<max; c++){
			cout<<"CELL "<<c<<endl;
			cout<<"Amplitude = "<<amplitudes[c]<<"   Period = "<<periods[c]<<"min"<<endl;
		}
		*/
		for (int p=0; p<peaksAndTroughs[0].size();p++){
			crit_point cp = peaksAndTroughs[0][p];
			string text;
			if (cp.is_peak){
				text = "Peak: ";
			}else{
				text = "Trough: ";
			}
			cout<<text<<cp.conc<<" at "<<cp.time<<"min"<<endl;
		}
		cout<<"Amplitude = "<<amplitudes[0]<<"   Period = "<<periods[0]<<"min"<<endl;
		cout<<endl<<endl;
	}

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

	CorrelationAnalysis(Observable *dLog,int mn, int mx, RATETYPE startT, 
            RATETYPE endT) : Analysis(dLog,mn,mx,startT,endT) {
	}

	void update(ContextBase& start){
	}

	bool pearsonCorrelate();
};

#endif
