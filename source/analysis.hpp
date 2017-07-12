#ifndef ANALYSIS_HPP
#define ANALYSIS_HPP

#include "observable.hpp"
#include <vector>
#include <set>
#include <string>
#include "queue.hpp"

using namespace std;

/*
* Superclass for Analysis Objects
* - observes passed "Observable"
* - does not implement any analysis
*/
class Analysis : public Observer {
protected:
	int time;
public:
	Analysis(Observable *dLog, int mn, int mx, RATETYPE startT, RATETYPE endT) 
        : Observer(dLog,mn,mx,startT,endT), time(0) {}

};

class ConcentrationCheck : public Analysis {

  private:
    RATETYPE lower_bound, upper_bound;
    specie_id target_specie;

  public:
    ConcentrationCheck(Observable *data_source, int min, int max, 
                        RATETYPE lowerB, RATETYPE upperB, RATETYPE startT, RATETYPE
                        endT, specie_id t_specie=(specie_id)-1) :
                        Analysis(data_source, min, max, startT, endT), target_specie(t_specie),
                        lower_bound(lowerB), upper_bound(upperB){}

    void update(ContextBase& start) {

        RATETYPE con;

        for (int c=min; c<max; c++){
            if (target_specie > -1){
                con = start.getCon(target_specie);
                if (con<lower_bound || con>upper_bound) {
                    subject->abort();
                }
            }else{
                for (int s=0; s<NUM_SPECIES; s++){
                    RATETYPE con = start.getCon((specie_id) s);
                    if (con<lower_bound || con>upper_bound) {
                        subject->abort();
                    }
                }
            }
        }
    }

    void finalize(){} 
};

/*
* Subclass of Analysis superclass
* - records overall mins and maxs for each specie
* - records mins and maxs for each specie per cell
* - records overall specie averages
* - records specie averages per cell
*/
class BasicAnalysis : public Analysis {

private:
	vector<RATETYPE> averages;
	vector<RATETYPE> mins;
	vector<RATETYPE> maxs;
	vector<vector<RATETYPE> > avgs_by_context;
	vector<vector<RATETYPE> > mins_by_context;
	vector<vector<RATETYPE> > maxs_by_context;

	void update_averages(const ContextBase& start,int c);
	void update_minmax(const ContextBase& start,int c);

public:
	BasicAnalysis(Observable *dLog, int mn, int mx, RATETYPE startT, RATETYPE endT) 
        : Analysis(dLog,mn,mx,startT,endT) {
	    for (int c=min; c<max; c++){
            avgs_by_context.emplace_back();
		    mins_by_context.emplace_back();
		    maxs_by_context.emplace_back();
		    for (int s = 0; s<NUM_SPECIES; s++){
			    mins_by_context[c].push_back(9999);
		    	maxs_by_context[c].push_back(0);
			    avgs_by_context[c].push_back(0);
			    if (c==0){
					mins.push_back(9999);
					maxs.push_back(0);
				  	averages.push_back(0);
				}
            }
        }
    }

	/*
	* Update: repeatedly called by observable to notify that there is more data
	* - arg ContextBase& start: reference to iterator over concentrations
	* - precondition: start.isValid() is true.
	* - postcondition: start.isValid() is false.
	* - update is overloaded virtual function of Observer
	*/
	void update(ContextBase& start){
		for (int c = min; c<max; c++){
			update_averages(start,c);
			update_minmax(start,c);
			start.advance();
		}
		time++;
	}

	// Test: prints output.
	void test(){
		for (int s=0; s<averages.size(); s++){
			cout<<"Specie "<<s<<endl<<"average="<<averages[s]<<endl;
			cout<<"minimum="<<mins[s]<<endl;
			cout<<"maximum="<<maxs[s]<<endl<<endl;
		}
	}

	/* Finalize: overloaded virtual function of observer
	   - does nothing.
	*/	
	void finalize(){
	}
	
};

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
